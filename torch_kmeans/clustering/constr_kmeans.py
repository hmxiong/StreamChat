#
from typing import Any, Optional, Tuple, Union
from warnings import warn

# import numpy as np
import torch
from torch import LongTensor, Tensor

from ..utils.distances import BaseDistance, LpDistance
from ..utils.utils import first_nonzero, group_by_label_mean, rm_kwargs
from .kmeans import KMeans

__all__ = ["ConstrainedKMeans"]


class InfeasibilityError(Exception):
    """Raised when no feasible assignment could be found."""


class ConstrainedKMeans(KMeans):
    """
    Implements constrained k-means clustering.
    Priority implementation is based on the method of

    Paper:
         Geetha, S., G. Poonthalir, and P. T. Vanathi.
         "Improved k-means algorithm for capacitated clustering problem."
         INFOCOMP Journal of Computer Science 8.4 (2009)


    Args:
        init_method: Method to initialize cluster centers:
                        ['rnd', 'topk', 'k-means++', 'ckm++']
                        (default: 'rnd')
        num_init: Number of different initial starting configurations,
                    i.e. different sets of initial centers (default: 8).
        max_iter: Maximum number of iterations (default: 100).
        distance: batched distance evaluator (default: LpDistance).
        p_norm: norm for lp distance (default: 2).
        tol: Relative tolerance with regards to Frobenius norm of the difference
                    in the cluster centers of two consecutive iterations to
                    declare convergence. (default: 1e-4)
        n_clusters: Default number of clusters to use if not provided in call
                (optional, default: 8).
        verbose: Verbosity flag to print additional info (default: True).
        seed: Seed to fix random state for randomized center inits
                (default: 123).
        n_priority_trials_before_fall_back: Number of trials trying to assign
                                            samples to constrained clusters based
                                            on priority values before falling back
                                            to assigning the node with the highest
                                            weight to a cluster which can still
                                            accommodate it or the dummy cluster
                                            otherwise. (default: 5)
        raise_infeasible: if set to False, will only display a warning
                            instead of raising an error (default: True)
        **kwargs: additional key word arguments for the distance function.

    """

    INIT_METHODS = ["rnd", "k-means++", "topk", "ckm++"]
    NORM_METHODS = []

    def __init__(
        self,
        init_method: str = "rnd",
        num_init: int = 8,
        max_iter: int = 100,
        distance: BaseDistance = LpDistance,
        p_norm: int = 2,
        tol: float = 1e-4,
        n_clusters: Optional[int] = 8,
        verbose: bool = True,
        seed: Optional[int] = 123,
        n_priority_trials_before_fall_back: int = 5,
        raise_infeasible: bool = True,
        **kwargs,
    ):
        kwargs = rm_kwargs(kwargs, ["normalize"])
        super(ConstrainedKMeans, self).__init__(
            init_method=init_method,
            num_init=num_init,
            max_iter=max_iter,
            distance=distance,
            p_norm=p_norm,
            tol=tol,
            normalize=None,
            n_clusters=n_clusters,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )
        self.n_trials = n_priority_trials_before_fall_back
        self.raise_infeasible = raise_infeasible
        # check
        if self.n_trials <= 0:
            raise ValueError(f"n_trials should be > 0, " f"but got {self.n_trials}.")
        if self.distance.is_inverted:
            raise ValueError(
                "constrained k-means does not work " "for inverted distance measures."
            )
        if self.init_method == "topk" and self.num_init > 1:
            raise ValueError(
                "topk init method is deterministic and "
                "does not work with num_init > 1."
            )

    def _check_weights(
        self,
        weights,
        dims: Tuple,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        if not isinstance(weights, Tensor):
            raise TypeError(
                f"weights has to be a torch.Tensor " f"but got {type(weights)}."
            )
        if not ((0 < weights) & (weights <= 1)).all():
            raise ValueError(
                "weights must be positive and " "be normalized between [0, 1]"
            )
        bs, n, d = dims
        if len(weights.shape) == 2:
            if weights.size(0) != bs or weights.size(1) != n:
                raise ValueError(
                    f"weights needs to be of shape "
                    f"({bs}, {n}, ),"
                    f"but got {tuple(weights.shape)}."
                )
        else:
            raise ValueError(
                f"weights have unsupported shape of "
                f"{tuple(weights.shape)} "
                f"instead of ({bs}, {n})."
            )
        return weights.contiguous().to(dtype=dtype, device=device)

    def predict(self, x: Tensor, weights: Tensor, **kwargs) -> LongTensor:
        """Predict the closest cluster each sample in X belongs to.

        Args:
            x: input features/coordinates (BS, N, D)
            weights: normalized weight for each sample (BS, N)
            **kwargs: additional kwargs for assignment procedure

        Returns:
            batch tensor of cluster labels for each sample (BS, N)
        """
        assert self.is_fitted
        x = self._check_x(x)
        weights = self._check_weights(weights, dims=x.shape, dtype=x.dtype)
        k_mask, _ = self._get_kmask(self._result.k, num_init=1)
        return self._assign(
            x,
            centers=self._result.centers[:, None, :, :],
            weights=weights,
            k_mask=k_mask,
            **kwargs,
        )

    @torch.no_grad()
    def _center_init(self, x: Tensor, k: LongTensor, **kwargs):
        if self.init_method == "rnd":
            return self._init_rnd(x, k)
        elif self.init_method == "k-means++":
            return self._init_plus(x, k)
        elif self.init_method == "topk":
            return self._init_topk(x, k, **kwargs)
        elif self.init_method == "ckm++":
            return self._init_ckm_plus(x, k, **kwargs)
        else:
            raise ValueError(f"unknown initialization method: {self.init_method}.")

    def _init_topk(self, x: Tensor, k: LongTensor, weights: Tensor, **kwargs) -> Tensor:
        """Choose k nodes with largest weights as initial centers.

        Args:
            x: (BS, N, D)
            k: (BS, )
            weights: (BS, N)

        Returns:
            centers: (BS, num_init, k, D)

        """
        if self.num_init > 1:
            raise RuntimeError(
                "topk init method is deterministic and "
                "does not work with num_init > 1."
            )
        bs, n, d = x.size()
        k_max = torch.max(k).cpu().item()
        # sample from uniform in batch and for num_init runs
        idx = torch.topk(weights, k=k_max, dim=-1).indices
        return x.gather(
            index=idx.view(bs, -1)[:, :, None].expand(bs, -1, d), dim=1
        ).view(bs, self.num_init, k_max, d)

    def _init_ckm_plus(
        self, x: Tensor, k: LongTensor, weights: Tensor, **kwargs
    ) -> Tensor:
        """Choose initial centers via adapted k-means++ method
        which also considers the weights.

        Args:
            x: (BS, N, D)
            k: (BS, )

        Returns:
            centers: (BS, num_init, k, D)

        """
        bs, n, d = x.size()
        k_max = torch.max(k).cpu().item()

        if self.seed is not None:
            # make random init reproducible independent of current iteration,
            # which otherwise would step and change the torch generator state
            gen = torch.Generator(device=x.device)
            gen.manual_seed(self.seed)
        else:
            gen = None

        bsm = bs * self.num_init
        bsm_idx = torch.arange(bsm, device=x.device)
        centers = torch.empty((bsm, k_max, d), dtype=x.dtype, device=x.device)
        weights = weights[:, None, :].expand(bs, self.num_init, n).reshape(bsm, n)

        # TODO: implement selection of n local trials
        #  (select center out of trials which minimizes inertia)
        # n_local_trials = 2 + int(np.log(k_max))

        # select first center randomly
        assert n > self.num_init, (
            f"Number of samples must be larger than <num_init> "
            f"but got {n} <= {self.num_init}"
        )
        idx = torch.multinomial(
            torch.empty((bs, n), device=x.device, dtype=x.dtype).fill_(1 / n),
            num_samples=self.num_init,
            replacement=False,
            generator=gen,
        )
        centers[:, 0] = x.gather(index=idx[:, :, None].expand(-1, -1, d), dim=1).view(
            -1, d
        )
        msk = torch.zeros((bsm, n, k_max), dtype=torch.bool, device=x.device)
        msk[bsm_idx, idx.view(-1), 0] = True

        # select the remaining k-1 centers
        # The intuition behind this approach is that spreading out the
        # k initial cluster centers is a good thing: the first cluster
        # center is chosen uniformly at random from the data points that
        # are being clustered, after which each subsequent cluster center
        # is chosen from the remaining data points with probability
        # proportional to its squared distance from the point's closest
        # existing cluster center weighted by its weight.
        for nc in range(1, k_max):
            dist = self._pairwise_distance(
                x, centers[:, :nc].view(bs, self.num_init, -1, d)
            ).view(bsm, n, nc)
            pot = weights[:, :, None].expand(bsm, n, nc) * dist**2
            pot[msk[:, :, :nc]] = 0
            pot = pot.min(dim=-1).values
            idx = torch.multinomial(pot, 1, generator=gen).view(bs, self.num_init)
            centers[:, nc] = x.gather(
                index=idx[:, :, None].expand(-1, -1, d), dim=1
            ).view(-1, d)
            msk[bsm_idx, idx.view(-1), nc] = True

        return centers.view(bs, self.num_init, k_max, d)

    def _get_kmask(self, k: Tensor, num_init: int = 1) -> Tuple[Tensor, Tensor]:
        """Compute mask of number of clusters k for centers of each instance."""
        bs = k.size(0)
        # mask centers for which  k < k_max with inf to get correct assignment
        k_max = torch.max(k).cpu().item() + 1  # dummy cluster
        k_max_range = torch.arange(k_max, device=k.device)[None, :].expand(bs, -1)
        k_mask = k_max_range >= k[:, None]
        k_mask = k_mask[:, None, :].expand(bs, num_init, -1)
        return k_mask, k_max_range

    @torch.no_grad()
    def _cluster(
        self, x: Tensor, centers: Tensor, k: LongTensor, weights: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Union[Tensor, Any]]:
        """
        Execute main algorithm.

        Args:
            x: (BS, N, D)
            centers: (BS, num_init, k_max, D)
            k: (BS, )
            weights: normalized weights w.r.t. constraint of 1.0 (BS, N, )

        """
        weights = self._check_weights(
            weights, dims=x.shape, dtype=x.dtype, device=x.device
        )
        bs, n, d = x.size()
        # add dummy center at origin to assign nodes which cannot be assigned
        # at an intermediate point because they violate all capacities
        centers = torch.cat(
            (centers, torch.zeros((bs, self.num_init, 1, d), device=x.device)), dim=2
        )
        k_mask, k_max_range = self._get_kmask(k, num_init=self.num_init)

        for i in range(self.max_iter):
            centers[k_mask] = float("inf")
            old_centers = centers.clone()
            # get cluster assignments
            c_assign = self._assign(x, centers, weights, k_mask)
            # update cluster centers
            centers = group_by_label_mean(x, c_assign, k_max_range)
            if self.tol is not None:
                # calculate center shift
                shift = self._calculate_shift(centers, old_centers, p=self.p_norm)
                if (shift < self.tol).all():
                    if self.verbose:
                        print(
                            f"Full batch converged at iteration "
                            f"{i + 1}/{self.max_iter} "
                            f"with center shifts: "
                            f"\n{shift.view(-1, self.num_init).mean(-1)}."
                        )
                    break

        if self.verbose and i == self.max_iter - 1:
            print(
                f"Full batch did not converge after "
                f"{self.max_iter} maximum iterations."
                f"\nThere were some center shifts in last iteration "
                f"larger than specified threshold {self.tol}: "
                f"\n{shift.view(-1, self.num_init).mean(-1)}"
            )

        # select best rnd restart according to inertia
        centers[k_mask] = float("inf")
        c_assign = self._assign(x, centers, weights, k_mask)
        if (c_assign < 0).any():
            # There remain some dummy clusters after convergence.
            # This means the algorithm could not find a
            # feasible assignment for at least one init
            # Check if there is at least 1 feasible solution for each instance
            feasible = (c_assign >= 0).all(-1).any(-1)
            if not feasible.all():
                inf_idx = (feasible == 0).nonzero().squeeze()
                msg = (
                    f"No feasible assignment found for "
                    f"instance(s) with idx: {inf_idx}.\n"
                    f"(Try to increase the number of clusters "
                    f"or loosen the constraints.)"
                )
                if self.raise_infeasible:
                    raise InfeasibilityError(msg)
                else:
                    warn(msg + "\nInfeasible instances removed from output.")
                    x = x[feasible]
                    centers = centers[feasible]
                    c_assign = c_assign[feasible]
                    bs = feasible.sum()

            # at least one init produced a feasible assignment
            # replace infeasible inits with feasible dummies to compute inertia
            feasible = (c_assign >= 0).all(-1)
            valid, dummy_row_idx = first_nonzero(feasible)
            assgn = (
                c_assign[torch.arange(bs, device=x.device), dummy_row_idx][:, None, :]
                .expand(c_assign.size())
                .contiguous()
            )
            assgn[feasible] = c_assign[feasible]
            c_assign = assgn

        inertia = self._calculate_inertia(x, centers, c_assign)
        best_init = torch.argmin(inertia, dim=-1)
        b_idx = torch.arange(bs, device=x.device)
        return (
            c_assign[b_idx, best_init],
            centers[b_idx, best_init],
            inertia[b_idx, best_init],
            None,
        )

    def _assign(
        self, x: Tensor, centers: Tensor, weights: Tensor, k_mask: Tensor, **kwargs
    ) -> LongTensor:
        # dist: (bs, num_init, n, k_max)
        dist = self._pairwise_distance(x, centers)
        bs, num_init, n, k_max = dist.size()
        bsm = bs * num_init
        dist = dist.view(bsm, n, k_max)

        # we use a heuristic approach to include the
        # cluster capacity by defining a priority value w.r.t. the weight
        # (demand, workload, etc.) of each point
        # The idea is to first assign points with a
        # relatively larger weight to the clusters
        # and then points with smaller weight which
        # can be more easily assigned to other clusters.
        weights = weights[:, None, :].expand(bs, num_init, n).reshape(bsm, n)
        priority = weights[:, :, None].expand(bsm, n, k_max) / dist
        priority[
            k_mask[:, :, None, :].expand(bs, num_init, n, k_max).reshape(bsm, n, k_max)
        ] = 0
        # loop over all nodes to sequentially assign them to clusters
        # while keeping track of cluster capacity
        assignment = -torch.ones((bsm, n), device=x.device, dtype=torch.long)
        cluster_capacity = torch.ones(k_mask.size(), device=x.device)
        cluster_capacity[k_mask] = 0
        cluster_capacity = cluster_capacity.view(bsm, k_max)
        for i in range(n):
            ##n_trials = min(n-i, self.n_trials)  # noqa
            n_trials = self.n_trials
            max_val_k, max_idx_k = priority.max(dim=-1)

            # select n_trials top priority nodes for each instance
            max_idx_n = max_val_k.topk(dim=-1, k=n_trials).indices
            # get corresponding cluster idx and weight
            cl_try = max_idx_k.gather(index=max_idx_n, dim=-1)
            w_try = weights.gather(index=max_idx_n, dim=-1)
            can_be_assigned = cluster_capacity.gather(index=cl_try, dim=-1) >= w_try
            # get first nonzero as idx and a validity mask
            # if any trial could be assigned
            valid_idx, fnz = first_nonzero(can_be_assigned, dim=-1)
            trial_select = fnz[valid_idx]
            cl_select = cl_try[valid_idx, trial_select]
            # do assignment
            n_select = max_idx_n[valid_idx, trial_select]
            assignment[valid_idx, n_select] = cl_select
            # mask priority of assigned nodes
            priority[valid_idx, n_select] = 0
            # adjust cluster capacity
            cur_cap = cluster_capacity[valid_idx, cl_select].clone()
            cluster_capacity[valid_idx, cl_select] = (
                cur_cap - w_try[valid_idx, trial_select]
            )
            # all instances with no valid idx could not assign any trial node
            not_assigned = ~valid_idx

            if not_assigned.any():
                # complete current assignment where for some instances
                # all trials based on priority were not feasible,
                # by assigning the node with the highest weight
                # to a cluster which can still accommodate it
                # or the dummy cluster at the origin otherwise (idx = -1)
                n_not_assigned = not_assigned.sum()
                cur_cap = cluster_capacity[not_assigned].clone()
                available = assignment[not_assigned] < 0
                # select node with highest weight from remaining unassigned nodes
                try:
                    w = weights[not_assigned][available].view(n_not_assigned, -1)
                except RuntimeError:
                    # fallback: just select best of first min available clusters
                    sm = available.sum(-1)
                    min_av = sm.min()
                    av_msk = sm > min_av
                    if min_av <= 1:
                        av_valid, av_idx = first_nonzero(available)
                        available[av_msk] = False
                        available[av_msk, av_idx[av_msk]] = True
                    else:
                        avbl = available[av_msk]
                        bi_cp = 0
                        cnter = 0
                        for bi, zi in zip(*avbl.nonzero(as_tuple=True)):
                            if bi == bi_cp:
                                cnter += 1
                            else:
                                bi_cp += 1
                                cnter = 1
                            if cnter > min_av:
                                avbl[bi, zi] = False
                        available[av_msk] = avbl
                    w = weights[not_assigned][available].view(n_not_assigned, -1)

                max_w, max_idx = w.max(dim=-1, keepdims=True)
                max_idx_n = (
                    available.nonzero(as_tuple=True)[1]
                    .view(n_not_assigned, -1)
                    .gather(dim=-1, index=max_idx)
                    .squeeze(-1)
                )
                # check the cluster priorities of this node
                msk = cur_cap >= max_w
                n_prio_idx = (
                    priority[not_assigned, max_idx_n]
                    .sort(dim=-1, descending=True)
                    .indices
                )
                # reorder msk according to priority and select first valid index
                select_msk = msk.gather(index=n_prio_idx, dim=-1)
                # get first nonzero as idx
                valid_idx, fnz = first_nonzero(select_msk, dim=-1)
                # nodes which cannot be assigned to any cluster anymore
                # since no sufficient capacity is available
                # are assigned to a dummy cluster with idx -1.
                cl_select = -torch.ones(
                    n_not_assigned, device=x.device, dtype=torch.long
                )
                cl_select[valid_idx] = n_prio_idx[valid_idx, fnz[valid_idx]]
                # do assignment
                assignment[not_assigned, max_idx_n] = cl_select
                # adapt priority
                priority[not_assigned, max_idx_n] = 0
                # adjust cluster capacity
                cur_cap = cluster_capacity[not_assigned, cl_select].clone()
                cluster_capacity[not_assigned, cl_select] = cur_cap - max_w.squeeze(-1)

        return assignment.view(bs, num_init, n)
