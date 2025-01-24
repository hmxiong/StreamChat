#
from typing import Any, Optional, Tuple, Union
from warnings import warn

import torch
from torch import LongTensor, Tensor

from ..utils.distances import BaseDistance, CosineSimilarity
from .kmeans import KMeans

__all__ = ["SoftKMeans"]


class SoftKMeans(KMeans):
    """
    Implements differentiable soft k-means clustering.
    Method adapted from https://github.com/bwilder0/clusternet
    to support batches.

    Paper:
        Wilder et al., "End to End Learning and Optimization on Graphs" (NeurIPS'2019)

    Args:
        init_method: Method to initialize cluster centers: ['rnd', 'topk']
                        (default: 'rnd')
        num_init: Number of different initial starting configurations,
                    i.e. different sets of initial centers.
                    If >1 selects the best configuration before
                    propagating through fixpoint (default: 1).
        max_iter: Maximum number of iterations (default: 100).
        distance: batched distance evaluator (default: CosineSimilarity).
        p_norm: norm for lp distance (default: 1).
        normalize: id of method to use to normalize input. (default: 'unit').
        tol: Relative tolerance with regards to Frobenius norm of the difference
                    in the cluster centers of two consecutive iterations to
                    declare convergence. (default: 1e-4)
        n_clusters: Default number of clusters to use if not provided in call
                (optional, default: 8).
        verbose: Verbosity flag to print additional info (default: True).
        seed: Seed to fix random state for randomized center inits
                (default: True).
        temp: temperature for soft cluster assignments (default: 5.0).
        **kwargs: additional key word arguments for the distance function.

    """

    def __init__(
        self,
        init_method: str = "rnd",
        num_init: int = 1,
        max_iter: int = 100,
        distance: BaseDistance = CosineSimilarity,
        p_norm: int = 1,
        normalize: str = "unit",
        tol: float = 1e-5,
        n_clusters: Optional[int] = 8,
        verbose: bool = True,
        seed: Optional[int] = 123,
        temp: float = 5.0,
        **kwargs,
    ):
        super(SoftKMeans, self).__init__(
            init_method=init_method,
            num_init=num_init,
            max_iter=max_iter,
            distance=distance,
            p_norm=p_norm,
            tol=tol,
            normalize=normalize,
            n_clusters=n_clusters,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )
        self.temp = temp
        if self.temp <= 0.0:
            raise ValueError(f"temp should be > 0, but got {self.temp}.")
        if not self.distance.is_inverted:
            raise ValueError(
                "soft k-means requires inverted " "distance measure (i.e. similarity)."
            )

    def _cluster(
        self, x: Tensor, centers: Tensor, k: LongTensor, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Union[Tensor, Any]]:
        """
        Run soft version of Lloyd's k-means algorithm.

        Args:
            x: (BS, N, D)
            centers: (BS, num_init, k_max, D)
            k: (BS, )

        """
        bs, n, d = x.size()
        # mask centers for which  k < k_max with inf to get correct assignment
        k_max = torch.max(k).cpu().item()
        k_max_range = torch.arange(k_max, device=x.device)[None, :].expand(bs, -1)
        k_mask = k_max_range >= k[:, None]
        k_mask = k_mask[:, None, :].expand(bs, self.num_init, -1)

        # run soft k-means to convergence
        with torch.no_grad():
            for i in range(self.max_iter):
                centers[k_mask] = 0
                old_centers = centers.clone()
                # update
                centers = self._cluster_iter(x, centers)
                # calculate center shift
                if self.tol is not None:
                    shift = self._calculate_shift(centers, old_centers, p=self.p_norm)
                    if (shift < self.tol).all():
                        if self.verbose:
                            print(
                                f"Full batch converged at iteration "
                                f"{i + 1}/{self.max_iter} "
                                f"with center shifts = "
                                f"{shift.view(-1, self.num_init).mean(-1)}."
                            )
                        break

        if self.verbose and i == self.max_iter - 1:
            print(
                f"Full batch did not converge after {self.max_iter} "
                f"maximum iterations."
                f"\nThere were some center shifts in last iteration "
                f"larger than specified threshold {self.tol}: "
                f"\n{shift.view(-1, self.num_init).mean(-1)}"
            )

        if self.num_init > 1:
            centers[k_mask] = 0
            dist = self._pairwise_distance(x, centers)
            dist[k_mask[:, :, None, :].expand(bs, self.num_init, n, -1)] = float("-inf")
            best_init = torch.argmax(dist.sum(-1).sum(-1), dim=-1)
            b_idx = torch.arange(bs, device=x.device)
            centers = centers[b_idx, best_init].unsqueeze(1)
            k_mask = k_mask[b_idx, best_init].unsqueeze(1)

        # enable (approx.) grad computation in final iteration
        with torch.enable_grad():
            centers[k_mask] = 0
            centers = self._cluster_iter(x, centers.detach().clone())
            centers[k_mask] = 0
            dist = self._pairwise_distance(x, centers)
            dist = dist.clone()
            # mask probability for non-existing centers
            dist[k_mask[:, :, None, :].expand(bs, 1, n, -1)] = float("-inf")
            soft_assignment = torch.softmax(self.temp * dist, dim=-1)

        dist = dist.squeeze(1)
        centers = centers.squeeze(1)
        soft_assignment = soft_assignment.squeeze(1)

        # hard assignment via argmax of similarity value to each cluster center
        c_assign = torch.argmax(dist, dim=-1).squeeze(1)
        all_same = (c_assign == c_assign[:, 0].unsqueeze(-1)).all(-1)
        if all_same.any():
            warn(
                f"Distance to all cluster centers is the same for instance(s) "
                f"with idx: {all_same.nonzero().squeeze().cpu().numpy().tolist()}. "
                f"Assignment will be random!"
            )
            same_dist = dist[all_same]
            if self.seed is not None:
                gen = torch.Generator(device=x.device)
                gen.manual_seed(self.seed)
            else:
                gen = None
            c_assign[all_same] = torch.randint(
                low=0,
                high=k_max,
                size=same_dist.shape[:-1],
                generator=gen,
                device=x.device,
            )
        return (c_assign, centers, dist, soft_assignment)

    def _cluster_iter(self, x: Tensor, centers: Tensor) -> Tensor:
        # x: (BS, N, D), centers: (BS, num_init, K, D) -> dist: (BS, num_init, N, K)
        bs, n, d = x.size()
        _, num_init, k, _ = centers.size()
        dist = self._pairwise_distance(x, centers)
        # mask probability for non-existing centers with -inf
        msk = dist == 0  # | (dist == float("inf")) | torch.isnan(dist)
        dist = dist.clone()
        dist[msk] = float("-inf")
        # get soft cluster assignments
        c_assign = torch.softmax(self.temp * dist, dim=-1)
        per_cluster = c_assign.sum(dim=-2)
        # update cluster centers
        # (BS, num_init, N, K)
        # -> (BS, num_init, K, 1, N) @ (BS, num_init, K, N, D)
        # -> (BS, num_init, K, D)
        cluster_mean = (
            c_assign.permute(0, 1, 3, 2)[:, :, :, None, :]
            @ x[:, None, None, :, :].expand(bs, num_init, k, n, d)
        ).squeeze(-2)
        centers = torch.diag_embed(1.0 / (per_cluster + self.eps)) @ cluster_mean
        centers[msk.any(dim=-2)] = 0
        return centers

    def _assign(self, x: Tensor, centers: Tensor, **kwargs) -> LongTensor:
        dist = self._pairwise_distance(x, centers)
        # mask probability for non-existing centers with -inf
        msk = dist == 0
        dist = dist.clone()
        dist[msk] = float("-inf")
        # get soft cluster assignments
        return torch.argmax(dist, dim=-1)  # type: ignore
