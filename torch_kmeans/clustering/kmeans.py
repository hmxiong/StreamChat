#
from typing import Any, Optional, Tuple, Union
from warnings import warn

import torch
import torch.nn as nn
from torch import LongTensor, Tensor

from ..utils.distances import (
    BaseDistance,
    CosineSimilarity,
    DotProductSimilarity,
    LpDistance,
)
from ..utils.utils import ClusterResult, group_by_label_mean

# import numpy as np
# from sklearn.cluster._kmeans import _kmeans_plusplus, row_norms

__all__ = ["KMeans"]


#
class KMeans(nn.Module):
    """
    Implements k-means clustering in terms of
    pytorch tensor operations which can be run on GPU.
    Supports batches of instances for use in
    batched training (e.g. for neural networks).

    Partly based on ideas from:
        - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        - https://github.com/overshiki/kmeans_pytorch


    Args:
            init_method: Method to initialize cluster centers ['rnd', 'k-means++']
                            (default: 'rnd')
            num_init: Number of different initial starting configurations,
                        i.e. different sets of initial centers (default: 8).
            max_iter: Maximum number of iterations (default: 100).
            distance: batched distance evaluator (default: LpDistance).
            p_norm: norm for lp distance (default: 2).
            tol: Relative tolerance with regards to Frobenius norm of the difference
                        in the cluster centers of two consecutive iterations to
                        declare convergence. (default: 1e-4)
            normalize: String id of method to use to normalize input.
                        one of ['mean', 'minmax', 'unit'].
                        None to disable normalization. (default: None).
            n_clusters: Default number of clusters to use if not provided in call
                    (optional, default: 8).
            verbose: Verbosity flag to print additional info (default: True).
            seed: Seed to fix random state for randomized center inits
                    (default: True).
            **kwargs: additional key word arguments for the distance function.
    """

    INIT_METHODS = ["rnd", "k-means++"]
    NORM_METHODS = ["mean", "minmax", "unit"]

    def __init__(
        self,
        init_method: str = "rnd",
        num_init: int = 8,
        max_iter: int = 100,
        distance: BaseDistance = LpDistance,
        p_norm: int = 2,
        tol: float = 1e-4,
        normalize: Optional[Union[str, bool]] = None,
        n_clusters: Optional[int] = 8,
        verbose: bool = True,
        seed: Optional[int] = 123,
        **kwargs,
    ):
        super(KMeans, self).__init__()
        self.init_method = init_method.lower()
        self.num_init = num_init
        self.max_iter = max_iter
        self.p_norm = p_norm
        self.tol = tol
        self.normalize = normalize
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.seed = seed

        self._check_params()
        self.distance = distance(p=self.p_norm, **kwargs)

        self.eps = None
        self._k_max = None
        self._result = None

    @property
    def is_fitted(self) -> bool:
        """True if model was already fitted."""
        return self._result is not None

    @property
    def num_clusters(self) -> Union[int, Tensor, Any]:
        """
        Number of clusters in fitted model.
        Returns a tensor with possibly different
        numbers of clusters per instance for whole batch.
        """
        if not self.is_fitted:
            return None
        return self._result.k

    def _check_params(self):
        if self.init_method not in self.INIT_METHODS:
            raise ValueError(
                f"unknown <init_method>: {self.init_method}. "
                f"Please choose one of {self.INIT_METHODS}"
            )
        if self.num_init <= 0:
            raise ValueError(f"num_init should be > 0, but got {self.num_init}.")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, but got {self.max_iter}.")
        if self.p_norm <= 0:
            raise ValueError(f"p_norm should be > 0, but got {self.p_norm}.")
        if self.tol < 0 or self.tol > 1:
            raise ValueError(f"tol should be > 0 and < 1, but got {self.tol}.")
        if isinstance(self.normalize, bool):
            if self.normalize:
                self.normalize = "mean"
            else:
                self.normalize = None
        if self.normalize is not None and self.normalize not in self.NORM_METHODS:
            raise ValueError(
                f"unknown <normalize> method: {self.normalize}. "
                f"Please choose one of {self.NORM_METHODS}"
            )
        if self.n_clusters is not None and self.n_clusters < 2:
            raise ValueError(f"n_clusters should be > 1, but got {self.n_clusters}.")

    def _check_x(self, x) -> Tensor:
        """Check and (re-)format input samples x."""
        if not isinstance(x, Tensor):
            raise TypeError(f"x has to be a torch.Tensor but got {type(x)}.")
        shp = x.shape
        if len(shp) < 3:
            raise ValueError(
                f"input <x> should be at least of shape (BS, N, D) "
                f"with batch size BS, number of points N "
                f"and number of dimensions D but got {shp}."
            )
        elif len(shp) > 3:
            x = x.squeeze()
            x = self._check_x(x)
        self.eps = torch.finfo(x.dtype).eps
        return x

    def _check_k(
        self, k, dims: Tuple, device: torch.device = torch.device("cpu")
    ) -> LongTensor:
        """Check and (re-)format number of clusters k."""
        bs, n, d = dims
        if not isinstance(k, Tensor):
            if k is None:  # use specified default number of clusters
                if self.n_clusters is None:
                    raise ValueError(
                        "Did not provide number of clusters k on call and "
                        "did not specify default 'n_clusters' at initialization."
                    )
                k = self.n_clusters
            if isinstance(k, int):  # convert to tensor
                k = torch.empty(bs, dtype=torch.long).fill_(k)
            else:
                raise TypeError(
                    f"k has to be int, torch.Tensor or None " f"but got {type(k)}."
                )
        if len(k.shape) > 1:
            k = k.squeeze()
            assert len(k.shape) == 1
        if k.shape[0] == 1:
            k = k.repeat(bs)
        if (k >= n).any():
            raise ValueError(
                f"Specified 'k' must be smaller than "
                f"number of samples n={n}, but got: {k}."
            )
        if (k <= 1).any():
            raise ValueError("Clustering for k=1 is ambiguous.")
        self._k_max = int(k.max())
        return k.to(dtype=torch.long, device=device)

    def _check_centers(
        self,
        centers,
        dims: Tuple,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        if not isinstance(centers, Tensor):
            raise TypeError(
                f"centers has to be a torch.Tensor " f"but got {type(centers)}."
            )
        bs, n, d = dims
        if len(centers.shape) == 3:
            if (
                centers.size(0) != bs
                or centers.size(1) != self._k_max
                or centers.size(2) != d
            ):
                raise ValueError(
                    f"centers needs to be of shape "
                    f"({bs}, {self._k_max}, {d}),"
                    f"but got {tuple(centers.shape)}."
                )
            if self.num_init > 1:
                warn(
                    f"Specified num_init={self.num_init} > 1 but provided "
                    f"only 1 center configuration per instance. "
                    f"Using same center configuration for all {self.num_init} runs."
                )
                # expand to num_init size
                centers = centers[:, None, :, :].expand(
                    centers.size(0), self.num_init, centers.size(1), centers.size(2)
                )
            else:
                centers = centers.unsqueeze(1)
        elif len(centers.shape) == 4:
            if (
                centers.size(0) != bs
                or centers.size(1) != self.num_init
                or centers.size(2) != self._k_max
                or centers.size(3) != d
            ):
                raise ValueError(
                    f"centers needs to be of shape "
                    f"({bs}, {self.num_init}, {self._k_max}, {d}),"
                    f"but got {tuple(centers.shape)}."
                )
        else:
            raise ValueError(
                f"centers have unsupported shape of "
                f"{tuple(centers.shape)} "
                f"instead of "
                f"({bs}, {self.num_init}, {self._k_max}, {d})."
            )
        return centers.contiguous().to(dtype=dtype, device=device)

    def forward(
        self,
        x: Tensor,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        centers: Optional[Tensor] = None,
        **kwargs,
    ) -> ClusterResult:
        """torch.nn like forward pass.

        Args:
            x: input features/coordinates (BS, N, D)
            k: optional batch of (possibly different)
                numbers of clusters per instance (BS, )
            centers: optional batch of initial centers to use (BS, K, D)
            **kwargs: additional kwargs for initialization or cluster procedure

        Returns:
            ClusterResult tuple

        """
        x = self._check_x(x)
        x_ = x
        k = self._check_k(k, dims=x.shape, device=x.device)

        # normalize input
        if self.normalize is not None:
            x = self._normalize(x, self.normalize, self.eps)
        # init centers
        if centers is None:
            centers = self._center_init(x, k, **kwargs)
        centers = self._check_centers(
            centers, dims=x.shape, dtype=x.dtype, device=x.device
        )

        labels, new_centers, inertia, soft_assign = self._cluster(
            x, centers, k, **kwargs
        )
        return ClusterResult(
            labels=labels,  # type: ignore
            centers=new_centers,
            inertia=inertia,
            x_org=x_,
            x_norm=x,
            k=k,
            soft_assignment=soft_assign,
        )

    def fit(
        self,
        x: Tensor,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        centers: Optional[Tensor] = None,
        **kwargs,
    ) -> nn.Module:
        """Compute cluster centers and predict cluster index for each sample.

        Args:
            x: input features/coordinates (BS, N, D)
            k: optional batch of (possibly different)
                numbers of clusters per instance (BS, )
            centers: optional batch of initial centers to use (BS, K, D)
            **kwargs: additional kwargs for initialization or cluster procedure

        Returns:
            KMeans model
        """
        self._result = self(x, k=k, centers=centers, **kwargs)
        return self

    def predict(self, x: Tensor, **kwargs) -> LongTensor:
        """Predict the closest cluster each sample in X belongs to.

        Args:
            x: input features/coordinates (BS, N, D)
            **kwargs: additional kwargs for assignment procedure

        Returns:
            batch tensor of cluster labels for each sample (BS, N)

        """
        assert self.is_fitted
        x = self._check_x(x)
        return self._assign(
            x, centers=self._result.centers[:, None, :, :], **kwargs
        ).squeeze(1)

    def fit_predict(
        self,
        x: Tensor,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        centers: Optional[Tensor] = None,
        **kwargs,
    ) -> LongTensor:
        """Compute cluster centers and predict cluster index for each sample.

        Args:
            x: input features/coordinates (BS, N, D)
            k: optional batch of (possibly different)
                numbers of clusters per instance (BS, )
            centers: optional batch of initial centers to use (BS, K, D)
            **kwargs: additional kwargs for initialization or cluster procedure

        Returns:
            batch tensor of cluster labels for each sample (BS, N)

        """
        return self(x, k=k, centers=centers, **kwargs).labels

    @torch.no_grad()
    def _center_init(self, x: Tensor, k: LongTensor, **kwargs) -> Tensor:
        """Wrapper to apply different methods for
        initialization of initial centers (centroids)."""
        if self.init_method == "rnd":
            return self._init_rnd(x, k)
        elif self.init_method == "k-means++":
            return self._init_plus(x, k)
        else:
            raise ValueError(f"unknown initialization method: {self.init_method}.")

    @staticmethod
    def _normalize(x: Tensor, normalize: str, eps: float = 1e-8):
        """Normalize input samples x according to specified method:

        - mean: subtract sample mean
        - minmax: min-max normalization subtracting sample min and divide by sample max
        - unit: normalize x to lie on D-dimensional unit sphere

        """
        if normalize == "mean":
            x -= x.mean(dim=1)[:, None, :]
        elif normalize == "minmax":
            x -= x.min(-1, keepdims=True).values  # type: ignore
            x /= x.max(-1, keepdims=True).values  # type: ignore
        elif normalize == "unit":
            # normalize x to unit sphere
            z_msk = x == 0
            x = x.clone()
            x[z_msk] = eps
            x = torch.diag_embed(1.0 / (torch.norm(x, p=2, dim=-1))) @ x
        else:
            raise ValueError(f"unknown normalization type {normalize}.")
        return x

    def _init_rnd(self, x: Tensor, k: LongTensor) -> Tensor:
        """Choose k random nodes as initial centers.

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

        # sample from uniform in batch and for num_init runs
        rnd_idx = torch.multinomial(
            torch.empty((bs * self.num_init, n), device=x.device, dtype=x.dtype).fill_(
                1 / n
            ),
            num_samples=k_max,
            replacement=False,
            generator=gen,
        )
        return x.gather(
            index=rnd_idx.view(bs, -1)[:, :, None].expand(bs, -1, d), dim=1
        ).view(bs, self.num_init, k_max, d)

    def _init_skl_plus(self, x: Tensor, k: LongTensor) -> Tensor:
        """Choose initial centers via kmeans++ method.
        https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/cluster/_kmeans.py#L50

        Args:
            x: (BS, N, D)
            k: (BS, )

        Returns:
            centers: (BS, num_init, k, D)

        """
        raise NotImplementedError
        # would require sklearn as additional dependency

        # bs, n, d = x.size()
        # k_max = torch.max(k).cpu().item()
        # rs = np.random.RandomState(self.seed if self.seed is not None else 1)
        # device = x.device
        # x = x.cpu().numpy()
        # k = k.cpu().numpy()
        # centers = []
        # for smp, nc in zip(x, k):
        #     center_inits = []
        #     x_squared_norms = row_norms(smp, squared=True)
        #     for i in range(self.num_init):
        #         c = np.zeros((k_max, d))
        #         c_init, _ = _kmeans_plusplus(
        #             smp, nc, random_state=rs, x_squared_norms=x_squared_norms
        #         )
        #         c[:nc] = c_init
        #         center_inits.append(c)
        #     centers.append(torch.from_numpy(np.stack(center_inits)))
        #
        # return torch.stack(centers).to(device)

    def _init_plus(self, x: Tensor, k: LongTensor) -> Tensor:
        """Choose initial centers via k-means++ method

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
        for nc in range(1, k_max):
            dist = self._pairwise_distance(
                x, centers[:, :nc].view(bs, self.num_init, -1, d)
            ).view(bsm, n, nc)
            pot = dist**2
            pot[msk[:, :, :nc]] = 0
            pot = pot.min(dim=-1).values
            idx = torch.multinomial(pot, 1, generator=gen).view(bs, self.num_init)
            centers[:, nc] = x.gather(
                index=idx[:, :, None].expand(-1, -1, d), dim=1
            ).view(-1, d)
            msk[bsm_idx, idx.view(-1), nc] = True

        return centers.view(bs, self.num_init, k_max, d)

    @torch.no_grad()
    def _cluster(
        self, x: Tensor, centers: Tensor, k: LongTensor, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Union[Tensor, Any]]:
        """
        Run Lloyd's k-means algorithm.

        Args:
            x: (BS, N, D)
            centers: (BS, num_init, k_max, D)
            k: (BS, )

        """
        if not isinstance(self.distance, LpDistance):
            warn("standard k-means should use a non-inverted distance measure.")
        bs, n, d = x.size()
        # mask centers for which  k < k_max with inf to get correct assignment
        k_max = torch.max(k).cpu().item()
        k_max_range = torch.arange(k_max, device=x.device)[None, :].expand(bs, -1)
        k_mask = k_max_range >= k[:, None]
        k_mask = k_mask[:, None, :].expand(bs, self.num_init, -1)

        for i in range(self.max_iter):
            centers[k_mask] = float("inf")
            old_centers = centers.clone()
            # get cluster assignments
            c_assign = self._assign(x, centers)
            # update cluster centers
            centers = group_by_label_mean(x, c_assign, k_max_range)
            if self.tol is not None:
                # calculate center shift
                shift = self._calculate_shift(centers, old_centers, p=self.p_norm)
                if (shift < self.tol).all():
                    if self.verbose:
                        print(
                            f"Full batch converged at iteration "
                            f"{i+1}/{self.max_iter} "
                            f"with center shifts = "
                            f"{shift.view(-1, self.num_init).mean(-1)}."
                        )
                    break

        # select best rnd restart according to inertia
        centers[k_mask] = float("inf")
        c_assign = self._assign(x, centers)
        inertia = self._calculate_inertia(x, centers, c_assign)
        best_init = torch.argmin(inertia, dim=-1)
        b_idx = torch.arange(bs, device=x.device)

        return (
            c_assign[b_idx, best_init],
            centers[b_idx, best_init],
            inertia[b_idx, best_init],
            None,
        )

    def _pairwise_distance(self, x: Tensor, centers: Tensor, **kwargs):
        """Calculate pairwise distances between samples in x and all centers."""
        # expand tensors to calculate pairwise distance over (d) dimensions
        # of each point (n) to each center (k_max)
        # for each random restart (num_init) in each batch instance (bs)
        bs, n, d = x.size()
        bs, num_init, k_max, d = centers.size()
        x = x[:, None, :, None, :].expand(bs, num_init, n, k_max, d).reshape(-1, d)
        centers = (
            centers[:, :, None, :, :].expand(bs, num_init, n, k_max, d).reshape(-1, d)
        )
        return self.distance.pairwise_distance(x, centers, **kwargs).view(
            bs, num_init, n, k_max
        )

    def _assign(self, x: Tensor, centers: Tensor, **kwargs) -> LongTensor:
        """Infer cluster assignment for each sample in x."""
        # dist: (bs, num_init, n, k_max)
        dist = self._pairwise_distance(x, centers)
        if isinstance(self.distance, (CosineSimilarity, DotProductSimilarity)):
            # Similarity is an inverted distance measure,
            # so we need to adapt it in order to calculate priority
            dist = 1 - dist

        # get cluster assignments (center with minimal distance)
        return torch.argmin(dist, dim=-1)  # type: ignore

    @staticmethod
    @torch.jit.script
    def _calculate_shift(centers: Tensor, old_centers: Tensor, p: int = 2) -> Tensor:
        """Calculate center shift w.r.t. centers from last iteration."""
        # calculate euclidean distance while replacing inf with 0 in sum
        d = torch.norm((centers - old_centers), p=p, dim=-1)
        d[d == float("inf")] = 0
        # sum(d, dim=-1)**2 -> use mean to be independent of number of points
        return torch.mean(d, dim=-1)

    @staticmethod
    @torch.jit.script
    def _calculate_inertia(x: Tensor, centers: Tensor, labels: Tensor) -> Tensor:
        """Compute sum of squared distances of samples
        to their closest cluster center."""
        bs, n, d = x.size()
        m = centers.size(1)
        assert m == labels.size(1)
        # select assigned center by label and calculate squared distance
        assigned_centers = centers.gather(
            index=labels[:, :, :, None].expand(
                labels.size(0), labels.size(1), labels.size(2), d
            ),
            dim=2,
        )
        # squared distance to closest center
        d = (
            torch.norm(
                (x[:, None, :, :].expand(bs, m, n, d) - assigned_centers), p=2, dim=-1
            )
            ** 2
        )
        d[d == float("inf")] = 0
        return torch.sum(d, dim=-1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"init: '{self.init_method}', "
            f"num_init: {self.num_init}, "
            f"max_iter: {self.max_iter}, "
            f"distance: {self.distance}, "
            f"tolerance: {self.tol}, "
            f"normalize: {self.normalize}"
            f")"
        )
