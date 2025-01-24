#
from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.distances import BaseDistance, LpDistance

__all__ = ["KNN"]


class KNeighbors(NamedTuple):
    """Named and typed result tuple for KNN search

    - distances: distance to each neighbor of each sample
    - indices: index of each neighbor of each sample
    - x_org: original x
    - x_norm: normalized x which was used for cluster centers and labels
    - k: number of neighbors

    """

    distances: Tensor
    indices: Tensor
    x_org: Tensor
    x_norm: Tensor
    k: Union[int, Tensor]


class KNN(nn.Module):
    """
    Implements k nearest neighbors in terms of
    pytorch tensor operations which can be run on GPU.
    Supports mini-batches of instances.

    Args:
        k: number of neighbors to consider
        distance: batched distance evaluator (default: LpDistance).
        p_norm: norm for lp distance (default: 2).
        normalize: String id of method to use to normalize input.
                        one of ['mean', 'minmax', 'unit'].
                        None to disable normalization. (default: None).

    """

    NORM_METHODS = ["mean", "minmax", "unit"]

    def __init__(
        self,
        k: int,
        distance: BaseDistance = LpDistance,
        p_norm: int = 2,
        normalize: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        super(KNN, self).__init__()
        self.k = k
        self.p_norm = p_norm
        self.normalize = normalize
        self._check_params()

        self.distance = distance(p=p_norm, **kwargs)
        self.eps = None

    def _check_params(self):
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError(f"k should be int > 0, but got {self.k}.")
        if self.p_norm <= 0:
            raise ValueError(f"p_norm should be > 0, but got {self.p_norm}.")
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

    def _check_k(self, k, dims: Optional[Tuple] = None) -> int:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k should be > 0, but got {k}.")
        if dims is not None:
            n = dims[1]
            if k >= n:
                raise ValueError(
                    f"k should be smaller than number " f"of samples {n} but got k={k}"
                )
        return k

    @torch.no_grad()
    def forward(
        self, x: Tensor, k: Optional[int] = None, same_source: bool = True
    ) -> KNeighbors:
        """torch.nn like forward pass.

        Args:
            x: input features/coordinates (BS, N, D)
            k: optional number of neighbors to use
            same_source: flag if each sample itself should be included
                            as its own neighbor (default: True)

        Returns:
            KNeighbors tuple

        """
        x = self._check_x(x)
        x_ = x
        k = self.k if k is None else k
        k = self._check_k(k, x.shape)
        # do not select self if from same source (instead of setting dist to inf)
        same_source = int(same_source)
        k += same_source
        # normalize input
        if self.normalize is not None:
            x = self._normalize(x, self.normalize, self.eps)

        values, indices = self.distance(x, x).sort(
            dim=-1, descending=self.distance.is_inverted
        )
        return KNeighbors(
            distances=values[:, :, same_source : k + 1],  # knn_distances
            indices=indices[:, :, same_source : k + 1],  # knn_indices
            x_org=x_,
            x_norm=x,
            k=k,
        )

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

    def fit(self, x: Tensor, k: Optional[int] = None, **kwargs) -> KNeighbors:
        """Compute k nearest neighbors for each sample.

        Args:
            x: input features/coordinates (BS, N, D)
            k: optional number of neighbors to use
            **kwargs: additional kwargs for fitting procedure

        Returns:
            KNeighbors tuple

        """
        return self(x, k=k, **kwargs)
