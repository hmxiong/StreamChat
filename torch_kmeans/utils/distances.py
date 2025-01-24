#
from typing import Optional, Union

import torch
from torch import Tensor

from .utils import rm_kwargs

__all__ = [
    "LpDistance",
    "DotProductSimilarity",
    "CosineSimilarity",
]

# the following code is mostly adapted from
# https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/src/pytorch_metric_learning/distances
# to work in an inductive setting and for mini-batches of instances


class BaseDistance(torch.nn.Module):
    """

    Args:
        normalize_embeddings: flag to normalize provided embeddings
                                before calculating distances
        p: the exponent value in the norm formulation. (default: 2)
        power: If not 1, each element of the distance/similarity
                matrix will be raised to this power.
        is_inverted: Should be set by child classes.
                        If False, then small values represent
                        embeddings that are close together.
                        If True, then large values represent
                        embeddings that are similar to each other.
    """

    def __init__(
        self,
        normalize_embeddings: bool = True,
        p: Union[int, float] = 2,
        power: Union[int, float] = 1,
        is_inverted: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.power = power
        self.is_inverted = is_inverted
        self._check_params()

    def _check_params(self):
        if not isinstance(self.normalize_embeddings, bool):
            raise ValueError(
                f"normalize_embeddings must be of type <bool>, "
                f"but got {type(self.normalize_embeddings)} instead."
            )
        if not (isinstance(self.p, (int, float))) or self.p <= 0:
            raise ValueError(f"p should be and int or float > 0, " f"but got {self.p}.")
        if not (isinstance(self.power, (int, float))) or self.power <= 0:
            raise ValueError(
                f"power should be and int or float > 0, " f"but got {self.power}."
            )
        if not isinstance(self.is_inverted, bool):
            raise ValueError(
                f"is_inverted must be of type <bool>, "
                f"but got {type(self.is_inverted)} instead."
            )

    def forward(self, query_emb: Tensor, ref_emb: Optional[Tensor] = None) -> Tensor:
        bs = query_emb.size(0)
        query_emb_normalized = self.maybe_normalize(query_emb, dim=-1)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb, dim=-1)
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat**self.power
        assert mat.size() == torch.Size((bs, query_emb.size(1), ref_emb.size(1)))
        return mat

    def normalize(self, embeddings: Tensor, dim: int = -1, **kwargs):
        return torch.nn.functional.normalize(embeddings, p=self.p, dim=dim, **kwargs)

    def get_norm(self, embeddings: Tensor, dim: int = -1, **kwargs):
        return torch.norm(embeddings, p=self.p, dim=dim, **kwargs)

    def compute_mat(
        self,
        query_emb: Tensor,
        ref_emb: Optional[Tensor],
    ) -> Tensor:
        raise NotImplementedError

    def pairwise_distance(
        self,
        query_emb: Tensor,
        ref_emb: Optional[Tensor],
    ) -> Tensor:
        raise NotImplementedError

    def maybe_normalize(self, embeddings: Tensor, dim: int = 1, **kwargs):
        if self.normalize_embeddings:
            return self.normalize(embeddings, dim=dim, **kwargs)
        return embeddings


class LpDistance(BaseDistance):
    def __init__(self, **kwargs):
        kwargs = rm_kwargs(kwargs, ["is_inverted"])
        super().__init__(is_inverted=False, **kwargs)
        assert not self.is_inverted

    def compute_mat(
        self, query_emb: Tensor, ref_emb: Optional[Tensor] = None
    ) -> Tensor:
        """Compute the batched p-norm distance between
        each pair of the two collections of row vectors."""
        if ref_emb is None:
            ref_emb = query_emb
        if query_emb.dtype == torch.float16:
            # cdist doesn't work for float16
            raise TypeError("LpDistance does not work for dtype=torch.float16")
        if len(query_emb.shape) == 2:
            query_emb = query_emb.unsqueeze(-1)
        if len(ref_emb.shape) == 2:
            ref_emb = ref_emb.unsqueeze(-1)
        assert len(query_emb.shape) == len(ref_emb.shape) == 3
        assert query_emb.size(-1) == ref_emb.size(-1) >= 1
        return torch.cdist(query_emb, ref_emb, p=self.p)

    def pairwise_distance(
        self,
        query_emb: Tensor,
        ref_emb: Tensor,
    ) -> Tensor:
        """Computes the pairwise distance between
        vectors v1, v2 using the p-norm"""
        return torch.nn.functional.pairwise_distance(query_emb, ref_emb, p=self.p)


class DotProductSimilarity(BaseDistance):
    def __init__(self, **kwargs):
        kwargs = rm_kwargs(kwargs, ["is_inverted"])
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def compute_mat(
        self,
        query_emb: Tensor,
        ref_emb: Tensor,
    ) -> Tensor:
        assert len(list(query_emb.size())) == len(list(ref_emb.size())) == 3
        return torch.matmul(query_emb, ref_emb.permute((0, 2, 1)))

    def pairwise_distance(
        self,
        query_emb: Tensor,
        ref_emb: Tensor,
    ) -> Tensor:
        return torch.sum(query_emb * ref_emb, dim=-1)


class CosineSimilarity(DotProductSimilarity):
    def __init__(self, **kwargs):
        kwargs = rm_kwargs(kwargs, ["is_inverted", "normalize_embeddings"])
        super().__init__(is_inverted=True, normalize_embeddings=True, **kwargs)
        assert self.is_inverted
        assert self.normalize_embeddings
