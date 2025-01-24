#
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor

__all__ = ["ClusterResult", "group_by_label_mean", "first_nonzero", "rm_kwargs"]


class ClusterResult(NamedTuple):
    """Named and typed result tuple for kmeans algorithms

    Args:
        labels: label for each sample in x
        centers: corresponding coordinates of cluster centers
        inertia: sum of squared distances of samples to their closest cluster center
        x_org: original x
        x_norm: normalized x which was used for cluster centers and labels
        k: number of clusters
        soft_assignment: assignment probabilities of soft kmeans
    """

    labels: LongTensor
    centers: Tensor
    inertia: Tensor
    x_org: Tensor
    x_norm: Tensor
    k: LongTensor
    soft_assignment: Optional[Tensor] = None


@torch.jit.script
def group_by_label_mean(
    x: Tensor,
    labels: Tensor,
    k_max_range: Tensor,
) -> Tensor:
    """Group samples in x by label
    and calculate grouped mean.

    Args:
        x: samples (BS, N, D)
        labels: label per sample (BS, M, N)
        k_max_range: range of max number if clusters (BS, K_max)

    Returns:

    """
    # main idea: https://stackoverflow.com/a/56155805
    assert isinstance(x, Tensor)
    assert isinstance(labels, Tensor)
    assert isinstance(k_max_range, Tensor)
    bs, n, d = x.size()
    bs_, m, n_ = labels.size()
    assert bs == bs_ and n == n_
    k_max = k_max_range.size(-1)
    M = (
        (
            labels[:, :, :, None].expand(bs, m, n, k_max)
            == k_max_range[:, None, None, :].expand(bs, m, n, k_max)
        )
        .permute(0, 1, 3, 2)
        .to(x.dtype)
    )
    M = F.normalize(M, p=1.0, dim=-1)
    return torch.matmul(M, x[:, None, :, :].expand(bs, m, n, d))


@torch.jit.script
def first_nonzero(x: Tensor, dim: int = -1) -> Tuple[Tensor, Tensor]:
    """Return idx of first positive (!) nonzero element
    of each row in 'dim' of tensor 'x'
    and a mask if such an element does exist.

    Returns:
        msk, idx
    """
    # from: https://discuss.pytorch.org/t/first-nonzero-index/24769/9
    assert isinstance(x, Tensor)
    if len(x.shape) > 1:
        assert dim == -1 or dim == len(x.shape) - 1
    nonz = x > 0
    return ((nonz.cumsum(dim) == 1) & nonz).max(dim)


def rm_kwargs(kwargs: Dict, keys: List):
    """Remove items corresponding to keys
    specified in 'keys' from kwargs dict."""
    keys_ = list(kwargs.keys())
    for k in keys:
        if k in keys_:
            del kwargs[k]
    return kwargs
