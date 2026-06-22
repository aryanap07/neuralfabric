from __future__ import annotations

from neuralfabric.core.tensor import Tensor


def accuracy_score(
    y_true: Tensor,
    y_pred: Tensor,
) -> float:
    correct = (y_true.data == y_pred.data).sum()

    return float(correct / y_true.data.shape[0])
