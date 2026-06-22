from __future__ import annotations

from neuralfabric.core.tensor import Tensor


def r2_score(
    y_true: Tensor,
    y_pred: Tensor,
) -> float:
    y_true_data = y_true.data
    y_pred_data = y_pred.data

    ss_res = ((y_true_data - y_pred_data) ** 2).sum()
    ss_tot = ((y_true_data - y_true_data.mean()) ** 2).sum()

    if ss_tot == 0:
        return 0.0

    return float(1.0 - (ss_res / ss_tot))
