from __future__ import annotations

import numpy as np

from neuralfabric.core.tensor import Tensor


def train_test_split(
    X: Tensor,
    y: Tensor,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Split arrays or tensors into random train and test subsets.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    n_samples = X.shape[0]

    if y.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of samples.")

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    split_idx = int(n_samples * (1 - test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = Tensor(X.data[train_idx])
    X_test = Tensor(X.data[test_idx])

    y_train = Tensor(y.data[train_idx])
    y_test = Tensor(y.data[test_idx])

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )
