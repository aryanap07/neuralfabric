"""Feature scalers: StandardScaler."""

from __future__ import annotations

from typing import Self

import numpy as np

from neuralfabric.base import BaseEstimator, TransformerMixin
from neuralfabric.core.tensor import Tensor


def _validate_2d(X: Tensor, name: str = "X") -> np.ndarray:
    """Return the underlying array of ``X`` after checking it is 2D and non-empty."""
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor.")

    if X.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")

    return X.data


def _check_n_features(X: np.ndarray, expected: int) -> None:
    if X.shape[1] != expected:
        raise ValueError(
            f"X has {X.shape[1]} features, but the scaler was fitted "
            f"with {expected} features."
        )


class StandardScaler(BaseEstimator, TransformerMixin):
    """
    Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample ``x`` is ``z = (x - mean) / std``, computed
    independently for each feature. The population standard deviation
    (``ddof=0``) is used. Features with zero variance are only centered, never
    divided by zero.

    Parameters
    ----------
    with_mean:
        Center the data before scaling.
    with_std:
        Scale the data to unit variance.

    Attributes
    ----------
    mean_:
        Per-feature mean seen during ``fit``.
    scale_:
        Per-feature divisor used by ``transform`` (the standard deviation, with
        zeros replaced by 1).
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std

        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: Tensor, y: Tensor | None = None) -> Self:
        data = _validate_2d(X)

        self.mean_ = data.mean(axis=0)

        std = data.std(axis=0)
        self.scale_ = np.where(std == 0, 1.0, std).astype(data.dtype)

        return self

    def transform(self, X: Tensor) -> Tensor:
        mean, scale = self._fitted_params()
        data = _validate_2d(X)
        _check_n_features(data, mean.shape[0])

        if self.with_mean:
            data = data - mean

        if self.with_std:
            data = data / scale

        return Tensor(data)

    def inverse_transform(self, X: Tensor) -> Tensor:
        """Undo ``transform``, mapping standardized data back to the original scale."""
        mean, scale = self._fitted_params()
        data = _validate_2d(X)
        _check_n_features(data, mean.shape[0])

        if self.with_std:
            data = data * scale

        if self.with_mean:
            data = data + mean

        return Tensor(data)

    def _fitted_params(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before use.")

        return self.mean_, self.scale_

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"with_mean={self.with_mean}, with_std={self.with_std})"
        )
