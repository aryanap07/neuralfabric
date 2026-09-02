from __future__ import annotations

from typing import Self

from neuralfabric.base import BaseEstimator, ClassifierMixin
from neuralfabric.core.tensor import Tensor


class LogisticRegression(
    BaseEstimator,
    ClassifierMixin,
):
    """
    Binary Logistic Regression
    optimized using Gradient Descent.
    """

    def __init__(
        self,
        lr: float = 0.01,
        epochs: int = 1000,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive.")

        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        self.lr = lr
        self.epochs = epochs

        self.weight: Tensor | None = None
        self.bias: Tensor | None = None

    def fit(
        self,
        X: Tensor,
        y: Tensor,
    ) -> Self:
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D tensor.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        n_features = X.shape[1]

        self.weight = Tensor(
            [[0.0] for _ in range(n_features)],
            requires_grad=True,
        )

        self.bias = Tensor(
            0.0,
            requires_grad=True,
        )

        for _ in range(self.epochs):
            logits = X @ self.weight + self.bias

            predictions = logits.sigmoid().clip(
                1e-7,
                1.0 - 1e-7,
            )

            loss = -(y * predictions.log() + (1 - y) * (1 - predictions).log()).mean()

            self.weight.zero_grad()
            self.bias.zero_grad()

            loss.backward()

            weight_grad = self.weight.grad
            bias_grad = self.bias.grad

            assert weight_grad is not None
            assert bias_grad is not None

            self.weight.data -= self.lr * weight_grad
            self.bias.data -= self.lr * bias_grad

        return self

    def predict_proba(
        self,
        X: Tensor,
    ) -> Tensor:
        if self.weight is None or self.bias is None:
            raise RuntimeError("LogisticRegression must be fitted before prediction.")

        logits = X @ self.weight + self.bias

        return logits.sigmoid()

    def predict(
        self,
        X: Tensor,
        threshold: float = 0.5,
    ) -> Tensor:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        probabilities = self.predict_proba(X)

        return Tensor((probabilities.data >= threshold).astype(float))

    def parameters(self) -> list[Tensor]:
        if self.weight is None or self.bias is None:
            return []

        return [self.weight, self.bias]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.lr}, epochs={self.epochs})"
