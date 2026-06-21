from __future__ import annotations

from neuralfabric.core.tensor import Tensor


class LinearRegression:
    """
    Ordinary Least Squares (OLS) Linear Regression
    optimized using Gradient Descent.
    """

    def __init__(
        self,
        lr: float = 0.01,
        epochs: int = 1000,
    ) -> None:
        self.lr = lr
        self.epochs = epochs

        self.weight: Tensor | None = None
        self.bias: Tensor | None = None

    def fit(
        self,
        X: Tensor,
        y: Tensor,
    ) -> LinearRegression:
        n_features = X.shape[1]

        self.weight = Tensor(
            [[0.0] for _ in range(n_features)],
            requires_grad=True,
        )

        self.bias = Tensor(
            [0.0],
            requires_grad=True,
        )

        for _ in range(self.epochs):
            predictions = X @ self.weight + self.bias

            loss = ((predictions - y) ** 2).mean()

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

    def predict(
        self,
        X: Tensor,
    ) -> Tensor:
        if self.weight is None or self.bias is None:
            raise RuntimeError("LinearRegression must be fitted before prediction.")

        return X @ self.weight + self.bias

    def score(
        self,
        X: Tensor,
        y: Tensor,
    ) -> float:
        predictions = self.predict(X)

        ss_res = ((y - predictions) ** 2).sum().item()

        ss_tot = ((y - y.mean()) ** 2).sum().item()

        return 1.0 - (ss_res / ss_tot)

    def parameters(self) -> list[Tensor]:
        if self.weight is None or self.bias is None:
            return []

        return [self.weight, self.bias]

    def __repr__(self) -> str:
        return "LinearRegression(" f"lr={self.lr}, " f"epochs={self.epochs})"
