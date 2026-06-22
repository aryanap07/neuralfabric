from __future__ import annotations

import pytest

from neuralfabric.core.tensor import Tensor
from neuralfabric.linear_model.logistic_regression import LogisticRegression


def test_fit_learns_and_gate() -> None:
    X = Tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    y = Tensor(
        [
            [0.0],
            [0.0],
            [0.0],
            [1.0],
        ]
    )

    model = LogisticRegression(
        lr=0.1,
        epochs=5000,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert (predictions.data == y.data).all()


def test_predict_before_fit_raises_error() -> None:
    model = LogisticRegression()

    X = Tensor(
        [
            [0.0, 0.0],
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="must be fitted before prediction",
    ):
        model.predict(X)


def test_parameters_exist_after_fit() -> None:
    X = Tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    y = Tensor(
        [
            [0.0],
            [1.0],
        ]
    )

    model = LogisticRegression()
    model.fit(X, y)

    params = model.parameters()

    assert len(params) == 2
    assert params[0] is model.weight
    assert params[1] is model.bias


def test_prediction_shape() -> None:
    X = Tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    y = Tensor(
        [
            [0.0],
            [0.0],
            [0.0],
            [1.0],
        ]
    )

    model = LogisticRegression(
        lr=0.1,
        epochs=5000,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape


def test_score_returns_high_accuracy() -> None:
    X = Tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    y = Tensor(
        [
            [0.0],
            [0.0],
            [0.0],
            [1.0],
        ]
    )

    model = LogisticRegression(
        lr=0.1,
        epochs=5000,
    )

    model.fit(X, y)

    score = model.score(X, y)

    assert score == 1.0


def test_predict_proba_returns_values_between_zero_and_one() -> None:
    X = Tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    y = Tensor(
        [
            [0.0],
            [1.0],
        ]
    )

    model = LogisticRegression(
        lr=0.1,
        epochs=3000,
    )

    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert (probabilities.data >= 0.0).all()
    assert (probabilities.data <= 1.0).all()


def test_repr() -> None:
    model = LogisticRegression(
        lr=0.1,
        epochs=100,
    )

    assert repr(model) == "LogisticRegression(lr=0.1, epochs=100)"
