import pytest

from neuralfabric.core.tensor import Tensor
from neuralfabric.linear_model.linear_regression import LinearRegression


def test_fit_learns_simple_linear_relationship():
    """
    y = 2x + 1
    """
    X = Tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ]
    )

    y = Tensor(
        [
            [3.0],
            [5.0],
            [7.0],
            [9.0],
        ]
    )

    model = LinearRegression(
        lr=0.01,
        epochs=5000,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    for pred, target in zip(
        predictions.data.flatten(),
        y.data.flatten(),
    ):
        assert abs(pred - target) < 0.1


def test_predict_before_fit_raises_error():
    model = LinearRegression()

    X = Tensor([[1.0]])

    with pytest.raises(
        RuntimeError,
        match="must be fitted before prediction",
    ):
        model.predict(X)


def test_parameters_exist_after_fit():
    X = Tensor(
        [
            [1.0],
            [2.0],
        ]
    )

    y = Tensor(
        [
            [2.0],
            [4.0],
        ]
    )

    model = LinearRegression()
    model.fit(X, y)

    params = model.parameters()

    assert len(params) == 2
    assert params[0] is model.weight
    assert params[1] is model.bias


def test_prediction_shape():
    X = Tensor(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    y = Tensor(
        [
            [2.0],
            [4.0],
            [6.0],
        ]
    )

    model = LinearRegression(
        lr=0.01,
        epochs=3000,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape


def test_score_returns_high_r2():
    X = Tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ]
    )

    y = Tensor(
        [
            [3.0],
            [5.0],
            [7.0],
            [9.0],
        ]
    )

    model = LinearRegression(
        lr=0.01,
        epochs=5000,
    )

    model.fit(X, y)

    score = model.score(X, y)

    assert score > 0.99


def test_repr():
    model = LinearRegression(
        lr=0.1,
        epochs=100,
    )

    assert repr(model) == "LinearRegression(lr=0.1, epochs=100)"
