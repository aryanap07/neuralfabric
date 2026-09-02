import pytest

from neuralfabric.core.tensor import Tensor
from neuralfabric.linear_model import LinearRegression


def test_fit_learns_simple_linear_relationship():
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
        strict=True,
    ):
        assert abs(pred - target) < 0.1


def test_predict_before_fit_raises_error():
    model = LinearRegression()

    with pytest.raises(
        RuntimeError,
        match="must be fitted before prediction",
    ):
        model.predict(Tensor([[1.0]]))


def test_parameters_before_fit():
    model = LinearRegression()

    assert model.parameters() == []


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

    assert model.score(X, y) > 0.99


def test_invalid_learning_rate():
    with pytest.raises(
        ValueError,
        match="lr must be positive",
    ):
        LinearRegression(lr=0.0)


def test_invalid_epochs():
    with pytest.raises(
        ValueError,
        match="epochs must be positive",
    ):
        LinearRegression(epochs=0)


def test_fit_requires_2d_input():
    X = Tensor([1.0, 2.0, 3.0])

    y = Tensor(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    model = LinearRegression()

    with pytest.raises(
        ValueError,
        match="X must be a 2D tensor",
    ):
        model.fit(X, y)


def test_fit_requires_matching_samples():
    X = Tensor(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    y = Tensor(
        [
            [1.0],
            [2.0],
        ]
    )

    model = LinearRegression()

    with pytest.raises(
        ValueError,
        match="same number of samples",
    ):
        model.fit(X, y)


def test_repr():
    model = LinearRegression(
        lr=0.1,
        epochs=100,
    )

    assert repr(model) == "LinearRegression(lr=0.1, epochs=100)"
