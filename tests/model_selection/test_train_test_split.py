import pytest

from neuralfabric.core.tensor import Tensor
from neuralfabric.model_selection import train_test_split


def test_split_shapes() -> None:
    X = Tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )

    y = Tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        shuffle=False,
    )

    assert X_train.shape == (3, 1)
    assert X_test.shape == (2, 1)

    assert y_train.shape == (3, 1)
    assert y_test.shape == (2, 1)


def test_invalid_test_size() -> None:
    X = Tensor([[1.0]])
    y = Tensor([[1.0]])

    with pytest.raises(
        ValueError,
        match="test_size must be between 0 and 1",
    ):
        train_test_split(
            X,
            y,
            test_size=1.0,
        )


def test_mismatched_samples() -> None:
    X = Tensor(
        [
            [1.0],
            [2.0],
        ]
    )

    y = Tensor(
        [
            [1.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="X and y must contain the same number of samples",
    ):
        train_test_split(X, y)


def test_random_state_reproducible() -> None:
    X = Tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )

    y = Tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )

    split_1 = train_test_split(
        X,
        y,
        random_state=42,
    )

    split_2 = train_test_split(
        X,
        y,
        random_state=42,
    )

    assert (split_1[0].data == split_2[0].data).all()

    assert (split_1[1].data == split_2[1].data).all()


def test_shuffle_false_preserves_order() -> None:
    X = Tensor(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )

    y = Tensor(
        [
            [10.0],
            [20.0],
            [30.0],
            [40.0],
            [50.0],
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        shuffle=False,
    )

    assert X_train.data.tolist() == [
        [1.0],
        [2.0],
        [3.0],
    ]

    assert X_test.data.tolist() == [
        [4.0],
        [5.0],
    ]

    assert y_train.data.tolist() == [
        [10.0],
        [20.0],
        [30.0],
    ]

    assert y_test.data.tolist() == [
        [40.0],
        [50.0],
    ]
