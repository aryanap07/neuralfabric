import pytest

from neuralfabric.core.tensor import Tensor
from neuralfabric.model_selection.train_test_split import train_test_split


def test_split_shapes():
    X = Tensor(
        [
            [1],
            [2],
            [3],
            [4],
            [5],
        ]
    )

    y = Tensor(
        [
            [1],
            [2],
            [3],
            [4],
            [5],
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
    )

    assert X_train.shape[0] == 4
    assert X_test.shape[0] == 1

    assert y_train.shape[0] == 4
    assert y_test.shape[0] == 1


def test_random_state_reproducibility():
    X = Tensor(
        [
            [1],
            [2],
            [3],
            [4],
            [5],
        ]
    )

    y = Tensor(
        [
            [1],
            [2],
            [3],
            [4],
            [5],
        ]
    )

    split1 = train_test_split(
        X,
        y,
        random_state=42,
    )

    split2 = train_test_split(
        X,
        y,
        random_state=42,
    )

    assert (split1[0].data == split2[0].data).all()

    assert (split1[1].data == split2[1].data).all()


def test_invalid_test_size():
    X = Tensor(
        [
            [1],
            [2],
        ]
    )

    y = Tensor(
        [
            [1],
            [2],
        ]
    )

    with pytest.raises(ValueError):
        train_test_split(
            X,
            y,
            test_size=1.5,
        )


def test_mismatched_sample_count():
    X = Tensor(
        [
            [1],
            [2],
            [3],
        ]
    )

    y = Tensor(
        [
            [1],
            [2],
        ]
    )

    with pytest.raises(ValueError):
        train_test_split(X, y)


def test_no_shuffle():
    X = Tensor(
        [
            [1],
            [2],
            [3],
            [4],
            [5],
        ]
    )

    y = Tensor(
        [
            [1],
            [2],
            [3],
            [4],
            [5],
        ]
    )

    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.4,
        shuffle=False,
    )

    assert X_train.data[0][0] == 1
    assert X_train.data[1][0] == 2
    assert X_train.data[2][0] == 3

    assert X_test.data[0][0] == 4
    assert X_test.data[1][0] == 5
