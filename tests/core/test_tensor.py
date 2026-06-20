import numpy as np
import pytest
from neuralfabric.core.tensor import Tensor

RTOL, ATOL = (0.0001, 1e-06)

def finite_diff_grad(f, x: np.ndarray, h: float = 1e-05) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + h
        fp = f(x)
        x[idx] = orig - h
        fm = f(x)
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * h)
    return grad


def assert_matches_finite_diff(build_fn, x_np: np.ndarray):
    x_t = Tensor(x_np.copy(), requires_grad=True)
    loss = build_fn(x_t)
    loss.backward()
    analytic = x_t.grad.astype(np.float64)
    numeric = finite_diff_grad(
        lambda arr: build_fn(Tensor(arr)).item(), x_np.astype(np.float64).copy()
    )
    np.testing.assert_allclose(analytic, numeric, rtol=RTOL, atol=ATOL)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_add_with_broadcasting_gradient(rng):
    bias = Tensor(np.array([1.0, 2.0, 3.0]))
    assert_matches_finite_diff(lambda x: (x + bias).sum(), rng.standard_normal((3, 3)))


def test_mul_gradient(rng):
    assert_matches_finite_diff(lambda x: (x * x).sum(), rng.standard_normal(4))


def test_pow_gradient(rng):
    x = rng.standard_normal(5) + 2.0
    assert_matches_finite_diff(lambda x: (x**3).sum(), x)


def test_matmul_gradient(rng):
    w = rng.standard_normal((3, 2))
    assert_matches_finite_diff(
        lambda x: x.matmul(Tensor(w)).sum(), rng.standard_normal((4, 3))
    )


def test_sum_with_axis_gradient(rng):
    assert_matches_finite_diff(
        lambda x: x.sum(axis=1).sum(), rng.standard_normal((3, 4))
    )


def test_mean_gradient(rng):
    assert_matches_finite_diff(lambda x: x.mean(), rng.standard_normal(6))


def test_exp_gradient(rng):
    assert_matches_finite_diff(lambda x: x.exp().sum(), rng.standard_normal(4) * 0.5)


def test_log_gradient(rng):
    x = np.abs(rng.standard_normal(4)) + 0.5
    assert_matches_finite_diff(lambda x: x.log().sum(), x)


def test_relu_gradient():
    x = np.array([-2.0, -0.5, 0.3, 1.7])
    assert_matches_finite_diff(lambda x: x.relu().sum(), x)


def test_sigmoid_gradient(rng):
    assert_matches_finite_diff(lambda x: x.sigmoid().sum(), rng.standard_normal(4))


def test_tanh_gradient(rng):
    assert_matches_finite_diff(lambda x: x.tanh().sum(), rng.standard_normal(4))


def test_transpose_gradient(rng):
    assert_matches_finite_diff(lambda x: x.T.sum(), rng.standard_normal((3, 4)))


def test_reshape_gradient(rng):
    assert_matches_finite_diff(
        lambda x: x.reshape(2, 6).sum(), rng.standard_normal((3, 4))
    )


def test_chained_mse_like_expression():
    y_true = Tensor(np.array([1.0, 2.0, 3.0]))

    def mse(w):
        pred = w * Tensor(np.array([1.0, 1.0, 1.0]))
        return ((pred - y_true) ** 2).mean()

    assert_matches_finite_diff(mse, np.array([0.1, 0.2, 0.3]))


def test_gradient_descent_converges_to_true_linear_params():
    rng = np.random.default_rng(42)
    true_w, true_b = (3.0, -1.5)
    X = rng.uniform(-2, 2, size=50)
    y = true_w * X + true_b + rng.normal(0, 0.05, size=50)
    w = Tensor(0.0, requires_grad=True)
    b = Tensor(0.0, requires_grad=True)
    X_t, y_t = (Tensor(X), Tensor(y))
    lr = 0.1
    for _ in range(200):
        w.zero_grad()
        b.zero_grad()
        pred = X_t * w + b
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()
        w.data -= lr * w.grad
        b.data -= lr * b.grad
    assert abs(w.item() - true_w) < 0.05
    assert abs(b.item() - true_b) < 0.05


def test_requires_grad_false_does_not_populate_grad():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=False)
    y = (x * x).sum()
    y.backward()
    assert x.grad is None


def test_zero_grad_resets_to_none():
    x = Tensor([1.0, 2.0], requires_grad=True)
    (x * x).sum().backward()
    assert x.grad is not None
    x.zero_grad()
    assert x.grad is None
