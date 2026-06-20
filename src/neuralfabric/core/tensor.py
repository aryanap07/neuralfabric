from __future__ import annotations
import numpy as np

class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_prev", "_op", "_backward")

    def __init__(
        self, data, requires_grad: bool = False, dtype=None, _children=(), _op=""
    ):
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data, dtype=dtype)
        if dtype is None and (not np.issubdtype(arr.dtype, np.floating)):
            arr = arr.astype(np.float32)
        self.data: np.ndarray = arr
        self.requires_grad: bool = requires_grad
        self.grad: "np.ndarray | None" = None
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return self.transpose()

    def item(self):
        return self.data.item()

    def numpy(self):
        return self.data

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self):
        self.grad = None

    def requires_grad_(self, flag: bool = True):
        self.requires_grad = flag
        return self

    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    def _out_requires_grad(self, *others):
        return self.requires_grad or any((o.requires_grad for o in others))

    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(shape)

    def _accumulate(self, grad: np.ndarray):
        grad = Tensor._unbroadcast(grad, self.data.shape)
        self.grad = grad if self.grad is None else self.grad + grad

    def __add__(self, other):
        other = Tensor._wrap(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self._out_requires_grad(other),
            _children=(self, other),
            _op="add",
        )

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad)
            if other.requires_grad:
                other._accumulate(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Tensor._wrap(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self._out_requires_grad(other),
            _children=(self, other),
            _op="mul",
        )

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * other.data)
            if other.requires_grad:
                other._accumulate(out.grad * self.data)

        out._backward = _backward
        return out

    def __pow__(self, power: float):
        assert isinstance(power, (int, float)), "only scalar powers are supported"
        out = Tensor(
            self.data**power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"pow{power}",
        )

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * (power * self.data ** (power - 1)))

        out._backward = _backward
        return out

    def matmul(self, other):
        other = Tensor._wrap(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self._out_requires_grad(other),
            _children=(self, other),
            _op="matmul",
        )

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad @ np.swapaxes(other.data, -1, -2))
            if other.requires_grad:
                other._accumulate(np.swapaxes(self.data, -1, -2) @ out.grad)

        out._backward = _backward
        return out

    __matmul__ = matmul

    def transpose(self, axes=None):
        axes = axes if axes is not None else tuple(reversed(range(self.ndim)))
        out = Tensor(
            np.transpose(self.data, axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )
        inv_axes = np.argsort(axes)

        def _backward():
            if self.requires_grad:
                self._accumulate(np.transpose(out.grad, inv_axes))

        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and (not keepdims):
                    g = np.expand_dims(g, axis=axis)
                self._accumulate(np.broadcast_to(g, self.data.shape))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def exp(self):
        out = Tensor(
            np.exp(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp",
        )

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * out.data)

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad / self.data)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(
            np.maximum(self.data, 0),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(
            s, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid"
        )

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * out.data * (1.0 - out.data))

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * (1.0 - out.data**2))

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + -Tensor._wrap(other)

    def __rsub__(self, other):
        return Tensor._wrap(other) + -self

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * Tensor._wrap(other) ** (-1.0)

    def __rtruediv__(self, other):
        return Tensor._wrap(other) * self ** (-1.0)

    def __repr__(self):
        grad_info = (
            f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        )
        return f"Tensor({self.data}{grad_info})"

    def backward(self, grad=None):
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build_topo(t: "Tensor"):
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)
        self.grad = (
            np.ones_like(self.data)
            if grad is None
            else np.asarray(grad, dtype=self.data.dtype)
        )
        for t in reversed(topo):
            t._backward()
