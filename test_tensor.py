import unittest
from typing import Optional

import numpy as np
from tinygrad import Tensor as TinyTensor

from tensor import Tensor

tol_kwargs = {"atol": 1e-5, "rtol": 1e-5, "equal_nan": False}


class TestGrad(unittest.TestCase):
  def _test_unary(self, func_name: str, shape: tuple[int, ...]):
    x_np = np.random.rand(*shape).astype(np.float32)

    x = Tensor(x_np, requires_grad=True)
    out = getattr(x, func_name)()
    out = out.sum()
    out.backward()

    x_t = TinyTensor(x_np, requires_grad=True)
    out_t = getattr(x_t, func_name)()
    out_t = out_t.sum()
    out_t.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(x_t.grad, TinyTensor)
    self.assertTrue(x.grad.shape == x_t.grad.shape)
    self.assertTrue(np.allclose(out.numpy(), out_t.numpy(), **tol_kwargs))
    self.assertTrue(np.allclose(x.grad.numpy(), x_t.grad.numpy(), **tol_kwargs))

  def test_unary(self):
    for func_name in ["relu", "tanh", "log", "exp", "neg", "reciprocal"]:
      for shape in [(3,), (3, 4), (2, 3, 4)]:
        self._test_unary(func_name, shape)

  def _test_binary(self, func_name: str, shapes: tuple[tuple[int, ...], tuple[int, ...]]):  # fmt: skip
    shape1, shape2 = shapes
    x_np = np.random.randn(*shape1).astype(np.float32)
    y_np = np.random.randn(*shape2).astype(np.float32)

    x = Tensor(x_np, requires_grad=True)
    y = Tensor(y_np, requires_grad=True)
    out = getattr(x, func_name)(y)
    out = out.sum()
    out.backward()

    x_t = TinyTensor(x_np, requires_grad=True)
    y_t = TinyTensor(y_np, requires_grad=True)
    out_t = getattr(x_t, func_name)(y_t)
    out_t = out_t.sum()
    out_t.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(y.grad, Tensor)
    assert isinstance(x_t.grad, TinyTensor)
    assert isinstance(y_t.grad, TinyTensor)
    self.assertTrue(x.grad.shape == x_t.grad.shape)
    self.assertTrue(y.grad.shape == y_t.grad.shape)
    self.assertTrue(np.allclose(out.numpy(), out_t.numpy(), **tol_kwargs))
    self.assertTrue(np.allclose(x.grad.numpy(), x_t.grad.numpy(), **tol_kwargs))
    self.assertTrue(np.allclose(y.grad.numpy(), y_t.grad.numpy(), **tol_kwargs))

  def test_binary(self):
    for func_name in ["add", "mul", "sub", "div"]:
      for shapes in [
        ((3,), (3,)),
        ((3, 4), (3, 4)),
        ((2,), (3, 2)),
        ((3, 4), (4,)),
        ((2, 3, 4), (2, 3, 4)),
        ((2, 3, 4), (3, 4)),
        ((2, 3, 4), (4,)),
        ((3, 4), (2, 3, 4)),
        ((4,), (2, 3, 4)),
      ]:
        self._test_binary(func_name, shapes)

  def _test_binary_matmul(self, shapes: tuple[tuple[int, ...], tuple[int, ...]]):
    shape1, shape2 = shapes
    x_np = np.random.randn(*shape1).astype(np.float32)
    y_np = np.random.randn(*shape2).astype(np.float32)

    x = Tensor(x_np, requires_grad=True)
    y = Tensor(y_np, requires_grad=True)
    out = x.matmul(y)
    out = out.sum()
    out.backward()

    x_t = TinyTensor(x_np, requires_grad=True)
    y_t = TinyTensor(y_np, requires_grad=True)
    out_t = x_t.matmul(y_t)
    out_t = out_t.sum()
    out_t.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(y.grad, Tensor)
    assert isinstance(x_t.grad, TinyTensor)
    assert isinstance(y_t.grad, TinyTensor)
    self.assertTrue(x.grad.shape == x_t.grad.shape)
    self.assertTrue(y.grad.shape == y_t.grad.shape)
    self.assertTrue(np.allclose(out.numpy(), out_t.numpy(), **tol_kwargs))
    self.assertTrue(np.allclose(x.grad.numpy(), x_t.grad.numpy(), **tol_kwargs))
    self.assertTrue(np.allclose(y.grad.numpy(), y_t.grad.numpy(), **tol_kwargs))

  def test_binary_matmul(self):
    for shape_pair in [
      ((2, 3), (3, 4)),
      ((3, 4), (4, 2)),
      # ((2, 3, 4), (4, 5)),  # TODO: not supported yet
      # ((1, 2, 3, 4), (4, 5)),
    ]:
      self._test_binary_matmul(shape_pair)

  def _test_reduce(self, func_name: str, shape: tuple[int, ...], axis: Optional[int]):
    x_np = np.random.randn(*shape).astype(np.float32)

    x = Tensor(x_np, requires_grad=True)
    out = getattr(x, func_name)(axis=axis)
    out = out.sum()
    out.backward()

    x_t = TinyTensor(x_np, requires_grad=True)
    out_t = getattr(x_t, func_name)(axis=axis)
    out_t = out_t.sum()
    out_t.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(x_t.grad, TinyTensor)
    self.assertTrue(x.grad.shape == x_t.grad.shape)
    self.assertTrue(np.allclose(out.numpy(), out_t.numpy(), **tol_kwargs))
    self.assertTrue(np.allclose(x.grad.numpy(), x_t.grad.numpy(), **tol_kwargs))

  def test_reduce(self):
    for func_name in ["max", "sum", "mean"]:  # TODO: prod
      for shape in [(3,), (3, 4), (3, 4, 5)]:
        self._test_reduce(func_name, shape, axis=None)
        for axis in range(len(shape)):
          self._test_reduce(func_name, shape, axis=axis)

  def test_composite_00(self):  # softmax
    # TODO: 0-axis still broken, likely add backward shape problem
    for axis in [None, 1]:
      x_np = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.88]]).astype(np.float32)
      x = Tensor(x_np, requires_grad=True)
      out = x.softmax(axis=axis)
      out.sum().backward()

      x_t = TinyTensor(x_np, requires_grad=True)
      out_t = x_t.softmax(axis=axis)
      out_t.sum().backward()

      assert isinstance(x.grad, Tensor)
      assert isinstance(x_t.grad, TinyTensor)
      self.assertTrue(np.allclose(out.numpy(), out_t.numpy(), **tol_kwargs))
      self.assertTrue(np.allclose(x.grad.numpy(), x_t.grad.numpy(), **tol_kwargs))

  def test_composite_01(self):  # crossentropy
    x_np = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.88]]).astype(np.float32)
    y_np = np.array([0, 2]).astype(np.int32)

    x = Tensor(x_np, requires_grad=True)
    y = Tensor(y_np)
    out = x.cross_entropy(y)
    out.backward()

    x_t = TinyTensor(x_np, requires_grad=True)
    y_t = TinyTensor(y_np)  # out_t.backward() breaks if requires_grad=True
    out_t = x_t.cross_entropy(y_t).backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(x_t.grad, TinyTensor)
    self.assertTrue(np.allclose(out.numpy(), out_t.numpy(), **tol_kwargs))
    self.assertTrue(np.allclose(x.grad.numpy(), x_t.grad.numpy(), **tol_kwargs))

  def test_composite_02(self):
    # mnist 784 -> 16 -> 16 -> 10
    x_np = np.random.randn(2, 784).astype(np.float32)
    y_np = np.array([0, 1]).astype(np.int32)
    w1_np = np.random.randn(784, 16).astype(np.float32)
    b1_np = np.random.randn(16).astype(np.float32)
    w2_np = np.random.randn(16, 16).astype(np.float32)
    b2_np = np.random.randn(16).astype(np.float32)
    w3_np = np.random.randn(16, 10).astype(np.float32)
    b3_np = np.random.randn(10).astype(np.float32)

    x = Tensor(x_np)
    y = Tensor(y_np)
    w1 = Tensor(w1_np, requires_grad=True)
    b1 = Tensor(b1_np, requires_grad=True)
    w2 = Tensor(w2_np, requires_grad=True)
    b2 = Tensor(b2_np, requires_grad=True)
    w3 = Tensor(w3_np, requires_grad=True)
    b3 = Tensor(b3_np, requires_grad=True)

    out = ((x @ w1 + b1).tanh() @ w2 + b2).tanh() @ w3 + b3
    out = out.cross_entropy(y)
    out.backward()

    x_t = TinyTensor(x_np)
    y_t = TinyTensor(y_np)
    w1_t = TinyTensor(w1_np, requires_grad=True)
    b1_t = TinyTensor(b1_np, requires_grad=True)
    w2_t = TinyTensor(w2_np, requires_grad=True)
    b2_t = TinyTensor(b2_np, requires_grad=True)
    w3_t = TinyTensor(w3_np, requires_grad=True)
    b3_t = TinyTensor(b3_np, requires_grad=True)

    out_t = (
      x_t.matmul(w1_t)
      .add(b1_t)
      .tanh()
      .matmul(w2_t)
      .add(b2_t)
      .tanh()
      .matmul(w3_t)
      .add(b3_t)
      .cross_entropy(y_t)
    )

    out_t = ((x_t @ w1_t + b1_t).tanh() @ w2_t + b2_t).tanh() @ w3_t + b3_t
    out_t = out_t.cross_entropy(y_t)
    out_t.backward()

    for t in [w1, b1, w2, b2, w3, b3]:
      assert isinstance(t.grad, Tensor)

    self.assertTrue(np.allclose(out.numpy(), out_t.numpy(), **tol_kwargs))
    self.assertTrue(np.allclose(w1.grad.numpy(), w1_t.grad.numpy(), atol=1e-3))  # type: ignore
    self.assertTrue(np.allclose(b1.grad.numpy(), b1_t.grad.numpy(), **tol_kwargs))  # type: ignore
    self.assertTrue(np.allclose(w2.grad.numpy(), w2_t.grad.numpy(), **tol_kwargs))  # type: ignore
    self.assertTrue(np.allclose(b2.grad.numpy(), b2_t.grad.numpy(), **tol_kwargs))  # type: ignore
    self.assertTrue(np.allclose(w3.grad.numpy(), w3_t.grad.numpy(), **tol_kwargs))  # type: ignore
    self.assertTrue(np.allclose(b3.grad.numpy(), b3_t.grad.numpy(), **tol_kwargs))  # type: ignore


if __name__ == "__main__":
  unittest.main()
