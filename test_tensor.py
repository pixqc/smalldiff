import unittest
from typing import Optional

import numpy as np
from tinygrad import Tensor as TinyTensor

from tensor import Tensor


class TestGrad(unittest.TestCase):
  def _test_unary(self, func_name: str, shape: tuple[int, ...]):
    x_np = np.random.rand(*shape).astype(np.float32)

    x = Tensor(x_np, requires_grad=True)
    out = getattr(x, func_name)()
    out = out.sum()
    out.backward()

    x_tiny = TinyTensor(x_np, requires_grad=True)
    out_tiny = getattr(x_tiny, func_name)()
    out_tiny = out_tiny.sum()
    out_tiny.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(x_tiny.grad, TinyTensor)
    self.assertTrue(x.grad.shape == x_tiny.grad.shape)
    self.assertTrue(np.allclose(out.numpy(), out_tiny.numpy()))
    self.assertTrue(np.allclose(x.grad.numpy(), x_tiny.grad.numpy()))

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

    x_tiny = TinyTensor(x_np, requires_grad=True)
    y_tiny = TinyTensor(y_np, requires_grad=True)
    out_tiny = getattr(x_tiny, func_name)(y_tiny)
    out_tiny = out_tiny.sum()
    out_tiny.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(y.grad, Tensor)
    assert isinstance(x_tiny.grad, TinyTensor)
    assert isinstance(y_tiny.grad, TinyTensor)
    self.assertTrue(x.grad.shape == x_tiny.grad.shape)
    self.assertTrue(y.grad.shape == y_tiny.grad.shape)
    self.assertTrue(np.allclose(out.numpy(), out_tiny.numpy()))
    self.assertTrue(np.allclose(x.grad.numpy(), x_tiny.grad.numpy()))
    self.assertTrue(np.allclose(y.grad.numpy(), y_tiny.grad.numpy()))

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

    x_tiny = TinyTensor(x_np, requires_grad=True)
    y_tiny = TinyTensor(y_np, requires_grad=True)
    out_tiny = x_tiny.matmul(y_tiny)
    out_tiny = out_tiny.sum()
    out_tiny.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(y.grad, Tensor)
    assert isinstance(x_tiny.grad, TinyTensor)
    assert isinstance(y_tiny.grad, TinyTensor)
    self.assertTrue(x.grad.shape == x_tiny.grad.shape)
    self.assertTrue(y.grad.shape == y_tiny.grad.shape)
    self.assertTrue(np.allclose(out.numpy(), out_tiny.numpy()))
    self.assertTrue(np.allclose(x.grad.numpy(), x_tiny.grad.numpy()))
    self.assertTrue(np.allclose(y.grad.numpy(), y_tiny.grad.numpy()))

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

    x_tiny = TinyTensor(x_np, requires_grad=True)
    out_tiny = getattr(x_tiny, func_name)(axis=axis)
    out_tiny = out_tiny.sum()
    out_tiny.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(x_tiny.grad, TinyTensor)
    self.assertTrue(x.grad.shape == x_tiny.grad.shape)
    self.assertTrue(np.allclose(out.numpy(), out_tiny.numpy()))
    self.assertTrue(np.allclose(x.grad.numpy(), x_tiny.grad.numpy()))

  def test_reduce(self):
    for func_name in ["max", "sum", "mean"]:  # TODO: prod
      for shape in [(3,), (3, 4), (3, 4, 5)]:
        self._test_reduce(func_name, shape, axis=None)
        for axis in range(len(shape)):
          self._test_reduce(func_name, shape, axis=axis)

  def test_composite_00(self):  # softmax
    x_np = np.array([0.7, 0.2, 0.1]).astype(np.float32)
    x = Tensor(x_np, requires_grad=True)
    out = x.softmax()
    out.sum().backward()

    x_tiny = TinyTensor(x_np, requires_grad=True)
    out_tiny = x_tiny.softmax()
    out_tiny.sum().backward()

    self.assertTrue(np.allclose(out.numpy(), out_tiny.numpy()))
    assert isinstance(x.grad, Tensor)
    assert isinstance(x_tiny.grad, TinyTensor)
    self.assertTrue(np.allclose(x.grad.numpy(), x_tiny.grad.numpy()))

  def test_composite_01(self):
    x_np = np.random.randn(2, 4).astype(np.float32)
    w_np = np.random.randn(4, 5).astype(np.float32)
    b_np = np.random.randn(5).astype(np.float32)

    x = Tensor(x_np, requires_grad=True)
    w = Tensor(w_np, requires_grad=True)
    b = Tensor(b_np, requires_grad=True)
    out = x.matmul(w).add(b).tanh().softmax().sum()
    out.backward()

    x_tiny = TinyTensor(x_np, requires_grad=True)
    w_tiny = TinyTensor(w_np, requires_grad=True)
    b_tiny = TinyTensor(b_np, requires_grad=True)
    out_tiny = x_tiny.matmul(w_tiny).add(b_tiny).tanh().softmax().sum()
    out_tiny.backward()

    self.assertTrue(np.allclose(out.numpy(), out_tiny.numpy()))
    assert isinstance(x.grad, Tensor)
    assert isinstance(w.grad, Tensor)
    assert isinstance(b.grad, Tensor)
    assert isinstance(x_tiny.grad, TinyTensor)
    assert isinstance(w_tiny.grad, TinyTensor)
    assert isinstance(b_tiny.grad, TinyTensor)
    self.assertTrue(np.allclose(x.grad.numpy(), x_tiny.grad.numpy()))
    self.assertTrue(np.allclose(w.grad.numpy(), w_tiny.grad.numpy()))
    self.assertTrue(np.allclose(b.grad.numpy(), b_tiny.grad.numpy()))


if __name__ == "__main__":
  unittest.main()
