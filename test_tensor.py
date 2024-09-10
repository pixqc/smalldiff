import unittest

import numpy as np
from tinygrad import Tensor as TinyTensor

from tensor import Tensor


class TestGrad(unittest.TestCase):
  def _test_unary(self, func_name, shape):
    x_np = np.random.randn(*shape).astype(np.float32)
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
    self.assertTrue(np.allclose(x.grad.data, x_tiny.grad.numpy()))

  def test_unary(self):
    for func_name in ["relu", "log", "exp", "neg", "reciprocal"]:
      for shape in [(3,), (3, 4)]:
        self._test_unary(func_name, shape)

  def _test_binary(self, func_name, shapes):
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
    self.assertTrue(np.allclose(x.grad.data, x_tiny.grad.numpy()))
    self.assertTrue(np.allclose(y.grad.data, y_tiny.grad.numpy()))

  def test_binary(self):
    for func_name in ["add"]:
      for shapes in [((3,), (3,)), ((3, 4), (3, 4)), ((2,), (3, 2)), ((3, 4), (4,))]:
        self._test_binary(func_name, shapes)

  def _test_reduce(self, func_name, shape):
    x_np = np.random.randn(*shape).astype(np.float32)
    x = Tensor(x_np, requires_grad=True)
    out = getattr(x, func_name)()
    out = out.sum()
    out.backward()

    x_tiny = TinyTensor(x_np, requires_grad=True)
    out_tiny = getattr(x_tiny, func_name)()
    out_tiny = out_tiny.sum()
    out_tiny.backward()

    assert isinstance(x.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, x_tiny.grad.numpy()))

  def test_reduce(self):
    for func_name in ["max", "sum", "mean"]:
      for shape in [(3,), (3, 4)]:
        self._test_reduce(func_name, shape)
