import unittest

import numpy as np
from tinygrad import Tensor as TinyTensor

from tensor import Tensor


class TestGrad(unittest.TestCase):
  def test_add_00(self):
    x_np = np.random.rand(3).astype(np.float32)
    y_np = np.random.rand(3).astype(np.float32)

    x = Tensor(x_np, requires_grad=True)
    y = Tensor(y_np, requires_grad=True)
    out = x.add(y).sum()
    out.backward()

    x_tiny = TinyTensor(x_np, requires_grad=True)
    y_tiny = TinyTensor(y_np, requires_grad=True)
    out_tiny = x_tiny.add(y_tiny).sum()
    out_tiny.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(y.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, x_tiny.grad.numpy()))
    self.assertTrue(np.allclose(x.grad.data, x_tiny.grad.numpy()))
    self.assertTrue(np.allclose(y.grad.data, y_tiny.grad.numpy()))

  def test_add_01(self):
    x_np = np.random.rand(3).astype(np.float32)

    x = Tensor(x_np, requires_grad=True)
    out = (x + 1).sum()
    out.backward()

    x_tiny = TinyTensor(x_np, requires_grad=True)
    out_tiny = (x_tiny + 1).sum()
    out_tiny.backward()

    assert isinstance(x.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, x_tiny.grad.numpy()))

  def test_add_02(self):
    x_np = np.random.rand(2).astype(np.float32)
    y_np = np.random.rand(3, 2).astype(np.float32)

    x = Tensor(x_np, requires_grad=True)
    y = Tensor(y_np, requires_grad=True)
    out = (x + y).sum()
    out.backward()

    x_tiny = TinyTensor(x_np, requires_grad=True)
    y_tiny = TinyTensor(y_np, requires_grad=True)
    out_tiny = (x_tiny + y_tiny).sum()
    out_tiny.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(y.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, x_tiny.grad.numpy()))
    self.assertTrue(np.allclose(y.grad.data, y_tiny.grad.numpy()))

  def test_matmul_00(self):
    x_np = np.random.rand(3, 5).astype(np.float32)
    y_np = np.random.rand(5, 2).astype(np.float32)

    x = Tensor(x_np, requires_grad=True)
    y = Tensor(y_np, requires_grad=True)
    out = (x.matmul(y)).sum()
    out.backward()

    x_tiny = TinyTensor(x_np, requires_grad=True)
    y_tiny = TinyTensor(y_np, requires_grad=True)
    out_tiny = (x_tiny.matmul(y_tiny)).sum()
    out_tiny.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(y.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, x_tiny.grad.numpy()))
    self.assertTrue(np.allclose(y.grad.data, y_tiny.grad.numpy()))

  def test_relu_00(self):
    x_np = np.random.randn(3, 4).astype(np.float32)
    x = Tensor(x_np, requires_grad=True)
    out = x.relu().sum()
    out.backward()

    x_tiny = TinyTensor(x_np, requires_grad=True)
    out_tiny = x_tiny.relu().sum()
    out_tiny.backward()

    assert isinstance(x.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, x_tiny.grad.numpy()))

  def test_exp_00(self):
    x_np = np.random.randn(3).astype(np.float32)
    x = Tensor(x_np, requires_grad=True)
    out = x.exp().sum()
    out.backward()

    x_tiny = TinyTensor(x_np, requires_grad=True)
    out_tiny = x_tiny.exp().sum()
    out_tiny.backward()

    assert isinstance(x.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, x_tiny.grad.numpy()))

  def test_max_00(self):
    x_np = np.random.randn(3).astype(np.float32)
    x = Tensor(x_np, requires_grad=True)
    out = x.max().sum()
    out.backward()

    x_tiny = TinyTensor(x_np, requires_grad=True)
    out_tiny = x_tiny.max().sum()
    out_tiny.backward()

    assert isinstance(x.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, x_tiny.grad.numpy()))

  # def test_logsoftmax_00(self):
  #   # testing max, exp, sum
  #   x_np = np.random.randn(3, 4).astype(np.float32)
  #   x = Tensor(x_np, requires_grad=True)
  #   out = x.log_softmax().sum()
  #   out.backward()
  #
  #   x_tiny = TinyTensor(x_np, requires_grad=True)
  #   out_tiny = x_tiny.log_softmax().sum()
  #   out_tiny.backward()
  #
  #   print(x.grad.numpy())
  #   print(x_tiny.grad.numpy())
  #   assert isinstance(x.grad, Tensor)
  #   self.assertTrue(np.allclose(x.grad.data, x_tiny.grad.numpy()))
