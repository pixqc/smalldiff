import unittest

import numpy as np
import torch

from tensor import Tensor


class TestGrad(unittest.TestCase):
  def test_add_00(self):
    x_np = np.random.rand(3)
    y_np = np.random.rand(3)
    x = Tensor(x_np)
    y = Tensor(y_np)
    out = (x + y).sum()
    out.backward()

    x_torch = torch.tensor(x_np, requires_grad=True)
    y_torch = torch.tensor(y_np, requires_grad=True)
    out_torch = (x_torch + y_torch).sum()
    out_torch.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(y.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, x_torch.grad.numpy()))
    self.assertTrue(np.allclose(y.grad.data, y_torch.grad.numpy()))

  def test_add_01(self):
    x_np = np.random.rand(3)
    x = Tensor(x_np)
    out = (x + 1).sum()
    out.backward()

    x_torch = torch.tensor(x_np, requires_grad=True)
    out_torch = (x_torch + 1).sum()
    out_torch.backward()

    assert isinstance(x.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, x_torch.grad.numpy()))

  def test_add_02(self):
    x_np = np.random.rand(2)
    y_np = np.random.rand(3, 2)
    x = Tensor(x_np)
    y = Tensor(y_np)
    out = (x + y).sum()
    out.backward()

    x_torch = torch.tensor(x_np, requires_grad=True)
    y_torch = torch.tensor(y_np, requires_grad=True)
    out_torch = (x_torch + y_torch).sum()
    out_torch.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(y.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, x_torch.grad.numpy()))
    self.assertTrue(np.allclose(y.grad.data, y_torch.grad.numpy()))

  def test_matmul_00(self):
    x_np = np.random.rand(3, 5)
    y_np = np.random.rand(5, 2)
    x = Tensor(x_np)
    y = Tensor(y_np)
    out = (x @ y).sum()
    out.backward()

    x_torch = torch.tensor(x_np, requires_grad=True)
    y_torch = torch.tensor(y_np, requires_grad=True)
    out_torch = (x_torch @ y_torch).sum()
    out_torch.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(y.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, x_torch.grad.numpy()))
    self.assertTrue(np.allclose(y.grad.data, y_torch.grad.numpy()))


if __name__ == "__main__":
  unittest.main()
