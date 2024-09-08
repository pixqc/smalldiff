import unittest

import numpy as np
import torch

from tensor import Tensor


class test_grad(unittest.TestCase):
  def test_add(self):
    x_np = np.array([1, 2, 3], dtype=np.float32)
    x = Tensor(x_np)
    out = (x + 1).sum()
    out.backward()

    x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    out_torch = (x_torch + 1).sum()
    out_torch.backward()

    assert isinstance(x.grad, Tensor)
    assert x.grad.shape == x_torch.grad.shape

    np.testing.assert_allclose(
      out.data, out_torch.detach().numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(x.grad.data, x_torch.grad.numpy(), rtol=1e-5, atol=1e-5)

  def test_matmul(self):
    x_np = Tensor.rand(5, 3)
    y_np = Tensor.rand(3, 2)
    x = Tensor(x_np)
    y = Tensor(y_np)
    out = (x @ y).sum()
    out.backward()

    x_torch = torch.tensor(x_np.data, dtype=torch.float32, requires_grad=True)
    y_torch = torch.tensor(y_np.data, dtype=torch.float32, requires_grad=True)
    out_torch = (x_torch @ y_torch).sum()
    out_torch.backward()

    assert isinstance(x.grad, Tensor)
    assert x.grad.shape == x_torch.grad.shape

    np.testing.assert_allclose(
      out.data, out_torch.detach().numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(x.grad.data, x_torch.grad.numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
