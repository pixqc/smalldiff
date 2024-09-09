import unittest

import numpy as np

from tensor import Tensor


class TestGrad(unittest.TestCase):
  def test_add_00(self):
    x = Tensor.rand(3)
    out = (x + 1).sum()
    out.backward()
    assert isinstance(x.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, np.ones_like(x.data)))

  def test_add_01(self):
    x = Tensor.rand(2)
    y = Tensor.rand(3, 2)
    out = (x + y).sum()
    out.backward()
    assert isinstance(x.grad, Tensor)
    assert isinstance(y.grad, Tensor)
    self.assertTrue(np.allclose(y.grad.data, np.ones_like(y.data)))
    self.assertTrue(np.allclose(x.grad.data, np.ones_like(x.data) * 3))

  def test_add_02(self):
    x = Tensor.rand(3)
    out = (x + x).sum()
    out.backward()
    assert isinstance(x.grad, Tensor)
    self.assertTrue(np.allclose(x.grad.data, np.ones_like(x.data)))


# class test_against_torch(unittest.TestCase):
#   def test_add(self):
#     x_np = np.array([1, 2, 3], dtype=np.float32)
#     x = Tensor(x_np)
#     out = (x + 1).sum()
#     out.backward()
#
#     x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
#     out_torch = (x_torch + 1).sum()
#     out_torch.backward()


if __name__ == "__main__":
  unittest.main()
