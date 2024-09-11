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
    for func_name in ["max", "sum", "mean"]:
      for shape in [(3,), (3, 4), (3, 4, 5)]:
        self._test_reduce(func_name, shape, axis=None)
        for axis in range(len(shape)):
          self._test_reduce(func_name, shape, axis=axis)

  def test_softmax(self):
    x_np = np.random.randn(3).astype(np.float32)

    x = Tensor(x_np, requires_grad=True)
    out = x.softmax()
    print(out.numpy())
    out = out.sum()
    out.backward()

    x_tiny = TinyTensor(x_np, requires_grad=True)
    out_tiny = x_tiny.softmax()
    print(out_tiny.numpy())
    out_tiny = out_tiny.sum()
    out_tiny.backward()

    print("--")
    print(x.grad.numpy())
    print(x_tiny.grad.numpy())

    assert isinstance(x.grad, Tensor)
    assert isinstance(x_tiny.grad, TinyTensor)
    self.assertTrue(x.grad.shape == x_tiny.grad.shape)
    self.assertTrue(np.allclose(out.numpy(), out_tiny.numpy()))
    self.assertTrue(np.allclose(x.grad.numpy(), x_tiny.grad.numpy()))

  @unittest.skip("not supported yet")
  def test_log_softmax(self):
    x_np = np.random.randn(10, 3).astype(np.float32)

    x = Tensor(x_np)
    out = x.log_softmax()
    out = out.sum()
    out.backward()

    x_tiny = TinyTensor(x_np)
    out_tiny = x_tiny.log_softmax()
    out_tiny = out_tiny.sum()
    out_tiny.backward()

    assert isinstance(x.grad, Tensor)
    assert isinstance(x_tiny.grad, TinyTensor)
    self.assertTrue(x.grad.shape == x_tiny.grad.shape)
    self.assertTrue(np.allclose(x.grad.numpy(), x_tiny.grad.numpy()))

  @unittest.skip("not supported yet")
  def test_crossentropy(self):
    x_np = np.random.randn(3, 10).astype(np.float32)
    y_np = np.random.randint(10, size=3).astype(np.int32)

    x = Tensor(x_np)
    y = Tensor(y_np)
    y = np.eye(10)[y.data]  # one hot

    out = x.softmax().log().mul(y).mean()
    # .neg().div(Tensor(3))
    print(out)

    x_tiny = TinyTensor(x_np)
    y_tiny = TinyTensor(y_np)
    out_tiny = x_tiny.sparse_categorical_crossentropy(y_tiny)

    print(out.numpy())
    print(out_tiny.numpy())

  @unittest.skip("not supported yet")
  def test_mlp_mnist(self):
    x_np = np.random.randn(10, 784).astype(np.float32)
    y_np = np.random.randint(10, size=10).astype(np.int32)
    w1_np = np.random.randn(784, 256).astype(np.float32)
    b1_np = np.random.randn(256).astype(np.float32)
    w2_np = np.random.randn(256, 10).astype(np.float32)
    b2_np = np.random.randn(10).astype(np.float32)

    x = Tensor(x_np)
    y = Tensor(y_np)
    w1 = Tensor(w1_np, requires_grad=True)
    b1 = Tensor(b1_np, requires_grad=True)
    w2 = Tensor(w2_np, requires_grad=True)
    b2 = Tensor(b2_np, requires_grad=True)
    out = x.matmul(w1).add(b1).tanh().matmul(w2).add(b2)
    out = out.sparse_categorical_crossentropy(y)
    out.backward()

    x_tiny = TinyTensor(x_np)
    y_tiny = TinyTensor(y_np)
    w1_tiny = TinyTensor(w1_np, requires_grad=True)
    b1_tiny = TinyTensor(b1_np, requires_grad=True)
    w2_tiny = TinyTensor(w2_np, requires_grad=True)
    b2_tiny = TinyTensor(b2_np, requires_grad=True)
    out_tiny = x_tiny.matmul(w1_tiny).add(b1_tiny).tanh()
    out_tiny = out_tiny.matmul(w2_tiny).add(b2_tiny)
    out_tiny = out_tiny.sparse_categorical_crossentropy(y_tiny)
    out_tiny.backward()

    self.assertTrue(np.allclose(out.numpy(), out_tiny.numpy()))
    assert isinstance(w1.grad, Tensor)
    assert isinstance(b1.grad, Tensor)
    assert isinstance(w2.grad, Tensor)
    assert isinstance(b2.grad, Tensor)
    assert isinstance(w1_tiny.grad, TinyTensor)
    assert isinstance(b1_tiny.grad, TinyTensor)
    assert isinstance(w2_tiny.grad, TinyTensor)
    assert isinstance(b2_tiny.grad, TinyTensor)
    self.assertTrue(np.allclose(w1.grad.numpy(), w1_tiny.grad.numpy()))
    self.assertTrue(np.allclose(b1.grad.numpy(), b1_tiny.grad.numpy()))
    self.assertTrue(np.allclose(w2.grad.numpy(), w2_tiny.grad.numpy()))
    self.assertTrue(np.allclose(b2.grad.numpy(), b2_tiny.grad.numpy()))


if __name__ == "__main__":
  unittest.main()
