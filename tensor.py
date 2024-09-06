import numpy as np
from typing import List, Union


class Tensor:
  def __init__(self, data):
    if np.isscalar(data):
      self.data = np.array([data])
    else:
      self.data = np.array(data)
    self.grad = np.zeros_like(data)
    self.prev: List[Tensor] = []
    self.op = None
    self._backward = lambda: None

  @property
  def shape(self):
    return self.data.shape

  @property
  def dtype(self):
    return self.data.dtype

  def add(self, t):
    t = t if isinstance(t, Tensor) else Tensor(t)
    out = Tensor(self.data + t.data)
    out.prev = [self, t]
    out.op = "add"

    def __backward():
      self.grad += out.grad
      t.grad += out.grad

    out._backward = __backward
    return out

  def mul(self, t):
    t = t if isinstance(t, Tensor) else Tensor(t)
    out = Tensor(self.data * t.data)
    out.prev = [self, t]
    out.op = "mul"

    def __backward():
      self.grad = t.data * out.grad
      t.grad = self.data * out.grad

    out._backward = __backward
    return out

  # https://claude.ai/chat/e1f052f7-5c73-47aa-ba5c-0b3d4316f1f2
  def matmul(self, t):
    t = t if isinstance(t, Tensor) else Tensor(t)
    out = Tensor(self.data @ t.data)
    out.prev = [self, t]
    out.op = "matmul"

    def __backward():
      bc = np.broadcast_to(out.grad, t.data.shape)
      self.grad += np.matmul(bc, t.data.T)
      t.grad += np.matmul(self.data.T, bc)

    out._backward = __backward
    return out

  def dot(self, t):
    t = t if isinstance(t, Tensor) else Tensor(t)
    out = Tensor(self.data @ t.data)
    out.prev = [self, t]
    out.op = "dot"

    def __backward():
      self.grad += np.matmul(t.data.T, out.grad)
      t.grad += np.matmul(self.data, out.grad)

    out._backward = __backward
    return out

  def sum(self):
    out = Tensor(np.sum(self.data))
    out.prev = [self]
    out.op = "sum"

    def __backward():
      self.grad = out.grad

    out._backward = __backward
    return out

  def tanh(self):
    out = Tensor(np.tanh(self.data))
    out.prev = [self]
    out.op = "tanh"

    def __backward():
      self.grad *= (1 - out.data**2) * out.grad

    out._backward = __backward
    return out

  def logsoftmax(self):
    max_val = np.max(self.data, axis=1, keepdims=True)
    exp_x = np.exp(self.data - max_val)
    log_sum_exp = np.log(np.sum(exp_x, axis=1, keepdims=True))
    out = Tensor(self.data - max_val - log_sum_exp)
    out.prev = [self]
    out.op = "logsoftmax"

    def __backward():
      softmax_output = np.exp(out.data)
      self.grad += out.grad - softmax_output * np.sum(out.grad, axis=1, keepdims=True)

    out._backward = __backward
    return out

  __add__ = add
  __mul__ = mul
  __matmul__ = matmul

  @staticmethod
  def rand(shape: Union[tuple, int]):
    return Tensor(np.random.rand(*shape))

  @staticmethod
  def full(shape: tuple, value):
    return Tensor(np.full(shape, value))

  @staticmethod
  def eye(dim: int):
    return Tensor(np.eye(dim))

  def __repr__(self):
    return f"<Buf {self.shape} dtype {self.dtype}>"

  def deepwalk(self):
    def _deepwalk(node, visited, nodes):
      visited.add(node)
      if len(node.prev) > 0:
        for i in node.prev:
          if i not in visited:
            _deepwalk(i, visited, nodes)
        nodes.append(node)
      return nodes

    return _deepwalk(self, set(), [])

  def backward(self):
    assert self.shape == (1,), "only scalars can be backwarded"
    self.grad = 1
    for node in reversed(self.deepwalk()):
      node._backward()


def simple_forward():
  input = Tensor([[1.0, -2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5], [-1.0, 0.0, 1.0, 2.0]])
  y = Tensor([[-6, 0.75], [0.25, 0.75], [0.25, 0.75]])
  w1 = Tensor.rand((4, 6))
  b1 = Tensor.rand(6)
  z1 = input @ w1 + b1
  a1 = z1.tanh()

  w2 = Tensor.rand((6, 2))
  b2 = Tensor.rand(2)
  z2 = a1 @ w2 + b2
  out = z2.softmax()
  # lo = out.cross_entropy(y)
  # print(lo.data)


# def cross_entropy(self, y):
#   y = y if isinstance(y, Tensor) else Tensor(y)
#   e = 1e-7  # to avoid log(0)
#   d = np.clip(self.data, e, 1 - e)
#   loss = np.mean(-(y.data * np.log(d) + (1 - y.data) * np.log(1 - d)))
#   out = Tensor(loss)
#   out.prev = [self, y]
#   out.op = "cross_entropy"
#
#   def _backward():
#     grad_scale = out.grad / self.data.size
#     self.grad += grad_scale * ((1 - y.data) / (1 - d) - y.data / d)
#     y.grad += grad_scale * (-np.log(d) + np.log(1 - d))
#
#   out._backward = _backward
#   return out
