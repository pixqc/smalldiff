from __future__ import annotations
import numpy as np
from enum import Enum
from typing import Optional


OP = Enum("OP", ["UNARY", "BINARY"])


class Tensor:
  def __init__(self, data):
    self.data = np.array(data)
    self.grad = None
    self._ctx: Optional[Function] = None

  @property
  def shape(self):
    return self.data.shape

  def relu(self):
    return Relu.apply(self)

  def tanh(self):
    return Tanh.apply(self)

  def add(self, other):
    return Add.apply(self, other)

  def sum(self):
    return Sum.apply(self)

  def mul(self, other):
    return Mul.apply(self, other)

  def matmul(self, other):
    return Matmul.apply(self, other)

  def __repr__(self):
    return f"Tensor(data={self.data}, grad={self.grad})"

  def deepwalk(self) -> list[Tensor]:
    def _deepwalk(node, visited, nodes):
      visited.add(node)
      if getattr(node, "_ctx", None):
        for i in node._ctx.prev:
          if i not in visited:
            _deepwalk(i, visited, nodes)
        nodes.append(node)
      return nodes

    return _deepwalk(self, set(), [])

  def backward(self):
    assert self.data.ndim == 0, "only scalar can be backwarded"
    self.grad = 1.0
    for prev in reversed(self.deepwalk()):
      assert isinstance(prev._ctx, Function), f"ctx is None for {prev}"
      prev._ctx.backward(prev.grad)

  __add__ = add
  __mul__ = mul
  __matmul__ = matmul


class Function:
  def __init__(self, *tensors: Tensor):
    self.prev = tensors

  def forward(self, *args, **kwargs):
    raise NotImplementedError(f"forward not implemented for {type(self)}")

  def backward(self, *args, **kwargs):
    raise NotImplementedError(f"backward not implemented for {type(self)}")

  @classmethod
  def apply(cls, *x: Tensor, **kwargs):
    fxn: Function = cls(*x)
    res: Tensor = fxn.forward(*[t.data for t in x], **kwargs)
    res._ctx = fxn
    return res


class Relu(Function):
  def forward(self, x):
    return Tensor(np.maximum(x, 0))


class Tanh(Function):
  def forward(self, x):
    res = Tensor(np.tanh(x))
    self.res = res
    return res

  def backward(self, out_grad):
    self.prev[0].grad = out_grad * (1 - self.res.data**2)


class Add(Function):
  def forward(self, x, y):
    return Tensor(x + y)

  def backward(self, out_grad):
    for prev in self.prev:
      if prev.shape != out_grad.shape:
        prev.grad = np.sum(out_grad, axis=0, keepdims=True)
      else:
        prev.grad = out_grad


class Sum(Function):
  def forward(self, x):
    return Tensor(np.sum(x))

  def backward(self, out_grad):
    self.prev[0].grad = np.broadcast_to(out_grad, self.prev[0].data.shape)


class Mul(Function):
  def forward(self, x, y):
    return Tensor(x * y)

  def backward(self, out_grad):
    self.prev[0].grad = out_grad * self.prev[1].data
    self.prev[1].grad = out_grad * self.prev[0].data


class Matmul(Function):
  def forward(self, x, y):
    return Tensor(x @ y)

  def backward(self, out_grad):
    self.prev[0].grad = out_grad @ self.prev[1].data.T
    self.prev[1].grad = (out_grad.T @ self.prev[0].data).T
