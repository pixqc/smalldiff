from __future__ import annotations

from typing import Optional, Union

import numpy as np


class Tensor:
  def __init__(self, data, dtype=None):
    self.data = np.array(data, dtype=dtype)
    self.grad: Optional[Tensor] = None
    self._ctx: Optional[Function] = None

  @property
  def shape(self):
    return self.data.shape

  @property
  def T(self):
    return Tensor(self.data.T)

  @property
  def dtype(self):
    return self.data.dtype

  @staticmethod
  def rand(*shape):
    return Tensor(np.random.randn(*shape))

  @staticmethod
  def ones_like(*shape) -> Tensor:
    return Tensor(np.ones(shape))

  @staticmethod
  def zeros_like(*shape) -> Tensor:
    return Tensor(np.zeros(shape))

  def add(self, other):
    return Add.apply(self, other)

  def sum(self):
    return Sum.apply(self)

  def matmul(self, other):
    return Matmul.apply(self, other)

  def relu(self):
    return Relu.apply(self)

  def softmax(self):
    return Softmax.apply(self)

  #
  #
  # def mul(self, other):
  #   return Mul.apply(self, other)
  #
  # def matmul(self, other):
  #   return Matmul.apply(self, other)

  def __repr__(self):
    grad_repr = self.grad.data if self.grad else None
    return f"<Tensor {self.data!r} with grad {grad_repr!r}>"

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
    self.grad = Tensor(1.0)
    for prev in reversed(self.deepwalk()):
      assert isinstance(prev._ctx, Function), f"ctx is None for {prev}"
      prev._ctx.backward(prev.grad)

  __add__ = add
  # __mul__ = mul
  __matmul__ = matmul


class Function:
  def __init__(self, *tensors: Tensor):
    self.prev = tensors

  def forward(self, *args, **kwargs):
    raise NotImplementedError(f"forward not implemented for {type(self)}")

  def backward(self, *args, **kwargs):
    raise NotImplementedError(f"backward not implemented for {type(self)}")

  @classmethod
  def apply(cls, *x: Union[Tensor, int, float, np.ndarray], **kwargs):
    ensure_tensor = lambda x: x if isinstance(x, Tensor) else Tensor(x)
    tensors = [ensure_tensor(t) for t in x]
    fxn: Function = cls(*tensors)
    res: Tensor = fxn.forward(*[t.data for t in tensors], **kwargs)
    res._ctx = fxn
    return res


class Add(Function):
  def forward(self, x, y) -> Tensor:
    return Tensor(x + y)

  def backward(self, out_grad: Tensor):
    # np.sum is reduce, opposite of implicit broadcast of numpy's x+y
    for tensor in (self.prev[0], self.prev[1]):
      tensor.grad = (
        out_grad
        if tensor.shape == out_grad.shape
        else Tensor(np.sum(out_grad.data, axis=0))
      )


class Sum(Function):
  def forward(self, x) -> Tensor:
    return Tensor(np.sum(x))

  def backward(self, out_grad: Tensor):
    x = self.prev[0]
    x.grad = Tensor(np.broadcast_to(out_grad.data, x.shape))


class Matmul(Function):
  def forward(self, x, y) -> Tensor:
    return Tensor(x @ y)

  def backward(self, out_grad: Tensor):
    x, y = self.prev
    x.grad = Tensor(out_grad.data @ y.data.T)
    y.grad = Tensor(x.T.data @ out_grad.data)


class Relu(Function):
  def forward(self, x) -> Tensor:
    return Tensor(np.maximum(x, 0))

  def backward(self, out_grad: Tensor):
    self.prev[0].grad = Tensor(out_grad.data * (self.prev[0].data > 0))


class Softmax(Function):
  def forward(self, x) -> Tensor:
    if x.ndim == 1:
      x = x.reshape(1, -1)
    max_x = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    res = exp_x / sum_exp_x
    self.res = res
    return Tensor(res)

  def backward(self, out_grad: Tensor):
    grad = self.res * (
      out_grad.data - np.sum(out_grad.data * self.res, axis=1, keepdims=True)
    )
    self.prev[0].grad = Tensor(grad)


#
#
# class Mul(Function):
#   def forward(self, x, y) -> Tensor:
#     return Tensor(x * y)
#
#   def backward(self, out_grad: Tensor):
#     self.prev[0].grad = out_grad * self.prev[1].data
#     self.prev[1].grad = out_grad * self.prev[0].data
#
#
# class Matmul(Function):
#   def forward(self, x, y) -> Tensor:
#     return Tensor(x @ y)
#
#   def backward(self, out_grad: Tensor):
#     self.prev[0].grad = out_grad @ self.prev[1].T
#     self.prev[1].grad = (out_grad.T @ self.prev[0].data).T
