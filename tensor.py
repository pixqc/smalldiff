from __future__ import annotations

from typing import List, Optional, Union

import numpy as np


class Tensor:
  def __init__(self, data: Union[Tensor, int, float, np.ndarray, List], dtype=None):
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

  # primitive operations

  def add(self, other):
    return Add.apply(self, other)

  def sub(self, other):
    return self + (-other)

  def sum(self, axis=None, keepdim=False):
    return Sum.apply(self, axis=axis, keepdim=keepdim)

  def mul(self, other):
    return Mul.apply(self, other)

  def mean(self):
    return Mean.apply(self)

  def matmul(self, other):
    return Matmul.apply(self, other)

  def div(self, other):
    return Div.apply(self, other)

  def relu(self):
    return Relu.apply(self)

  def max(self, axis=-1, keepdim=False):
    return Max.apply(self, axis=axis, keepdim=keepdim)

  def neg(self):
    return Neg.apply(self)

  def log(self):
    return Log.apply(self)

  def exp(self):
    return Exp.apply(self)

  # composite operations

  def _softmax(self, axis):
    m = self - self.max(axis=axis, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis=-1):
    _, e, ss = self._softmax(axis)
    return e.div(ss)

  def log_softmax(self, axis=-1):
    m, _, ss = self._softmax(axis)
    return m - ss.log()

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
  __mul__ = mul
  __sub__ = sub
  __neg__ = neg
  __matmul__ = matmul


class Function:
  def __init__(self, *tensors: Tensor):
    self.prev = tensors

  def forward(self, *args, **kwargs):
    raise NotImplementedError(f"forward not implemented for {type(self)}")

  def backward(self, *args, **kwargs):
    raise NotImplementedError(f"backward not implemented for {type(self)}")

  @classmethod
  def apply(cls, *x: Union[Tensor, int, float, np.ndarray, List], **kwargs):
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
  def forward(self, x, axis=None, keepdim=False) -> Tensor:
    return Tensor(np.sum(x, axis=axis, keepdims=keepdim))

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


class Div(Function):
  def forward(self, x, y) -> Tensor:
    return Tensor(x / y)


class Mul(Function):
  def forward(self, x, y) -> Tensor:
    return Tensor(x * y)

  # def backward(self, out_grad: Tensor):
  #   self.prev[0].grad = out_grad * self.prev[1].data
  #   self.prev[1].grad = out_grad * self.prev[0].data


class Mean(Function):
  def forward(self, x) -> Tensor:
    return Tensor(np.mean(x))


class Relu(Function):
  def forward(self, x) -> Tensor:
    return Tensor(np.maximum(x, 0))

  def backward(self, out_grad: Tensor):
    self.prev[0].grad = Tensor(out_grad.data * (self.prev[0].data > 0))


class Max(Function):
  def forward(self, x, axis=-1, keepdim=False) -> Tensor:
    return Tensor(np.max(x, axis=axis, keepdims=keepdim))

  def backward(self, out_grad: Tensor):
    self.prev[0].grad = Tensor(np.zeros_like(self.prev[0].data))


class Neg(Function):
  def forward(self, x) -> Tensor:
    return Tensor(-x)

  # def backward(self, out_grad: Tensor):
  #   self.prev[0].grad = -out_grad


class Log(Function):
  def forward(self, x) -> Tensor:
    return Tensor(np.log(x))

  # def backward(self, out_grad: Tensor):
  #   self.prev[0].grad = out_grad / self.prev[0].data


class Exp(Function):
  def forward(self, x) -> Tensor:
    return Tensor(np.exp(x))


#
# # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# class SparseCategoricalCrossEntropy(Function):
#   def forward(self, y_pred, y_gt) -> Tensor:
#     preds = y_pred[np.arange(y_gt.shape[0]), y_gt]
#     out = -np.log(preds + 1e-8).mean()
#     return Tensor(out)
