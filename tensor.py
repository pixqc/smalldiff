from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

import numpy as np

# not used (for now), just for reference
UnaryOps = Enum("UnaryOps", ["EXP2", "LOG2", "CAST", "BITCAST", "SIN", "SQRT", "RECIP"])
BinaryOps = Enum("BinaryOps", ["ADD", "MUL", "IDIV", "MAX", "MOD", "CMPLT", "CMPNE", "XOR"])  # fmt: skip
ReduceOps = Enum("ReduceOps", ["SUM", "PROD", "MAX"])
TernaryOps = Enum("TernaryOps", ["WHERE", "MULACC"])
MetaOps = Enum("MetaOps", ["EMPTY", "CONST", "COPY", "CONTIGUOUS", "CUSTOM", "ASSIGN", "VIEW"])  # fmt: skip
# MovementOps doesn't actually exist in tinygrad
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "SHRINK", "STRIDE", "EXPAND", "PAD"])  # fmt: skip
Op = Union[UnaryOps, BinaryOps, ReduceOps, MetaOps, TernaryOps]


class Tensor:
  def __init__(
    self,
    data: Union[Tensor, int, float, np.ndarray, List],
    dtype=None,
    requires_grad=False,
  ):
    self.data = np.array(data, dtype=dtype)
    self.grad: Optional[Tensor] = None
    self._ctx: Optional[Function] = None
    self.requires_grad = requires_grad

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

  def numpy(self):
    return self.data

  # fmt: off
  # ----- primitive operations -----
  # unary
  def relu(self): return Relu.apply(self)
  def recip(self): return Recip.apply(self)
  def log(self): return Log.apply(self)
  def exp(self): return Exp.apply(self)
  def neg(self): return self * (-1)

  # binary
  def add(self, x): return Add.apply(self, x)
  def mul(self, x): return Mul.apply(self, x)
  def sub(self, x): return self + (-x)
  def div(self, x): return self * x.recip()

  # reduce
  def max(self, axis=None, keepdim=False): return Max.apply(self, axis=axis, keepdim=keepdim)
  def sum(self, axis=None, keepdim=False): return Sum.apply(self, axis=axis, keepdim=keepdim)
  def mean(self, axis=None, keepdim=False): return Mean.apply(self, axis=axis, keepdim=keepdim)
  # fmt: on

  # ----- composite operations -----

  def dot(self, x):
    pass  # TODO: impl

  def matmul(self, x):
    return self.dot(x)

  def _softmax(self, axis):
    m = self - self.max(axis=axis, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis=None):
    _, e, ss = self._softmax(axis)
    return e.div(ss)

  def log_softmax(self, axis=None):
    m, _, ss = self._softmax(axis)
    return m - ss.log()

  __add__ = add
  __mul__ = mul
  __sub__ = sub
  __neg__ = neg
  __pow__ = pow
  __matmul__ = matmul

  # ----- backward -----

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

  def __repr__(self):
    grad_repr = self.grad.data if self.grad else None
    return f"<Tensor {self.data!r} with grad {grad_repr!r}>"


class Function:
  def __init__(self, *tensors: Tensor):
    self.prev = tensors
    self.requires_grad = any([t.requires_grad for t in tensors])

  def forward(self, *args, **kwargs):
    raise NotImplementedError(f"forward not implemented for {type(self)}")

  def backward(self, *args, **kwargs):
    raise NotImplementedError(f"backward not implemented for {type(self)}")

  @classmethod
  def apply(cls, *x: Union[Tensor, int, float, np.ndarray, List], **kwargs):
    ensure_tensor = lambda x: x if isinstance(x, Tensor) else Tensor(x)
    tensors = [ensure_tensor(t) for t in x]
    fn: Function = cls(*tensors)
    res: Tensor = fn.forward(*[t.data for t in tensors], **kwargs)
    res.requires_grad = fn.requires_grad
    res._ctx = fn if fn.requires_grad else None
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


class Recip(Function):
  def forward(self, x) -> Tensor:
    return Tensor(1 / x)


class Mul(Function):
  def forward(self, x, y) -> Tensor:
    return Tensor(x * y)

  # def backward(self, out_grad: Tensor):
  #   self.prev[0].grad = out_grad * self.prev[1].data
  #   self.prev[1].grad = out_grad * self.prev[0].data


class Mean(Function):
  def forward(self, x, axis=None, keepdim=False) -> Tensor:
    return Tensor(np.mean(x, axis=axis, keepdims=keepdim))


class Relu(Function):
  def forward(self, x) -> Tensor:
    return Tensor(np.maximum(x, 0))

  def backward(self, out_grad: Tensor):
    self.prev[0].grad = Tensor(out_grad.data * (self.prev[0].data > 0))


class Max(Function):
  def forward(self, x, axis=None, keepdim=False) -> Tensor:
    return Tensor(np.max(x, axis=axis, keepdims=keepdim))

  # def backward(self, out_grad: Tensor):
  #   self.prev[0].grad = Tensor(np.zeros_like(self.prev[0].data))


class Log(Function):
  def forward(self, x) -> Tensor:
    return Tensor(np.log(x))

  # def backward(self, out_grad: Tensor):
  #   self.prev[0].grad = out_grad / self.prev[0].data


class Exp(Function):
  def forward(self, x) -> Tensor:
    return Tensor(np.exp(x))
