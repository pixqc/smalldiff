from __future__ import annotations

from typing import List, Optional, Union

import numpy as np


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
  def ndim(self):
    return self.data.ndim

  @property
  def T(self):
    return Tensor(self.data.T)

  @property
  def dtype(self):
    return self.data.dtype

  @staticmethod
  def rand(*shape, **kwargs):
    return Tensor(np.random.randn(*shape), **kwargs)

  @staticmethod
  def ones_like(*shape) -> Tensor:
    return Tensor(np.ones(shape))

  @staticmethod
  def zeros_like(*shape) -> Tensor:
    return Tensor(np.zeros(shape))

  def numpy(self):
    return self.data

  def relu(self):
    return Relu.apply(self)

  def tanh(self):
    return Tanh.apply(self)

  def recip(self):
    return Recip.apply(self)

  def log(self):
    return Log.apply(self)

  def exp(self):
    return Exp.apply(self)

  def neg(self):
    return self * (-1)

  def reciprocal(self):
    return self.recip()  # tinygrad compat

  def add(self, x, reverse=False):
    if reverse:
      self, x = x, self
    return Add.apply(self, x)

  def mul(self, x, reverse=False):
    if reverse:
      self, x = x, self
    return Mul.apply(self, x)

  def sub(self, x, reverse=False):
    if reverse:
      self, x = x, self
    return self + (-x)

  def div(self, x, reverse=False):
    x = Tensor(x) if not isinstance(x, Tensor) else x
    if reverse:
      self, x = x, self
    return self * x.recip()

  def dot(self, x):
    return Dot.apply(self, x)

  def matmul(self, x):
    return self.dot(x)

  # fmt: off
  def pow(self, x):
    if x == -1: return self.recip()
    elif x == 0: return 1 + self * 0
    elif x == 1: return self
    elif x == 2: return self * self
    elif x == 3: return self * self * self
    else: NotImplementedError(f"pow({self}, {x}) not implemented")  # lol
  # fmt: on

  def max(self, axis=None, keepdim=False):
    return Max.apply(self, axis=axis, keepdim=keepdim)

  def sum(self, axis=None, keepdim=False):
    return Sum.apply(self, axis=axis, keepdim=keepdim)

  def mean(self, axis=None, keepdim=False):
    divisor = np.prod(self.shape) if axis is None else self.shape[axis]
    return self.sum(axis=axis, keepdim=keepdim) / divisor

  def softmax(self, axis: Optional[int] = -1):
    t = self
    kwargs = {"axis": axis, "keepdim": True}
    return (t - t.max(**kwargs)).exp() / (t - t.max(**kwargs)).exp().sum(**kwargs)

  def log_softmax(self, axis: Optional[int] = -1):
    return self.softmax(axis=axis).log()

  def cross_entropy(self, y):
    y_oh = np.eye(self.shape[-1])[y.data]
    return -self.log_softmax(axis=1).mul(y_oh).sum(axis=1).mean()

  __add__ = add
  __mul__ = mul
  __sub__ = sub
  __neg__ = neg
  __pow__ = pow
  __matmul__ = dot
  __truediv__ = div

  def __radd__(self, x):
    return self.add(x, reverse=True)

  def __rmul__(self, x):
    return self.mul(x, reverse=True)

  def __rsub__(self, x):
    return self.sub(x, reverse=True)

  def __rdiv__(self, x):
    return self.div(x, reverse=True)

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
    return f"<Tensor {self.data!r} grad_fn={self._ctx!r}>"


class Function:
  def __init__(self, *tensors: Tensor):
    self.requires_grad = any([t.requires_grad for t in tensors])
    self.prev = tensors if self.requires_grad else None

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


class Relu(Function):
  def forward(self, x) -> Tensor:
    self.x = x
    return Tensor(np.maximum(x, 0))

  def backward(self, out_grad: Tensor):
    if not self.prev or not self.prev[0].requires_grad:
      return
    grad = out_grad * (self.x > 0)
    self.prev[0].grad = grad if self.prev[0].grad is None else self.prev[0].grad + grad


class Tanh(Function):
  def forward(self, x) -> Tensor:
    self.res = Tensor(np.tanh(x))
    return self.res

  def backward(self, out_grad: Tensor):
    if not self.prev or not self.prev[0].requires_grad:
      return
    grad = out_grad * (1 - self.res**2)  #  type: ignore
    self.prev[0].grad = grad if self.prev[0].grad is None else self.prev[0].grad + grad


class Recip(Function):
  def forward(self, x) -> Tensor:
    self.x = x
    return Tensor(1 / x)

  def backward(self, out_grad: Tensor):
    if not self.prev or not self.prev[0].requires_grad:
      return
    grad = -out_grad / (self.x**2)
    self.prev[0].grad = grad if self.prev[0].grad is None else self.prev[0].grad + grad


class Log(Function):
  def forward(self, x) -> Tensor:
    self.x = x
    return Tensor(np.log(x))

  def backward(self, out_grad: Tensor):
    if not self.prev or not self.prev[0].requires_grad:
      return
    grad = out_grad / self.x
    self.prev[0].grad = grad if self.prev[0].grad is None else self.prev[0].grad + grad


class Exp(Function):
  def forward(self, x) -> Tensor:
    self.res = Tensor(np.exp(x))
    return self.res

  def backward(self, out_grad: Tensor):
    if not self.prev or not self.prev[0].requires_grad:
      return
    grad = out_grad * self.res
    self.prev[0].grad = grad if self.prev[0].grad is None else self.prev[0].grad + grad


class Add(Function):
  def forward(self, x: Tensor, y: Tensor) -> Tensor:
    return Tensor(x + y)

  def backward(self, out_grad: Tensor):
    if not self.prev:
      return
    for t in self.prev:
      if t.requires_grad:
        sum_axes = tuple(
          i for i in range(out_grad.ndim) if t.shape[i] == 1 and out_grad.shape[i] != 1
        )
        grad = out_grad.data.sum(axis=sum_axes, keepdims=True)
        grad = Tensor(grad.reshape(t.shape))
        t.grad = grad if t.grad is None else t.grad + grad


class Mul(Function):
  def forward(self, x, y) -> Tensor:
    return Tensor(x * y)

  def backward(self, out_grad: Tensor):
    if not self.prev:
      return
    for i, t in enumerate(self.prev):
      if t.requires_grad:
        other = self.prev[1 - i]
        sum_axes = tuple(
          i for i in range(out_grad.ndim) if t.shape[i] == 1 and out_grad.shape[i] != 1
        )
        grad = out_grad * other
        grad = grad.data.sum(axis=sum_axes, keepdims=True)
        grad = Tensor(grad.reshape(t.shape))
        t.grad = grad if t.grad is None else t.grad + grad


class Dot(Function):
  def forward(self, x, y) -> Tensor:
    return Tensor(x @ y)

  def backward(self, out_grad: Tensor):
    if not self.prev:
      return
    x, y = self.prev
    x.grad = out_grad @ y.T if x.grad is None else x.grad + out_grad @ y.T
    y.grad = x.T @ out_grad if y.grad is None else y.grad + x.T @ out_grad


# should np be removed from max and sum here?
class Max(Function):
  def forward(self, x, axis=None, keepdim=False) -> Tensor:
    self.x = x
    self.axis = axis
    self.keepdim = keepdim
    return Tensor(np.max(x, axis=axis, keepdims=keepdim))

  def backward(self, out_grad: Tensor):
    if self.prev is None:
      return
    prev = self.prev[0]
    max_values = np.max(self.x.data, axis=self.axis, keepdims=True)
    mask = np.equal(self.x.data, max_values)
    if self.axis is not None and not self.keepdim:
      out_grad.data = np.expand_dims(out_grad.data, axis=self.axis)
    grad = Tensor(mask * out_grad.data)
    prev.grad = grad if prev.grad is None else prev.grad + grad


class Sum(Function):
  def forward(self, x, axis=None, keepdim=False) -> Tensor:
    self.axis = axis
    self.keepdim = keepdim
    return Tensor(np.sum(x, axis=axis, keepdims=keepdim))

  def backward(self, out_grad: Tensor):
    if self.prev is None:
      return
    prev = self.prev[0]
    if self.axis is None:
      prev.grad = Tensor(np.broadcast_to(out_grad.data, prev.shape))
    else:
      if not self.keepdim:
        out_grad.data = np.expand_dims(out_grad.data, axis=self.axis)
      grad = Tensor(np.broadcast_to(out_grad.data, prev.shape))
      prev.grad = grad if prev.grad is None else prev.grad + grad
