from __future__ import annotations

import functools
from typing import List, Optional, Union

import numpy as np


class Tensor:
  def __init__(
    self,
    data: Union[Tensor, int, float, np.ndarray, List],
    dtype=None,
    requires_grad=False,
  ):
    data = data.data if isinstance(data, Tensor) else data
    self.data = np.array(data, dtype=dtype)
    self.grad: Optional[np.ndarray] = None
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
  def randn(*shape, **kwargs):
    return Tensor(np.random.randn(*shape), **kwargs)

  @staticmethod
  def zeros(*shape, **kwargs):
    return Tensor(np.zeros(shape), **kwargs)

  @staticmethod
  def zeros_like(*shape) -> Tensor:
    return Tensor(np.zeros(shape))

  @staticmethod
  def ones(*shape, **kwargs):
    return Tensor(np.ones(shape), **kwargs)

  @staticmethod
  def ones_like(*shape) -> Tensor:
    return Tensor(np.ones(shape))

  # -- unary ops --

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

  def sqrt(self):
    return Sqrt.apply(self)

  def rsqrt(self):
    return self.sqrt().recip()

  def neg(self):
    return self * (-1)

  def reciprocal(self):  # tinygrad compat
    return self.recip()

  def square(self):
    return self * self

  def _batchnorm(self, axis=None):  # without affine
    return (self - self.mean(axis=axis)) * self.var(axis=axis).add(1e-5).rsqrt()

  def batchnorm(self):  # use _batchnorm + affine (scale, shift)
    pass

  # -- binary ops --

  def add(self, x, reverse=False):
    return Add.apply(self, x) if not reverse else Add.apply(x, self)

  def mul(self, x, reverse=False):
    return Mul.apply(self, x) if not reverse else Mul.apply(x, self)

  def sub(self, x, reverse=False):
    return self + (-x) if not reverse else self + (-x)

  def div(self, x, reverse=False):
    x = Tensor(x) if not isinstance(x, Tensor) else x
    return self * x.recip() if not reverse else self * x.recip()

  def dot(self, x):
    # can be a composition of mul, add, reshape
    return Dot.apply(self, x)

  def matmul(self, x):
    return self.dot(x)

  # fmt: off
  def pow(self, x):
    if x == -1: return self.recip()
    elif x == 0: return 1 + self * 0
    elif x == 0.5: return self.sqrt()
    elif x == 1: return self
    elif x == 2: return self * self
    elif x == 3: return self * self * self
    else: NotImplementedError(f"pow({self}, {x}) not implemented")  # lol
  # fmt: on

  # -- reduce ops --

  def max(self, axis=None, keepdim=False):
    return Max.apply(self, axis=axis, keepdim=keepdim)

  def sum(self, axis=None, keepdim=False):
    return Sum.apply(self, axis=axis, keepdim=keepdim)

  def mean(self, axis=None, keepdim=False):
    divisor = np.prod(self.shape) if axis is None else self.shape[axis]
    return self.sum(axis=axis, keepdim=keepdim) / divisor

  def var(self, axis=None, keepdim=False, correction=1):
    prod = lambda xs: functools.reduce(lambda x, y: x * y, xs)
    squares = (self - self.mean(axis=axis, keepdim=True)).square()
    n = prod([si for si, so in zip(self.shape, squares.sum(axis=axis, keepdim=True).shape)if si != so])  # fmt: skip
    return squares.sum(axis=axis, keepdim=keepdim).div(max(0, n - correction))

  def std(self, axis=None, keepdim=False, correction=1):
    return self.var(axis=axis, keepdim=keepdim, correction=correction).sqrt()

  def _softmax(self, axis):  # from tinygrad's Tensor.softmax
    m = self - self.max(axis=axis, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis=-1):
    _, e, ss = self._softmax(axis)
    return e.div(ss)

  def log_softmax(self, axis=-1):
    m, _, ss = self._softmax(axis)
    return m - ss.log()

  def cross_entropy(self, y):
    y_oh = np.eye(self.shape[-1])[y.data]
    return -self.log_softmax(axis=1).mul(y_oh).sum(axis=1).mean()

  # fmt: off
  __add__ = add
  __mul__ = mul
  __sub__ = sub
  __neg__ = neg
  __pow__ = pow
  __matmul__ = dot
  __truediv__ = div
  def __radd__(self, x): return self.add(x, reverse=True)
  def __rmul__(self, x): return self.mul(x, reverse=True)
  def __rsub__(self, x): return self.sub(x, reverse=True)
  def __rdiv__(self, x): return self.div(x, reverse=True)
  # fmt: on

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
    self.grad = np.array(1.0)
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
  def forward(self, x: np.ndarray) -> Tensor:
    self.x = x
    return Tensor(np.maximum(self.x, 0))

  def backward(self, out_grad: np.ndarray):
    if not self.prev or not self.prev[0].requires_grad:
      return
    prev: Tensor = self.prev[0]
    grad = out_grad * (self.x > 0)
    prev.grad = grad if prev.grad is None else prev.grad + grad


class Tanh(Function):
  def forward(self, x: np.ndarray) -> Tensor:
    self.res = np.tanh(x)
    return Tensor(self.res)

  def backward(self, out_grad: np.ndarray):
    if not self.prev or not self.prev[0].requires_grad:
      return
    prev: Tensor = self.prev[0]
    grad = out_grad * (1 - self.res**2)
    prev.grad = grad if prev.grad is None else prev.grad + grad


class Recip(Function):
  def forward(self, x: np.ndarray) -> Tensor:
    self.x = x
    return Tensor(1 / self.x)

  def backward(self, out_grad: np.ndarray):
    if not self.prev or not self.prev[0].requires_grad:
      return
    prev: Tensor = self.prev[0]
    grad = -out_grad / (self.x**2)
    prev.grad = grad if prev.grad is None else prev.grad + grad


class Log(Function):
  def forward(self, x: np.ndarray) -> Tensor:
    self.x = x
    return Tensor(np.log(self.x))

  def backward(self, out_grad: np.ndarray):
    if not self.prev or not self.prev[0].requires_grad:
      return
    prev: Tensor = self.prev[0]
    grad = out_grad / self.x
    prev.grad = grad if prev.grad is None else prev.grad + grad


class Exp(Function):
  def forward(self, x: np.ndarray) -> Tensor:
    self.res = np.exp(x)
    return Tensor(self.res)

  def backward(self, out_grad: np.ndarray):
    if not self.prev or not self.prev[0].requires_grad:
      return
    prev: Tensor = self.prev[0]
    grad = out_grad * self.res
    prev.grad = grad if prev.grad is None else prev.grad + grad


class Sqrt(Function):
  def forward(self, x: np.ndarray) -> Tensor:
    self.res = np.sqrt(x)
    return Tensor(self.res)

  def backward(self, out_grad: np.ndarray):
    if not self.prev or not self.prev[0].requires_grad:
      return
    prev: Tensor = self.prev[0]
    grad = out_grad / (2 * self.res)
    prev.grad = grad if prev.grad is None else prev.grad + grad


class Add(Function):
  def forward(self, x: np.ndarray, y: np.ndarray) -> Tensor:
    return Tensor(x + y)

  def backward(self, out_grad: np.ndarray):
    if not self.prev:
      return
    for t in self.prev:
      if t.requires_grad:
        t_shape = (1,) * (out_grad.ndim - t.ndim) + t.shape
        is_sum = lambda i: t_shape[i] == 1 and out_grad.shape[i] != 1
        sum_axes = tuple(i for i in range(out_grad.ndim) if is_sum(i))
        grad = out_grad.sum(axis=sum_axes, keepdims=True)
        grad = grad.reshape(t.shape)
        t.grad = grad if t.grad is None else t.grad + grad


class Mul(Function):
  def forward(self, x: np.ndarray, y: np.ndarray) -> Tensor:
    return Tensor(x * y)

  def backward(self, out_grad: np.ndarray):
    if not self.prev:
      return
    for i, t in enumerate(self.prev):
      if t.requires_grad:
        other = self.prev[1 - i].data
        t_shape = (1,) * (out_grad.ndim - t.ndim) + t.shape
        is_sum = lambda i: t_shape[i] == 1 and out_grad.shape[i] != 1
        sum_axes = tuple(i for i in range(out_grad.ndim) if is_sum(i))
        grad = out_grad * other
        grad = grad.sum(axis=sum_axes, keepdims=True)
        grad = grad.reshape(t.shape)
        t.grad = grad if t.grad is None else t.grad + grad


class Dot(Function):
  def forward(self, x: np.ndarray, y: np.ndarray) -> Tensor:
    return Tensor(x @ y)

  def backward(self, out_grad: np.ndarray):
    if not self.prev:
      return
    x, y = self.prev
    x.grad = out_grad @ y.data.T if x.grad is None else x.grad + out_grad @ y.data.T
    y.grad = x.data.T @ out_grad if y.grad is None else y.grad + x.data.T @ out_grad


class Max(Function):
  def forward(self, x: np.ndarray, axis=None, keepdim=False) -> Tensor:
    self.x = x
    self.axis = axis
    self.keepdim = keepdim
    return Tensor(np.max(x, axis=axis, keepdims=keepdim))

  def backward(self, out_grad: np.ndarray):
    if self.prev is None:
      return
    prev: Tensor = self.prev[0]
    max_values = np.max(self.x, axis=self.axis, keepdims=True)
    mask = np.equal(self.x, max_values)
    if self.axis is not None and not self.keepdim:
      out_grad = np.expand_dims(out_grad, axis=self.axis)
    grad = mask * out_grad
    prev.grad = grad if prev.grad is None else prev.grad + grad


class Sum(Function):
  def forward(self, x: np.ndarray, axis=None, keepdim=False) -> Tensor:
    self.x = x
    self.axis = axis
    self.keepdim = keepdim
    return Tensor(np.sum(self.x, axis=axis, keepdims=keepdim))

  def backward(self, out_grad: np.ndarray):
    if self.prev is None:
      return
    prev: Tensor = self.prev[0]
    if self.axis is None:
      grad = np.broadcast_to(out_grad, prev.shape)
    else:
      if not self.keepdim:
        out_grad = np.expand_dims(out_grad, axis=self.axis)
      grad = np.broadcast_to(out_grad, prev.shape)
    prev.grad = grad if prev.grad is None else prev.grad + grad


class Optimizer:
  def __init__(self, params: List[Tensor]):
    self.params = params

  def step(self):
    NotImplementedError("step not implemented")

  def zero_grad(self):
    for param in self.params:
      param.grad = None
      param._ctx = None  # cleanup computational graph


class SGD(Optimizer):
  def __init__(self, params: List[Tensor], lr, momentum=None, weight_decay=0.0):
    super().__init__(params)
    self.lr = lr
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.prev_grads = [np.zeros_like(p.data) for p in params]

  def step(self):
    for i, p in enumerate(self.params):
      if p.grad is not None:
        g = p.grad
        if self.weight_decay > 0:
          g += self.weight_decay * p.data
        if self.momentum:
          g = self.momentum * self.prev_grads[i] + self.lr * g
        else:
          g = self.lr * g
        p.data -= g
        self.prev_grads[i] = g
    self.zero_grad()


class AdamW(Optimizer):
  def __init__(
    self,
    params: List[Tensor],
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
  ):
    super().__init__(params)
    self.lr = lr
    self.betas = betas
    self.eps = eps
    self.weight_decay = weight_decay
    self.t = 0
    self.ms = [np.zeros_like(p.data) for p in params]
    self.vs = [np.zeros_like(p.data) for p in params]

  def step(self):
    self.t += 1
    for i, p in enumerate(self.params):
      if p.grad is not None:
        g = p.grad
        if self.weight_decay != 0:
          g = g + self.weight_decay * p.data
        self.ms[i] = self.betas[0] * self.ms[i] + (1 - self.betas[0]) * g
        self.vs[i] = self.betas[1] * self.vs[i] + (1 - self.betas[1]) * g**2
        m_hat = self.ms[i] / (1 - self.betas[0] ** self.t)
        v_hat = self.vs[i] / (1 - self.betas[1] ** self.t)
        p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    self.zero_grad()
