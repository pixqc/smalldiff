import numpy as np
from enum import Enum


OP = Enum("OP", ["UNARY", "BINARY"])


class Tensor:
  def __init__(self, data):
    self.data = np.array(data)
    self.grad = None
    self.prev = []
    self._backward = lambda: None

  def add(self, other):
    return Add.apply(OP.BINARY, self, other)

  def tanh(self):
    return Tanh.apply(OP.UNARY, self)

  def sum(self):
    return Sum.apply(OP.UNARY, self)

  def logsoftmax(self):
    return LogSoftmax.apply(OP.UNARY, self)

  def relu(self):
    return Relu.apply(OP.UNARY, self)

  def mul(self, other):
    return Mul.apply(OP.BINARY, self, other)

  def matmul(self, other):
    return Matmul.apply(OP.BINARY, self, other)

  def __repr__(self):
    return f"Tensor(data={self.data}, grad={self.grad})"

  __add__ = add


class Function:
  @staticmethod
  def forward(*args):
    raise NotImplementedError

  @classmethod
  def apply(cls, *args):
    op = args[0]
    if op == OP.UNARY:
      return cls.forward(args[1].data)
    elif op == OP.BINARY:
      return cls.forward(args[1].data, args[2].data)
    else:
      raise NotImplementedError


class Tanh(Function):
  @staticmethod
  def forward(x):
    return Tensor(np.tanh(x))


class Sum(Function):
  @staticmethod
  def forward(x):
    return Tensor(np.sum(x))


class Add(Function):
  @staticmethod
  def forward(x, y):
    return Tensor(x + y)


class LogSoftmax(Function):
  @staticmethod
  def forward(x):
    max_val = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - max_val)
    log_sum_exp = np.log(np.sum(exp_x, axis=1, keepdims=True))
    return Tensor(x - max_val - log_sum_exp)


class Relu(Function):
  @staticmethod
  def forward(x):
    return Tensor(np.maximum(x, 0))


class Mul(Function):
  @staticmethod
  def forward(x, y):
    return Tensor(x * y)


class Matmul(Function):
  @staticmethod
  def forward(x, y):
    return Tensor(x @ y)
