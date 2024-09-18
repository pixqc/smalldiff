# inspired by https://github.com/srush/Tensor-Puzzles
# to run: ls examples/tensor_puzzles.py | entr -s 'PYTHONPATH="." python3 examples/tensor_puzzles.py'

# constraints:
# - each puzzle to be solved in 1 line (<80 columns) of code
# - allowed @, arithmetic, comparison, shape, any indexing
#   (e.g. a[:j], a[:, None], a[arange(10)]), and previous puzzle functions
# - allowable "base/primitive functions": Tensor.where and arange
# - NOT allowed: anything else: no view, sum, take, squeeze, tensor


import numpy as np

# TODO: s/tinygrad.tensor/tensor and everything should work
from tinygrad.tensor import Tensor

# TODO: triu, cumsum, diff, vstack, roll, flip, compress, pad_to, sequence_mask, bincount, scatter_add, flatten, linspace, heaviside, repeat, bucketize

arange = Tensor.arange


# puzzle 1: ones
def ones(i: int) -> Tensor:
  return (arange(i) >= 0).where(1, 0)


# puzzle 2: sum
def sum(t: Tensor) -> Tensor:
  return ones(t.shape[0]) @ t  # type: ignore (only using int)


# puzzle 3: outer product
def outer(t1: Tensor, t2: Tensor) -> Tensor:
  return t1[:, None] @ t2[None, :]


# puzzle 4: diagonal
def diag(t: Tensor) -> Tensor:
  return t[arange(t.shape[0]), arange(t.shape[0])]


# puzzle 5: eye/identity
def eye(i: int) -> Tensor:
  return arange(i)[:, None] == arange(i)


# puzzle 6: triu
def triu(t: Tensor) -> Tensor:
  return (arange(t.shape[0])[:, None] <= arange(t.shape[0])).where(1, 0) * t


# puzzle 7: cumsum
def cumsum(t: Tensor) -> Tensor:
  return sum(triu(t))


# --- tests ---

for _ in range(3):
  i = np.random.randint(100)
  assert np.allclose(np.ones(i), ones(i).numpy())

for _ in range(3):
  shape = np.random.randint(1, 10)
  t = Tensor(np.random.rand(shape).astype(np.float32))
  assert np.allclose(np.sum(t.numpy()), sum(t).numpy())

for _ in range(3):
  shape1 = np.random.randint(1, 10)
  shape2 = np.random.randint(1, 10)
  t1 = Tensor(np.random.rand(shape1).astype(np.float32))
  t2 = Tensor(np.random.rand(shape2).astype(np.float32))
  assert np.allclose(np.outer(t1.numpy(), t2.numpy()), outer(t1, t2).numpy())

for _ in range(3):
  shape = np.random.randint(1, 10)
  t = Tensor(np.random.randn(shape, shape).astype(np.float32))
  assert np.allclose(np.diag(t.numpy()), diag(t).numpy())

for _ in range(3):
  i = np.random.randint(1, 10)
  assert np.allclose(np.eye(i), eye(i).numpy())

for _ in range(3):
  shape = np.random.randint(1, 10)
  t = Tensor(np.random.randn(shape, shape).astype(np.float32))
  assert np.allclose(np.triu(t.numpy()), triu(t).numpy())

for _ in range(3):
  shape = np.random.randint(1, 10)
  t = Tensor(np.random.rand(shape).astype(np.float32))
  assert np.allclose(np.cumsum(t.numpy()), cumsum(t).numpy())

print("all tests passed!")
