# inspired by https://github.com/srush/Tensor-Puzzles
# to run: ls examples/tensor_puzzles.py | entr -s 'PYTHONPATH="." python3 examples/tensor_puzzles.py'

# constraints:
# - each puzzle to be solved in 1 line (<80 columns) of code
# - allowed @, arithmetic, comparison, shape, any indexing
#   (e.g. a[:j], a[:, None], a[arange(10)]), and previous puzzle functions
# - NOT allowed: anything else: no view, sum, take, squeeze, tensor


import numpy as np

# TODO: s/tinygrad.tensor/tensor and everything should work
from tinygrad.tensor import Tensor


# puzzle 1: ones
def ones(i: int) -> Tensor:
  return (Tensor.arange(i) >= 0).where(1, 0)


# puzzle 2: sum
def sum(t: Tensor) -> Tensor:
  return ones(t.shape[0]) @ t  # type: ignore (only using int)


# puzzle 3: outer product
def outer(t1: Tensor, t2: Tensor) -> Tensor:
  return t1[:, None] @ t2[None, :]


def diag(t: Tensor) -> Tensor:
  pass


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

print("all tests passed!")
