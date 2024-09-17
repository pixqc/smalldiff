# attempting to explain why cross entropy is used as loss function in ml
# https://chatgpt.com/c/66e97d74-19f8-8004-a9b3-b2e91002e58f

# to run: ls examples/cross_entropy.py | entr -s 'PYTHONPATH="." python3 examples/cross_entropy.py'

import numpy as np


class CharGen:
  # distribution of lowercase alphabet
  distributions = {
    "uniform": np.full(26, 1.0 / 26.0),
    "3bit": np.full(8, 0.125),
    # 3bit-skewed: think huffman tree
    "3bit-skewed": np.array([0.5**i if i < 7 else 0.5**7 for i in range(1, 9)]),
    "1bit": np.array([0.5, 0.5]),
    "1bit-skewed": np.array([0.8, 0.2]),
    "0bit": np.array([1.0]),
  }

  def __init__(self, type_):
    if type_ not in self.distributions:
      raise ValueError(f"Unknown distribution type: {type_}")
    self.dist = self.distributions[type_]
    # 97 is 'a' in ascii
    self.chars = np.array([chr(97 + i) for i in range(len(self.dist))])

  def _full_distribution(self):
    tmp = np.zeros(26)
    tmp[: len(self.dist)] = self.dist
    return tmp

  def sample(self, count):
    idxs = np.random.choice(len(self.dist), size=count, p=self.dist)
    return "".join(self.chars[idxs])

  # how varied are the samples? how hard is it to predict?
  def entropy(self):
    return -np.sum(self.dist * np.log2(self.dist + 1e-12))

  # if i encode message from self with code optimized for other,
  # what is the average amount of bits required?
  def cross_entropy(self, other):
    self_full = self._full_distribution()
    other_full = other._full_distribution()
    return -np.sum(self_full * np.log2(other_full + 1e-12))

  def kl_divergence(self, other):
    return self.cross_entropy(other) - self.entropy()


if __name__ == "__main__":
  cg1 = CharGen("3bit-skewed")
  print(cg1.sample(10))
