# to run: ls examples/gpt2.py | entr -s 'PYTHONPATH="." python3 examples/gpt2.py'

from collections import Counter

import numpy as np
from tinygrad.nn import Embedding
from tinygrad.tensor import Tensor

from helpers import load_shakespeare


class Tokenizer:
  def __init__(self, text, iters=10):
    self.iters = iters
    self.m, self.rm = self.map_tokens(text)
    self.vocab = set(list(self.m.keys()) + self.to_byte_list(text))
    self.idx, self.ridx = self.index_tokens()

  @staticmethod
  def to_byte_list(text):
    return list(text.encode("utf-8"))

  @staticmethod
  def to_text(byte_list):
    return bytes(byte_list).decode("utf-8")

  def get_max_pair(self, byte_list):
    if len(byte_list) < 2:
      return None
    pair_counts = Counter(zip(byte_list, byte_list[1:]))
    most_common = pair_counts.most_common(1)
    if not most_common or most_common[0][1] == 1:
      return None
    return most_common[0][0]

  def merge_max_pair(self, byte_list, max_pair, new_token):
    merged = []
    i = 0
    while i < len(byte_list):
      if i < len(byte_list) - 1 and (byte_list[i], byte_list[i + 1]) == max_pair:
        merged.append(new_token)
        i += 2
      else:
        merged.append(byte_list[i])
        i += 1
    return merged

  def map_tokens(self, text):
    byte_list = self.to_byte_list(text)
    m = {}

    for i in range(self.iters):
      max_pair = self.get_max_pair(byte_list)
      if max_pair is None:
        break
      new_token = 256 + i
      byte_list = self.merge_max_pair(byte_list, max_pair, new_token)
      m[new_token] = max_pair

    rm = {v: k for k, v in m.items()}
    return m, rm

  def index_tokens(self):
    idx = {v: i for i, v in enumerate(self.vocab)}
    ridx = {i: v for i, v in enumerate(self.vocab)}
    return idx, ridx

  def tokenize(self, byte_list):
    tokens = byte_list.copy()
    i = 0
    while i < len(tokens) - 1:
      pair = (tokens[i], tokens[i + 1])
      if pair in self.rm:
        tokens[i] = self.rm[pair]
        del tokens[i + 1]
        # after merging, check previous token again
        if i > 0:
          i -= 1
      else:
        i += 1
    return tokens

  def detokenize(self, tokens):
    tokens = tokens.copy()
    # must be sorted to do it in one pass
    sorted_merges = sorted(self.m.items(), key=lambda x: x[0], reverse=True)
    for new_token, pair in sorted_merges:
      i = 0
      while i < len(tokens):
        if tokens[i] == new_token:
          tokens[i : i + 1] = list(pair)
        i += 1
    return tokens

  def encode(self, text):
    byte_list = self.to_byte_list(text)
    tokens = self.tokenize(byte_list)
    return [self.idx[token] for token in tokens]

  def decode(self, tokens):
    byte_list = self.detokenize([self.ridx[token] for token in tokens])
    return self.to_text(byte_list)


def prep_input(tokens, T):
  trils = []
  ys = []

  # neat trick to right-pad inputs
  for i in range(len(tokens) - T + 1):
    tmp = np.tril(tokens[i : i + T])
    trils.append(tmp)
    tmp_y = tokens[i + 1 : i + T + 1]
    ys.append(tmp_y)

  xs = np.vstack(trils)[:-1]  # last one doesn't have a y
  ys = np.concatenate(ys)
  return xs, ys


class Head:
  def __init__(self, C, head_size):
    self.C = C
    self.head_size = head_size

    self.W_q = Tensor.randn(C, head_size, requires_grad=True)
    self.W_k = Tensor.randn(C, head_size, requires_grad=True)
    self.W_v = Tensor.randn(C, head_size, requires_grad=True)

  def __call__(self, pe):
    Q = pe @ self.W_q
    K = pe @ self.W_k
    V = pe @ self.W_v

    # shape = (B, T, T) "how much each token attend to each other"; high = relevant
    scores = Q @ K.transpose(2, 1) / Tensor(Q.shape[-1]).sqrt()
    # only pay attention to the prev tokens, not the future ones
    mask = Tensor.tril(Tensor.ones((pe.shape[1], pe.shape[1])))
    scores = Tensor.where(mask == 1, scores, -np.inf)
    scores = scores.softmax()
    return scores @ V


class Attention:
  def __init__(self, C, n_heads):
    self.heads = [Head(C, C // n_heads) for _ in range(n_heads)]
    self.W_o = Tensor.randn(C, C, requires_grad=True)

  def __call__(self, pe):
    ho = [h(pe) for h in self.heads]
    c_ho = ho[0]  # might be a better way to do this
    for i in range(1, len(ho)):
      c_ho = c_ho.cat(ho[i], dim=-1)
    return c_ho @ self.W_o


if __name__ == "__main__":
  text = load_shakespeare()[:100]
  tokenizer = Tokenizer(text=text, iters=100)
  tokens = tokenizer.encode(text)

  B, T, C = 64, 16, 32

  xs, ys = prep_input(tokens, T)
  idxs = np.random.choice(xs.shape[0], B, replace=False)
  xs, ys = Tensor(xs[idxs]), Tensor(ys[idxs])

  # positional encoding
  p = Tensor.arange(T)
  d = (Tensor.arange(0, C, 2) * -(Tensor(10000).log() / C)).exp()
  sin_p = (p * d).sin()
  cos_p = (p * d).cos()
  pe = sin_p.stack(cos_p).repeat(B // 4, 1).T
  emb = Embedding(len(tokenizer.vocab), C)
  pe = emb(xs) + pe

  attn = Attention(C, n_heads=4)
  output = attn(pe)
  print(output.shape)
