# inspired by https://github.com/xjdr-alt/simple_transformer
# to run: ls examples/gpt2.py | entr -s 'PYTHONPATH="." python3 examples/gpt2.py'

from collections import Counter
from typing import List, NamedTuple

import numpy as np
import tinygrad.nn as nn
from tinygrad.nn.optim import AdamW
from tinygrad.tensor import Tensor

from helpers import load_shakespeare, tqdm

# -- input processing --


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


## -- neural nets --


class LayerParams(NamedTuple):
  w_q_dhk: Tensor
  w_k_dhk: Tensor
  w_v_dhk: Tensor
  w_o_hkd: Tensor
  w_ff1_dh: Tensor
  w_ff2_hd: Tensor
  b_ff1_h: Tensor
  b_ff2_d: Tensor


class TransformerParams(NamedTuple):
  layer_params: List[LayerParams]
  emb_vd: nn.Embedding
  w_dk: Tensor
  b_d: Tensor


class Size(NamedTuple):
  B: int  # batch size
  L: int  # sequence length
  D: int  # model dimension
  H: int  # number of attention heads in a layer
  K: int  # size of each attention key or value
  V: int  # vocab size


def prep_input(tokens, sz: Size):
  xs = np.array([tokens[i : i + sz.L] for i in range(len(tokens) - sz.L)])
  ys = np.array([tokens[i + 1 : i + sz.L + 1] for i in range(len(tokens) - sz.L)])
  return xs, ys


def sample_batch(xs, ys, sz: Size):
  idxs = np.random.choice(xs.shape[0], sz.B, replace=False)
  ys = ys[idxs]
  one_hot = np.zeros((sz.B, sz.L, sz.V)).astype(np.uint32)
  one_hot[np.arange(sz.B)[:, None], np.arange(sz.L), ys] = 1
  return Tensor(xs[idxs]), Tensor(one_hot)


def get_pe(sz: Size):
  p = Tensor.arange(sz.L)
  d = (Tensor.arange(0, sz.D, 2) * -(Tensor(10000).log() / sz.D)).exp()
  pd = p * d
  return pd.sin().stack(pd.cos()).repeat(sz.D // 2, 1).T


def norm(x, eps=1e-5):
  kwargs = {"axis": -1, "keepdim": True}
  return (x - x.mean(**kwargs)) / (x.std(**kwargs) + eps)


def attention(input_bld: Tensor, params: LayerParams, sz: Size):
  q_blhk = Tensor.einsum("bld,dhk->blhk", input_bld, params.w_q_dhk)
  k_blhk = Tensor.einsum("bld,dhk->blhk", input_bld, params.w_k_dhk)
  v_blhk = Tensor.einsum("bld,dhk->blhk", input_bld, params.w_v_dhk)
  scores_bhll = Tensor.einsum("bihk,bjhk->bhij", q_blhk, k_blhk)
  mask = Tensor.where(Tensor.tril(Tensor.ones((sz.L, sz.L))) == 1, 0.0, -np.inf)
  scores_bhll = ((scores_bhll + mask) * sz.K**-0.5).softmax(axis=-1)
  out_blhk = Tensor.einsum("bhij,bjhk->bihk", scores_bhll, v_blhk)
  out_bld = Tensor.einsum("blhk,hkd->bld", out_blhk, params.w_o_hkd)
  return out_bld


def ffw(input_bld: Tensor, params: LayerParams):
  return (
    input_bld @ params.w_ff1_dh + params.b_ff1_h
  ).gelu() @ params.w_ff2_hd + params.b_ff2_d


def transformer(xs: Tensor, params: TransformerParams, sz: Size):
  pe = get_pe(sz)
  input_bld = params.emb_vd(xs) + pe
  for layer_params in params.layer_params:
    input_bld = input_bld + attention(norm(input_bld), layer_params, sz)
    input_bld = input_bld + ffw(norm(input_bld), layer_params)
  return input_bld @ params.w_dk + params.b_d


def loss(y_pred, ys):
  return -y_pred.log_softmax(axis=2).mul(ys).sum(axis=2).mean()


def params(params: TransformerParams):
  layer_attrs = [
    "w_q_dhk",
    "w_k_dhk",
    "w_v_dhk",
    "w_o_hkd",
    "w_ff1_dh",
    "b_ff1_h",
    "w_ff2_hd",
    "b_ff2_d",
  ]
  transformer_attrs = ["w_dk", "b_d"]
  return (
    [getattr(layer, attr) for layer in params.layer_params for attr in layer_attrs]
    + [getattr(params, attr) for attr in transformer_attrs]
    + [params.emb_vd.weight]
  )


if __name__ == "__main__":
  text = load_shakespeare()[:100000]
  tokenizer = Tokenizer(text=text, iters=500)
  tokens = tokenizer.encode(text)
  sz = Size(64, 16, 32, 4, 32 // 4, len(tokenizer.vocab))
  xs, ys = prep_input(tokens, sz)
  xs, ys = sample_batch(xs, ys, sz)  # (B, L) (B, L, V)

  l_params = LayerParams(
    Tensor.uniform(sz.D, sz.H, sz.K),
    Tensor.uniform(sz.D, sz.H, sz.K),
    Tensor.uniform(sz.D, sz.H, sz.K),
    Tensor.uniform(sz.H, sz.K, sz.D),
    Tensor.uniform(sz.D, sz.H),
    Tensor.uniform(sz.H, sz.D),
    Tensor.zeros(sz.H),
    Tensor.zeros(sz.D),
  )
  t_params = TransformerParams(
    [l_params],
    nn.Embedding(sz.V, sz.D),
    Tensor.uniform(sz.D, sz.V),
    Tensor.zeros(sz.V),
  )

  optimizer = AdamW(params(t_params), lr=1e-3)
  with Tensor.train():
    for step in tqdm(range(5000), desc="training gpt2", unit="step"):
      optimizer.zero_grad()
      out_bld = transformer(xs, t_params, sz)
      loss_val = loss(out_bld, ys)
      loss_val.backward()
      optimizer.step()

      if step % 100 == 0:
        print(f"step {step}: loss={loss_val.numpy():.2f}")
