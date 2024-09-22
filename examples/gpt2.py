# to run: ls examples/gpt2.py | entr -s 'PYTHONPATH="." python3 examples/gpt2.py'

from collections import Counter
from typing import List

import numpy as np
from tinygrad import nn
from tinygrad.nn.optim import AdamW
from tinygrad.tensor import Tensor

from helpers import load_shakespeare, tqdm


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
  xs = np.array([tokens[i : i + T] for i in range(len(tokens) - T)])
  ys = np.array([tokens[i + 1 : i + T + 1] for i in range(len(tokens) - T)])
  return xs, ys


def sample_batch(xs, ys, sz):
  B, T, _, _, VS = sz
  idxs = np.random.choice(xs.shape[0], B, replace=False)
  ys = ys[idxs]
  one_hot = np.zeros((B, T, VS)).astype(np.uint32)
  one_hot[np.arange(B)[:, None], np.arange(T), ys] = 1
  return Tensor(xs[idxs]), Tensor(one_hot)


class PositionalEncoding:
  def __init__(self, sz):
    _, T, C, _, VS = sz
    self.p = Tensor.arange(T)
    self.d = (Tensor.arange(0, C, 2) * -(Tensor(10000).log() / C)).exp()
    self.sin_p = (self.p * self.d).sin()
    self.cos_p = (self.p * self.d).cos()
    self.emb = nn.Embedding(VS, C)
    self.aa = B // 4  # idk what to name this, TODO

  def __call__(self, xs):
    return self.emb(xs) + self.sin_p.stack(self.cos_p).repeat(self.aa, 1).T

  def params(self):
    return [self.emb.weight]


class Head:
  def __init__(self, C, head_size):
    self.C = C
    self.head_size = head_size

    self.W_q = Tensor.uniform(C, head_size, requires_grad=True)
    self.W_k = Tensor.uniform(C, head_size, requires_grad=True)
    self.W_v = Tensor.uniform(C, head_size, requires_grad=True)

  def __call__(self, pe):
    Q = pe @ self.W_q
    K = pe @ self.W_k
    V = pe @ self.W_v

    # shape = (B, T, T) "how much each token attend to each other"; high = relevant
    scores = (Q @ K.transpose(2, 1)) * self.head_size**-0.5
    # only pay attention to the prev tokens, not the future ones
    mask = Tensor.tril(Tensor.ones((pe.shape[1], pe.shape[1])))
    scores = Tensor.where(mask == 1, scores, -np.inf)
    scores = scores.softmax()
    return scores @ V

  def params(self):
    return [self.W_q, self.W_k, self.W_v]


class Attention:
  def __init__(self, C, n_heads):
    self.heads = [Head(C, C // n_heads) for _ in range(n_heads)]
    self.W_o = Tensor.uniform(C, C, requires_grad=True)

  def __call__(self, pe):
    pe = pe.layernorm() + pe
    ho = [h(pe) for h in self.heads]
    c_ho = ho[0]  # might be a better way to do this
    for i in range(1, len(ho)):
      c_ho = c_ho.cat(ho[i], dim=-1)
    return c_ho @ self.W_o

  def params(self):
    params = [self.W_o]
    for head in self.heads:
      params.extend(head.params())
    return params


class Block:
  def __init__(self, sz):
    _, _, C, NH, _ = sz
    self.w = Tensor.uniform(C, C, requires_grad=True)  # size: (C,C)
    self.b = Tensor.uniform(C, requires_grad=True)
    self.attn = Attention(C, NH)

  def __call__(self, x):
    out = self.attn(x)
    return (x + out @ self.w + self.b).gelu()

  def params(self):
    return [self.w, self.b] + self.attn.params()


class Bigram:
  def __init__(self, sz):
    _, _, _, _, VS = sz
    self.emb = nn.Embedding(VS, VS)

  def __call__(self, xs, ys):
    logits = self.emb(xs)
    ce = -logits.log_softmax(axis=2).mul(ys).sum(axis=2).mean()  # logits is attn_out
    return logits, ce

  def generate(self, input):
    out = self.emb(input)  # assuming B is 1
    return out[0][-1].argmax()

  def params(self) -> List[Tensor]:
    return [self.emb.weight]


if __name__ == "__main__":
  is_testing = True
  text_size = 1000 if is_testing else 100000
  steps = 200 if is_testing else 2000

  text = load_shakespeare()[:text_size]
  tokenizer = Tokenizer(text=text, iters=100)
  tokens = tokenizer.encode(text)

  # block size, seq len, embedding size, num heads, vocab size
  sz = (64, 16, 32, 4, len(tokenizer.vocab))
  B, T, C, NH, VS = sz

  xs, ys = prep_input(tokens, T)
  xs, ys = sample_batch(xs, ys, sz)

  # NOTE: note used for now
  # attn
  pos_enc = PositionalEncoding(sz)
  pe = pos_enc(xs)
  blocks = [Block(sz) for _ in range(4)]
  for block in blocks:
    pe = block(pe)

  # project it back to vocab size
  W_proj = Tensor.uniform(C, VS, requires_grad=True)
  b_proj = Tensor.uniform(VS, requires_grad=True)
  attn_out = pe @ W_proj + b_proj

  model = Bigram(sz)
  all_params = model.params() + pos_enc.params() + [W_proj, b_proj]
  for block in blocks:
    all_params.extend(block.params())
  # end of not used for now

  optimizer = AdamW(model.params(), lr=1e-3)
  with Tensor.train():
    for step in tqdm(range(steps), desc="training gpt2", unit="step"):
      logits, loss = model(xs, ys)
      loss.backward()
      optimizer.step()

      if step % 100 == 0:
        print(f"step {step}: loss {loss.numpy():.2f}")

  input = tokenizer.encode("With the")
  print(tokenizer.decode(input), end="")
  for i in range(100):
    tok = model.generate(Tensor(input)).item()
    assert isinstance(tok, int)
    print(tokenizer.decode([tok]), end="")
    input.append(tok)
    input = input[-T:]
