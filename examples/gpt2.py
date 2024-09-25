# to run: ls examples/gpt2.py | entr -s 'PYTHONPATH="." python3 examples/gpt2.py'

from collections import Counter

import numpy as np
from tinygrad import nn
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


def prep_input(tokens, T):
  xs = np.array([tokens[i : i + T] for i in range(len(tokens) - T)])
  ys = np.array([tokens[i + 1 : i + T + 1] for i in range(len(tokens) - T)])
  return xs, ys


def sample_batch(xs, ys, sz):
  B, T, _, _, _, VS = sz
  idxs = np.random.choice(xs.shape[0], B, replace=False)
  ys = ys[idxs]
  one_hot = np.zeros((B, T, VS)).astype(np.uint32)
  one_hot[np.arange(B)[:, None], np.arange(T), ys] = 1
  return Tensor(xs[idxs]), Tensor(one_hot)


## -- neural nets --


# TODO: https://arxiv.org/abs/2104.09864
class PositionalEncoding:
  def __init__(self, sz):
    _, T, C, _, _, VS = sz
    p = Tensor.arange(T)
    d = (Tensor.arange(0, C, 2) * -(Tensor(10000).log() / C)).exp()
    pd = p * d
    self.posenc = pd.sin().stack(pd.cos()).repeat(C // 2, 1).T
    self.emb = nn.Embedding(VS, C)

  def __call__(self, xs):
    return self.emb(xs) + self.posenc

  def params(self):
    return [self.emb.weight]


class Attention:
  def __init__(self, sz):
    self.B, self.T, self.C, self.NH, self.HSZ, self.VS = sz
    self.W_q = nn.Linear(C, C, bias=False)
    self.W_k = nn.Linear(C, C, bias=False)
    self.W_v = nn.Linear(C, C, bias=False)
    self.W_o = nn.Linear(C, C, bias=False)

  def __call__(self, pe):
    B, T, C = pe.shape
    # this layernorm and residual connection correct?
    pe_norm = pe.layernorm()
    pe = pe + pe_norm

    Q = self.W_q(pe).reshape(B, T, NH, HSZ).transpose(1, 2)
    K = self.W_k(pe).reshape(B, T, NH, HSZ).transpose(1, 2)
    V = self.W_v(pe).reshape(B, T, NH, HSZ).transpose(1, 2)
    Q = Q.reshape(B * NH, T, HSZ)
    K = K.reshape(B * NH, T, HSZ)

    # "how much each token attend to each other"; high = relevant
    scores = (Q @ K.transpose(2, 1)) * (HSZ**-0.5)
    # only pay attention to the prev tokens, not the future ones
    mask = Tensor.tril(Tensor.ones((T, T)))
    scores = Tensor.where(mask == 1, scores, float("-inf"))
    scores = scores.reshape(B, NH, T, T)
    scores = scores.softmax(axis=-1)
    attn_output = scores @ V
    attn_output = attn_output.transpose(1, 2).reshape(B, T, C)
    return self.W_o(attn_output)

  def params(self):
    return [
      self.W_q.weight,
      self.W_k.weight,
      self.W_v.weight,
      self.W_o.weight,
    ]


class Transformer:
  def __init__(self, sz):
    _, _, C, _, _, VS = sz
    self.linear = nn.Linear(C, VS)
    self.attn = Attention(sz)
    self.posenc = PositionalEncoding(sz)

  def __call__(self, xs):
    pe = self.posenc(xs)
    return self.linear(self.attn(pe)).tanh()

  def loss(self, xs, ys):
    out = self(xs)
    return -out.log_softmax(axis=2).mul(ys).sum(axis=2).mean()

  def generate(self, xs, temp=1.0):
    out = self(xs)
    return (out[0][-1].softmax() / temp).multinomial()

  def params(self):
    return [
      self.posenc.emb.weight,
      self.linear.weight,
      self.linear.bias,
    ] + self.attn.params()


if __name__ == "__main__":
  text = load_shakespeare()[:1000]
  tokenizer = Tokenizer(text=text, iters=500)
  tokens = tokenizer.encode(text)

  B, T, C, NH, HSZ, VS = 64, 16, 32, 4, 32 // 4, len(tokenizer.vocab)
  sz = (B, T, C, NH, HSZ, VS)

  xs, ys = prep_input(tokens, T)
  xs, ys = sample_batch(xs, ys, sz)

  model = Transformer(sz)
  optimizer = AdamW(model.params(), lr=1e-4)
  with Tensor.train():
    for step in tqdm(range(5000), desc="training gpt2", unit="step"):
      loss = model.loss(xs, ys)
      loss.backward()
      optimizer.step()

      if step % 100 == 0:
        print(f"step {step}: loss {loss.numpy():.2f}")

  pad_right = lambda xs: xs + [0] * (T - len(xs))
  input = tokenizer.encode("With the")
  print(tokenizer.decode(input), end="")
  for i in range(100):
    input = pad_right(input)
    tok = model.generate(Tensor(pad_right(input))).item()
    print(tokenizer.decode([tok]), end="")
    input = input[: min(len(input), T)]
    input.append(tok)
    input = input[-T:]
