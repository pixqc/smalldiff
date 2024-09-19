# to run: ls examples/gpt2.py | entr -s 'PYTHONPATH="." python3 examples/gpt2.py'

from collections import defaultdict

import numpy as np

from helpers import load_shakespeare


class Tokenizer:
  def __init__(self, text, iters=10):
    self.iters = iters
    self.m, self.rm = self.map_tokens(text)

  @staticmethod
  def to_byte_list(text):
    return [b for c in text for b in c.encode("utf8")]

  @staticmethod
  def to_text(byte_list):
    return "".join([chr(b) for b in byte_list])

  def get_max_pair(self, byte_list):
    d = defaultdict(int)
    for pair in zip(byte_list, byte_list[1:]):
      d[pair] += 1
    if not d:
      return None
    if set(d.values()) == {1}:  # nothing to compress
      return None
    return max(d, key=d.get)  # type: ignore

  def merge_max_pair(self, byte_list, max_pair, token):
    i = 0
    merged = []
    while i < len(byte_list):
      if i < len(byte_list) - 1 and (byte_list[i], byte_list[i + 1]) == max_pair:
        merged.append(token)
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

  def tokenize(self, byte_list):
    def _tokenize(rm, bytes_input):
      tokenized = []
      i = 0
      while i < len(bytes_input):
        if i + 1 < len(bytes_input):
          pair = (bytes_input[i], bytes_input[i + 1])
          tok = rm.get(pair)
          if tok is not None:
            tokenized.append(tok)
            i += 2
            continue
        tokenized.append(bytes_input[i])
        i += 1
      return tokenized

    rm = self.rm
    prev = []
    tokenized = _tokenize(rm, byte_list)
    while tokenized != prev:
      prev = tokenized
      tokenized = _tokenize(rm, tokenized)
    return tokenized

  def detokenize(self, tokens):
    def _detokenize(mapping, tokens_input):
      detokenized = []
      for tok in tokens_input:
        if tok in mapping:
          detokenized.extend(mapping[tok])
        else:
          detokenized.append(tok)
      return detokenized

    detokenized = _detokenize(self.m, tokens)
    while any(tok >= 256 for tok in detokenized):
      detokenized = _detokenize(self.m, detokenized)
    return detokenized

  def encode(self, text):
    self.map_tokens(text)
    byte_list = self.to_byte_list(text)
    return self.tokenize(byte_list)

  def decode(self, tokens):
    byte_list = self.detokenize(tokens)
    return self.to_text(byte_list)


if __name__ == "__main__":
  text = load_shakespeare()[:20]
  tokenizer = Tokenizer(text=text, iters=10)
  tokens = tokenizer.encode(text)

  block_size = 8
  trils = []
  ys = []
  for i in range(len(tokens) - block_size + 1):
    tmp = np.tril(tokens[i : i + block_size])
    trils.append(tmp)
    tmp_y = tokens[i + 1 : i + block_size + 1]
    ys.append(tmp_y)

  xs = np.vstack(trils)[:-1]  # last one doesn't have a y
  ys = np.concatenate(ys)
  print(xs)
  print(ys)
  print(xs.shape)
  print(ys.shape)
