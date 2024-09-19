# to run: ls examples/gpt2.py | entr -s 'PYTHONPATH="." python3 examples/gpt2.py'

from collections import defaultdict

import numpy as np

from helpers import load_shakespeare


def to_byte_list(text: str) -> list[int]:
  return [b for c in text for b in c.encode("utf8")]


def to_text(byte_list: list[int]) -> str:
  return "".join([chr(b) for b in byte_list])


def get_max_pair(byte_list):
  d = defaultdict(int)
  for pair in zip(byte_list, byte_list[1:]):
    d[pair] += 1
  if set(d.values()) == {1}:  # nothing to compress
    return None
  return max(d, key=d.get)  # type: ignore


def merge_max_pair(byte_list, max_pair, token):
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


def map_token(text) -> dict[int, tuple[int, int]]:
  byte_list = to_byte_list(text)
  m = {}

  for i in range(10):
    _max = get_max_pair(byte_list)
    if _max is None:
      break
    byte_list = merge_max_pair(byte_list, _max, 256 + i)
    m[256 + i] = _max

  return m


def tokenize(m: dict[int, tuple[int, int]], byte_list):
  def _tokenize(rm: dict[tuple[int, int], int], byte_list):
    tokenized = []
    i = 0
    while i < len(byte_list):
      if i + 1 < len(byte_list):
        pair = (byte_list[i], byte_list[i + 1])
        tok = rm.get(pair)
        if tok is not None:
          tokenized.append(tok)
          i += 2
        else:
          tokenized.append(byte_list[i])
          i += 1
      else:
        tokenized.append(byte_list[i])
        i += 1
    return tokenized

  rm = {v: k for k, v in m.items()}
  prev = []
  tokenized = _tokenize(rm, byte_list)
  while tokenized != prev:
    prev = tokenized
    tokenized = _tokenize(rm, tokenized)
  return tokenized


def detokenize(m: dict[int, tuple[int, int]], tokens):
  def _detokenize(m: dict[int, tuple[int, int]], tokens):
    detokenized = []
    i = 0
    while i < len(tokens):
      tok = m.get(tokens[i])
      if tok is not None:
        detokenized.append(tok[0])
        detokenized.append(tok[1])
      else:
        detokenized.append(tokens[i])
      i += 1
    return detokenized

  detokenized = _detokenize(m, tokens)
  while any(i >= 256 for i in detokenized):
    detokenized = _detokenize(m, detokenized)
  return detokenized


text = load_shakespeare()[:20]
m = map_token(text)
tokens = tokenize(m, to_byte_list(text))

train_size = int(len(tokens) * 0.9)
train_tokens = np.array(tokens[:train_size])
test_tokens = np.array(tokens[train_size:])

block_size = 8

trils = []
ys = []

block_size = 8
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
