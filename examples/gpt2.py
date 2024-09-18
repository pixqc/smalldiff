# to run: ls examples/gpt2.py | entr -s 'PYTHONPATH="." python3 examples/gpt2.py'


from collections import defaultdict

from helpers import load_shakespeare


def get_max_pair(chars):
  d = defaultdict(int)
  for pair in zip(chars, chars[1:]):
    d[pair] += 1
  return max(d, key=d.get)  # type: ignore


def merge_max_pair(chars, max_pair, token):
  i = 0
  merged = []
  while i < len(chars):
    if i < len(chars) - 1 and (chars[i], chars[i + 1]) == max_pair:
      merged.append(token)
      i += 2
    else:
      merged.append(chars[i])
      i += 1
  return merged


text = load_shakespeare()[:1000]
chars = [c.encode("utf8") for c in text]
chars = [b for char in chars for b in char]
token = 256
m = {}

for i in range(100):
  _max = get_max_pair(chars)
  chars = merge_max_pair(chars, _max, token)
  m[_max] = token
  token += 1

print(chars)
print(m)
