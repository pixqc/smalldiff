import gzip
import math
import os
import shutil
import sys
import time
from typing import Optional

import numpy as np
import requests


def load_mnist(data_dir="./data"):
  os.makedirs(data_dir, exist_ok=True)
  base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
  files = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
  ]

  def _load(path, offset, dtype, shape):
    with open(path, "rb") as f:
      f.read(offset)
      return np.frombuffer(f.read(), dtype=dtype).reshape(shape)

  for f in files:
    gz, ex = os.path.join(data_dir, f + ".gz"), os.path.join(data_dir, f)
    if not os.path.exists(ex):
      if not os.path.exists(gz):
        print(f"downloading {f}.gz...")
        with open(gz, "wb") as f_out:
          f_out.write(requests.get(base_url + f + ".gz").content)
      print(f"extracting {f}.gz...")
      with gzip.open(gz, "rb") as f_in, open(ex, "wb") as f_out:
        f_out.write(f_in.read())

  x_train = _load(os.path.join(data_dir, files[0]), 16, np.uint8, (-1, 28 * 28))
  y_train = _load(os.path.join(data_dir, files[1]), 8, np.uint8, -1)
  x_test = _load(os.path.join(data_dir, files[2]), 16, np.uint8, (-1, 28 * 28))
  y_test = _load(os.path.join(data_dir, files[3]), 8, np.uint8, -1)
  return x_train, y_train, x_test, y_test


def load_shakespeare(data_dir="./data"):
  os.makedirs(data_dir, exist_ok=True)
  base_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
  file_path = os.path.join(data_dir, "tinyshakespeare.txt")

  if not os.path.exists(file_path):
    print("downloading tiny shakespeare...")
    with open(file_path, "wb") as f_out:
      f_out.write(requests.get(base_url).content)

  # Read the text file
  with open(file_path, "r", encoding="utf-8") as f:
    data = f.read()

  return data


# lifted from tinygrad/helpers.py
class tqdm:
  def __init__(
    self,
    iterable=None,
    desc: str = "",
    disable: bool = False,
    unit: str = "it",
    unit_scale=False,
    total: Optional[int] = None,
    rate: int = 100,
  ):
    self.iterable, self.disable, self.unit, self.unit_scale, self.rate = (
      iterable,
      disable,
      unit,
      unit_scale,
      rate,
    )
    self.st, self.i, self.n, self.skip, self.t = (
      time.perf_counter(),
      -1,
      0,
      1,
      getattr(iterable, "__len__", lambda: 0)() if total is None else total,
    )
    self.set_description(desc)
    self.update(0)

  def __iter__(self):
    for item in self.iterable:  # type: ignore
      yield item
      self.update(1)
    self.update(close=True)

  def set_description(self, desc: str):
    self.desc = f"{desc}: " if desc else ""

  def update(self, n: int = 0, close: bool = False):
    self.n, self.i = self.n + n, self.i + 1
    if self.disable or (not close and self.i % self.skip != 0):
      return
    prog, elapsed, ncols = (
      self.n / self.t if self.t else 0,
      time.perf_counter() - self.st,
      shutil.get_terminal_size().columns,
    )
    if self.i / elapsed > self.rate and self.i:
      self.skip = max(int(self.i / elapsed) // self.rate, 1)

    def HMS(t):
      return ":".join(
        f"{x:02d}" if i else str(x)
        for i, x in enumerate([int(t) // 3600, int(t) % 3600 // 60, int(t) % 60])
        if i or x
      )

    def SI(x):
      return (
        (
          f"{x/1000**int(g:=math.log(x,1000)):.{int(3-3*math.fmod(g,1))}f}"[:4].rstrip(
            "."
          )
          + " kMGTPEZY"[int(g)].strip()
        )
        if x
        else "0.00"
      )

    prog_text = (
      f'{SI(self.n)}{f"/{SI(self.t)}" if self.t else self.unit}'
      if self.unit_scale
      else f'{self.n}{f"/{self.t}" if self.t else self.unit}'
    )
    elapsed_text = HMS(elapsed) + (
      f'<{HMS(elapsed/prog-elapsed) if self.n else "?"}' if self.t else ""
    )
    it_text = (
      (SI(self.n / elapsed) if self.unit_scale else f"{self.n/elapsed:5.2f}")
      if self.n
      else "?"
    )
    suf = f"{prog_text} [{elapsed_text}, {it_text}{self.unit}/s]"
    sz = max(ncols - len(self.desc) - 3 - 2 - 2 - len(suf), 1)
    bar = (
      "\r"
      + self.desc
      + (
        f'{100*prog:3.0f}%|{("█"*int(num:=sz*prog)+" ▏▎▍▌▋▊▉"[int(8*num)%8].strip()).ljust(sz," ")}| '
        if self.t
        else ""
      )
      + suf
    )
    print(bar[: ncols + 1], flush=True, end="\n" * close, file=sys.stderr)
