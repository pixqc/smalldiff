import gzip
import os

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
