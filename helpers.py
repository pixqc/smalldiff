import gzip
import os

import numpy as np
import requests


def download_mnist(data_dir="./data"):
  base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
  files = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
  ]

  os.makedirs(data_dir, exist_ok=True)
  for filename in files:
    gz_path = os.path.join(data_dir, filename + ".gz")
    extracted_path = os.path.join(data_dir, filename)

    if not os.path.exists(gz_path):
      print(f"downloading {filename}.gz...")
      r = requests.get(base_url + filename + ".gz", stream=True)
      with open(gz_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
          if chunk:
            f.write(chunk)

    if not os.path.exists(extracted_path):
      print(f"extracting {filename}.gz...")
      with gzip.open(gz_path, "rb") as f_in:
        with open(extracted_path, "wb") as f_out:
          f_out.write(f_in.read())

  print("mnist downloaded")


def load_mnist(data_dir="./data"):
  def load_images(file_path):
    with open(file_path, "rb") as f:
      f.read(16)  # skip header
      data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28 * 28)
    return data

  def load_labels(file_path):
    with open(file_path, "rb") as f:
      f.read(8)  # skip header
      labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

  x_train = load_images(os.path.join(data_dir, "train-images-idx3-ubyte"))
  y_train = load_labels(os.path.join(data_dir, "train-labels-idx1-ubyte"))
  x_test = load_images(os.path.join(data_dir, "t10k-images-idx3-ubyte"))
  y_test = load_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))

  return x_train, y_train, x_test, y_test
