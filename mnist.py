import gzip
import os

import numpy as np
import requests

from tensor import Tensor


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
      f.read(16)  # Skip header
      data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28 * 28)
    return data

  def load_labels(file_path):
    with open(file_path, "rb") as f:
      f.read(8)  # Skip header
      labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

  x_train = load_images(os.path.join(data_dir, "train-images-idx3-ubyte"))
  y_train = load_labels(os.path.join(data_dir, "train-labels-idx1-ubyte"))
  x_test = load_images(os.path.join(data_dir, "t10k-images-idx3-ubyte"))
  y_test = load_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))

  return x_train, y_train, x_test, y_test


if __name__ == "__main__":
  x_train, y_train, x_test, y_test = load_mnist()

  learning_rate = 0.01
  epochs = 1000

  x_train = Tensor(x_train)
  y_train = Tensor(y_train)
  x_test = Tensor(x_test)
  y_test = Tensor(y_test)
  w1 = Tensor.rand(784, 16, dtype=np.float32, requires_grad=True)
  b1 = Tensor.rand(16, dtype=np.float32, requires_grad=True)
  w2 = Tensor.rand(16, 16, dtype=np.float32, requires_grad=True)
  b2 = Tensor.rand(16, dtype=np.float32, requires_grad=True)
  w3 = Tensor.rand(16, 10, dtype=np.float32, requires_grad=True)
  b3 = Tensor.rand(10, dtype=np.float32, requires_grad=True)

  for epoch in range(epochs):
    out = ((x_train @ w1 + b1).tanh() @ w2 + b2).tanh() @ w3 + b3
    loss = out.cross_entropy(y_train)
    loss.backward()

    print(f"epoch: {epoch+1}; loss: {loss.numpy()}")

    w1 -= learning_rate * w1.grad
    b1 -= learning_rate * b1.grad
    w2 -= learning_rate * w2.grad
    b2 -= learning_rate * b2.grad
    w3 -= learning_rate * w3.grad
    b3 -= learning_rate * b3.grad

    if epoch % 10 == 0 and epoch != 0:
      out_test = ((x_test @ w1 + b1).tanh() @ w2 + b2).tanh() @ w3 + b3
      predictions = out_test.data.argmax(axis=1)
      accuracy = np.mean(predictions == y_test.numpy())
      print(f"test accuracy: {accuracy}")
