import gzip
import os

import numpy as np
import requests

from tensor import SGD, Tensor


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


if __name__ == "__main__":
  x_train, y_train, x_test, y_test = load_mnist()
  batch_size = 50
  epochs = 50
  num_batches = x_train.shape[0] // batch_size

  w1 = Tensor.rand(784, 128, dtype=np.float32, requires_grad=True)
  b1 = Tensor.rand(128, dtype=np.float32, requires_grad=True)
  w2 = Tensor.rand(128, 10, dtype=np.float32, requires_grad=True)
  b2 = Tensor.rand(10, dtype=np.float32, requires_grad=True)
  optimizer = SGD([w1, b1, w2, b2], lr=0.01)

  print("mnist start train...")
  for epoch in range(epochs):
    shuffle_indices = np.random.permutation(x_train.shape[0])
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    for batch in range(num_batches):
      start = batch * batch_size
      end = start + batch_size
      x_batch = Tensor(x_train[start:end])
      y_batch = Tensor(y_train[start:end])

      out = (x_batch @ w1 + b1).tanh() @ w2 + b2
      loss = out.cross_entropy(y_batch)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    print(f"epoch: {epoch+1}; loss: {loss.numpy()}")

    if epoch % 5 == 0 and epoch != 0:
      out_test = (Tensor(x_test) @ w1 + b1).tanh() @ w2 + b2
      predictions = out_test.data.argmax(axis=1)
      accuracy = np.mean(predictions == y_test)
      print(f"test accuracy: {accuracy}")
