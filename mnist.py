import gzip
import os
import pickle

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
      data = data.astype(np.float32) / 255.0
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


class Model:
  def __init__(self):
    kwargs = {"dtype": np.float32, "requires_grad": True}
    self.w1 = Tensor.rand(784, 128, **kwargs)
    self.b1 = Tensor.rand(128, **kwargs)
    self.w2 = Tensor.rand(128, 64, **kwargs)
    self.b2 = Tensor.rand(64, **kwargs)
    self.w3 = Tensor.rand(64, 10, **kwargs)
    self.b3 = Tensor.rand(10, **kwargs)

  def __call__(self, x):
    out = (x @ self.w1 + self.b1).tanh()
    out = (out @ self.w2 + self.b2).tanh()
    out = out @ self.w3 + self.b3
    return out

  def params(self):
    return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

  def save_weights(self, filepath):
    with open(filepath, "wb") as f:
      pickle.dump(self.params(), f)
    print(f"Weights saved to {filepath}")

  def load_weights(self, filepath):
    with open(filepath, "rb") as f:
      params = pickle.load(f)
    self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = params
    print(f"Weights loaded from {filepath}")


if __name__ == "__main__":
  x_train, y_train, x_test, y_test = load_mnist()
  batch_size = 50
  epochs = 50
  num_batches = x_train.shape[0] // batch_size

  model = Model()
  optimizer = SGD(model.params(), lr=0.01)

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

      out = model(x_batch)
      loss = out.cross_entropy(y_batch)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    print(f"epoch: {epoch+1}; loss: {loss.numpy()}")

    if epoch % 5 == 0 and epoch != 0:
      out_test = model(Tensor(x_test))
      predictions = out_test.data.argmax(axis=1)
      accuracy = np.mean(predictions == y_test)
      print(f"test accuracy: {accuracy:.2f}")

  model.save_weights("./data/mnist.pkl")
