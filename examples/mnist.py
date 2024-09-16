import pickle
from typing import List, Optional

import numpy as np

from helpers import load_mnist
from tensor import SGD, Tensor


class Model:
  def __init__(self, params: Optional[List[Tensor]] = None):
    kwargs = {"dtype": np.float32, "requires_grad": True}

    if params:
      (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3) = params
    else:
      self.w1 = Tensor.randn(784, 128, **kwargs)
      self.b1 = Tensor.randn(128, **kwargs)
      self.w2 = Tensor.randn(128, 64, **kwargs)
      self.b2 = Tensor.randn(64, **kwargs)
      self.w3 = Tensor.randn(64, 10, **kwargs)
      self.b3 = Tensor.randn(10, **kwargs)

  def __call__(self, x):
    out = (x @ self.w1 + self.b1)._batchnorm().relu()
    out = (out @ self.w2 + self.b2)._batchnorm().relu()
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
    (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3) = params
    print(f"Weights loaded from {filepath}")


if __name__ == "__main__":
  normalize = lambda x: x.astype(np.float32) / 255.0
  x_train, y_train, x_test, y_test = load_mnist()
  x_train, x_test = normalize(x_train), normalize(x_test)
  batch_size = 500
  epochs = 10
  num_batches = x_train.shape[0] // batch_size

  model = Model()
  optimizer = SGD(model.params(), lr=0.01, momentum=0.9, weight_decay=0.1)

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

    print(f"epoch: {epoch+1}; loss: {loss.numpy()}")

    if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
      out_test = model(Tensor(x_test))
      predictions = out_test.data.argmax(axis=1)
      accuracy = np.mean(predictions == y_test)
      print(f"test accuracy: {accuracy:.2f}")

  model.save_weights("./data/mnist.pkl")
