import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from tensor import Tensor


x_data = np.random.randn(1, 3).astype(np.float32)
w_data = np.random.randn(3, 6).astype(np.float32)
b_data = np.random.randn(6).astype(np.float32)


def smalldiff_00():
  x = Tensor(x_data)
  w = Tensor(w_data)
  b = Tensor(b_data)

  out1 = x @ w
  out2 = out1 + b
  out3 = out2.sum()
  out = out3.tanh()
  out.backward()
  return out.data, x.grad, w.grad


def jax_00():
  @jit
  def forward(x, w, b):
    out = x @ w
    out = out + b
    out = out.sum()
    out = jnp.tanh(out)
    return out

  grad_fn = jit(grad(forward, argnums=(0, 1)))
  x = jnp.array(x_data)
  w = jnp.array(w_data)
  b = jnp.array(b_data)
  out = forward(x, w, b)
  x_grad, w_grad = grad_fn(x, w, b)
  return out.item(), x_grad, w_grad


out_smalldiff, x_grad_smalldiff, w_grad_smalldiff = smalldiff_00()
out_jax, x_grad_jax, w_grad_jax = jax_00()


def compare_outputs(smalldiff_output, jax_output, name, rtol=1e-5, atol=1e-5):
  try:
    np.testing.assert_allclose(smalldiff_output, jax_output, rtol=rtol, atol=atol)
    print(f"{name} comparison passed!")
  except AssertionError as e:
    print(f"{name} comparison failed:")
    print(e)
    print(f"Smalldiff output:\n{smalldiff_output}")
    print(f"JAX output:\n{jax_output}")


compare_outputs(out_smalldiff, out_jax, "Output")
compare_outputs(x_grad_smalldiff, x_grad_jax, "X Gradient")
compare_outputs(w_grad_smalldiff, w_grad_jax, "W Gradient")
