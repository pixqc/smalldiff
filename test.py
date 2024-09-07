import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from tensor import Tensor


def forward_smalldiff(x_data, w_data):
  x = Tensor(x_data)
  w = Tensor(w_data)

  a = x @ w
  b = a + Tensor([1.0])
  c = b.sum()
  out = c.tanh()

  out._ctx.backward(1.0)
  c._ctx.backward(c.grad)
  b._ctx.backward(b.grad)
  a._ctx.backward(a.grad)

  return out.data, x.grad, w.grad


def forward_jax(x_data, w_data):
  @jit
  def forward(x, w):
    out = jnp.matmul(x, w)
    out = out + 1.0
    out = jnp.sum(out)
    out = jax.nn.tanh(out)
    return out

  grad_fn = jit(grad(forward, argnums=(0, 1)))
  x = jnp.array(x_data)
  w = jnp.array(w_data)
  out = forward(x, w)
  x_grad, w_grad = grad_fn(x, w)
  return out.item(), x_grad, w_grad


x_data = np.random.randn(1, 3).astype(np.float32)
w_data = np.random.randn(3, 3).astype(np.float32)
out_smalldiff, x_grad_smalldiff, w_grad_smalldiff = forward_smalldiff(x_data, w_data)
out_jax, x_grad_jax, w_grad_jax = forward_jax(x_data, w_data)

# print(out_smalldiff, out_jax)
# print(x_grad_smalldiff, x_grad_jax)
print(f"w_grad_smalldiff: {w_grad_smalldiff}")
print(f"w_grad_jax: {w_grad_jax}")

# np.testing.assert_allclose(out_smalldiff, out_jax, rtol=1e-5, atol=1e-5)
# np.testing.assert_allclose(x_grad_smalldiff, x_grad_jax, rtol=1e-5, atol=1e-5)
# np.testing.assert_allclose(w_grad_smalldiff, w_grad_jax, rtol=1e-5, atol=1e-5)

# print("Tests passed!")
