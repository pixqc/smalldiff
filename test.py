import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from tensor import Tensor

x_init = np.random.randn(1, 3).astype(np.float32)
w_init = np.random.randn(3, 3).astype(np.float32)
m_init = np.random.randn(1, 3).astype(np.float32)


def forward_smalldiff(incl_grad=False):
  x = Tensor(x_init)
  w = Tensor(w_init)
  m = Tensor(m_init)
  out = x.matmul(w)
  outr = out.tanh()
  outl = outr.logsoftmax()
  outm = outl.mul(m)
  outx = outm.sum()
  return outx.data


def forward_jax(incl_grad=False):
  @jit
  def forward(x, W, m):
    out = jnp.matmul(x, W)
    outr = jax.nn.tanh(out)
    outl = jax.nn.log_softmax(outr, axis=1)
    outm = outl * m
    return jnp.sum(outm)

  _grad_fn = jit(grad(forward, argnums=(0, 1)))
  x = jnp.array(x_init)
  w = jnp.array(w_init)
  m = jnp.array(m_init)
  outx = forward(x, w, m)
  return outx.item()


np.testing.assert_allclose(forward_smalldiff(), forward_jax())
