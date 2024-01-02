from jax import random
import jax.numpy as jnp


def matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return random.normal(key=key, shape=shape, dtype=dtype) / normalization
    
def nu_init(key, shape, r_min, r_max, dtype=jnp.float32):
    u = random.uniform(key=key, shape=shape, dtype=dtype)
    nu = -0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2)
    return jnp.log(nu)


def theta_init(key, shape, max_phase, dtype=jnp.float32, log=True):
    u = random.uniform(key, shape=shape, dtype=dtype)
    theta = max_phase * u
    return jnp.log(theta)


def gamma_log_init(key, lamb):
    nu, theta = lamb
    nu, theta = jnp.exp(nu), jnp.exp(theta)
    diag_lambda = jnp.exp(-nu + 1j * theta)
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))
