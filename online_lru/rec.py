from functools import partial
import jax
from jax import random
import jax.numpy as jnp
from flax import linen as nn, core
from .rec_init import matrix_init, theta_init, nu_init, gamma_log_init

@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


class LRU(nn.Module):
    dim: int

    def setup(self):
        self.theta = self.param("theta", partial(theta_init, max_phase=6.28), (self.dim,))
        self.nu = self.param("nu", partial(nu_init, r_min=0.0, r_max=1.0), (self.dim,))
        self.gamma_log = self.param("gamma_log", partial(gamma_log_init), (self.nu, self.theta))

        init = partial(matrix_init, normalization=jnp.sqrt(2 * self.dim))
        self.B_re = self.param("B_re", init, (self.dim, self.dim))
        self.B_im = self.param("B_im", init, (self.dim, self.dim))

    def get_B_norm(self):
        return (self.B_re + 1j * self.B_im) * jnp.expand_dims(jnp.exp(self.gamma_log), axis=-1)

    def __call__(self, inputs):
        # Materializing the diagonal of Lambda and input projections
        diag_lambda = jnp.exp(-jnp.exp(self.nu) + 1j * jnp.exp(self.theta))

        # Running the LRU + output projection
        Lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: self.get_B_norm() @ u)(inputs)
        elements = (Lambda_elements, Bu_elements)
        _, hidden_states = jax.lax.associative_scan(binary_operator_diag, elements)
        return jax.vmap(lambda x, u: x.real + u)(hidden_states, inputs)


class LRUOnline(nn.Module):
    lru_cls: LRU
    training_mode: str = "bptt"  # which learning algorithm will be used

    def setup(self):
        self.lru = self.lru_cls

    def __call__(self, inputs):
        assert self.training_mode in ["bptt", "online_full"]

        if self.training_mode == "bptt":
            return self.lru(inputs)
        else:  # overwrite the backward pass

            def f(module, x, backward_params):
                return module(x)

            def forward(module, x, backward_params):
                primals_out, vjp_fun = nn.vjp(f, module, x, backward_params)
                return primals_out, (vjp_fun, backward_params)

            def backward(vjp_fun_ext, delta):
                vjp_fun, backward_params = vjp_fun_ext
                delta_params, delta_x, _ = vjp_fun(delta)  # compute params gradient with autodiff
                if self.training_mode == "online_full":  # change error backprop to spatial error backprop
                    delta_x = delta + jax.vmap(lambda delta: backward_params["back"] @ delta)(delta)
                return delta_params, delta_x, jax.tree_map(jnp.zeros_like, backward_params)

            custom_f = nn.custom_vjp(fn=f, forward_fn=forward, backward_fn=backward)
            if not self.is_initializing():
                backward_params = {"back": self.lru.get_B_norm().real.T}
            else:
                backward_params = {}
            return custom_f(self.lru, inputs, backward_params)

def init_layer(layer_cls, **kwargs):
    if layer_cls == "LRU":
        layer = LRUOnline
    return partial(layer, **kwargs)
