import os
import unittest
from functools import partial
from online_lru.seq_model import StackedEncoder
from online_lru.rec import LRUOnline, LRU
import jax
from tests.utils import base_params, inputs, y, mask, compute_grads, check_grad_all
import flax.linen as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


batched_StackedEncoder = nn.vmap(
    StackedEncoder,
    in_axes=0,
    out_axes=0,
    variable_axes={
        "params": None,
        "dropout": None,
        "traces": 0,
        "perturbations": 0,
    },
    methods=["__call__", "update_gradients"],
    split_rngs={"params": False, "dropout": True},
)


class TestStackedEncoder(unittest.TestCase):
    def test_online(self):
        lru = partial(LRUOnline, training_mode="online_dfa", lru_cls=LRU(dim=2), dfa_output_dims=2)
        seq_layer_params = {
            "rec": lru,
            "dropout": 0.1,
            "d_input": 2,
            "d_model": 2,
            "final_output_dim": 2,
            "seq_length": base_params["seq_length"],
            "training_mode": "online_full",
            "training": True,
            "activation": "gelu",
            "n_layers": 2,
            "readout": 2,
        }
        batched_stacked_encoder = batched_StackedEncoder(**seq_layer_params)
        batched_stacked_encoder.rec_type = "LRU"
        params_states = batched_stacked_encoder.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, inputs
        )
        grad, online_grad = compute_grads(batched_stacked_encoder, params_states, inputs, y, mask)

        # lru = partial(LRUOnline, training_mode="bptt", lru_cls=LRU(dim=2), dfa_output_dims=2)
        # seq_layer_params["training_mode"] = "bptt"
        # seq_layer_params["rec"] = lru
        # batched_seqlayer = batched_StackedEncoder(**seq_layer_params)
        # batched_seqlayer.rec_type = "LRU"
        # grad, _ = compute_grads(batched_seqlayer, params_states, inputs, y, mask)
        print(jax.tree_structure(grad))
        print(jax.tree_structure(online_grad))
        dict_check = {
            "layers_0": {
                "seq": {
                    "lru_cls": [
                        "B_re",
                        "B_im",
                        "gamma_log",
                        "nu",
                        "theta",
                    ],
                },
            },
        }
        check_grad_all(grad, online_grad, to_check=dict_check, atol=1e-1)
