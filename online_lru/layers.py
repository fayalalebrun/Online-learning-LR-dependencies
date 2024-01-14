from flax import linen as nn
import jax
from .bioflax import RandomDenseLinearFA


class SequenceLayer(nn.Module):
    """
    Defines a single layer, with one of the rec module, nonlinearity, dropout, batch/layer norm,
        etc.

    Args:
        rec             (nn.Module):    the recurrent layer to use
        dropout         (float32):      dropout rate
        d_model         (int32):        this is the feature size of the layer inputs and outputs
                                        we usually refer to this size as H
        seq_length      (int32):        length of the time sequence considered
                                        we usually refer to this size as T
        activation      (string):       type of activation function to use
        training_mode   (string):       type of training
        online          (bool):         whether gradient calculation is done online
        prenorm         (bool):         apply prenorm if true or postnorm if false
    """

    rec: nn.Module
    dropout: float
    d_model: int
    seq_length: int
    final_output_dim: int
    activation: str = "gelu"
    training: bool = True
    training_mode: str = "bptt"
    prenorm: bool = False

    def setup(self):
        """Initializes the rec, layer norm and dropout"""
        self.seq = self.rec()
        self.is_dfa = "dfa" in self.training_mode

        if self.activation in ["full_glu"]:
            self.out1 = CustomDense(self.is_dfa, self.d_model, self.final_output_dim)
            self.out2 = CustomDense(self.is_dfa, self.d_model, self.final_output_dim)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = CustomDense(self.is_dfa, self.d_model, self.final_output_dim)

        self.norm = nn.LayerNorm()

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def pre_seq(self, x):
        """
        Processing done to the input before calling the recurrent module.
        """
        if self.prenorm:
            x = self.norm(x)
        return x

    def post_seq(self, x):
        """
        Processing done after the recurrent layer, inside the main branch, before the
        merge with the skip connection.
        """
        if self.activation in ["full_glu", "half_glu1", "half_glu2"]:
            out2 = self.out2
        if self.activation in ["full_glu"]:
            out1 = self.out1

        if self.activation in ["full_glu"]:
            x = self.drop(nn.gelu(x))
            x = out1(x) * jax.nn.sigmoid(out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu1"]:
            x = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(out2(x1))
            x = self.drop(x)
        elif self.activation in ["gelu"]:
            ""
            #x = nn.gelu(x) # FIXME this for DFA
        elif self.activation in ["none"]:
            x = x
        else:
            raise NotImplementedError("Activation: {} not implemented".format(self.activation))
        return x

    def post_skip(self, x):
        """
        Processing done after the skip and main connections have been added together.
        """
        if not self.prenorm:
            x = self.norm(x)
        return x

    def __call__(self, x):
        inputs = x
        hiddens_pre_seq = self.pre_seq(inputs)
        hiddens_post_seq = self.seq(hiddens_pre_seq)
        hiddens_post_skip = self.post_seq(hiddens_post_seq) # + inputs FIXME
        return self.post_skip(hiddens_post_skip)

    def update_gradients(self, grad):
        # Update the gradients of the recurrent module
        # NOTE no need to update other gradients as they will be updated through spatial backprop
        if self.training_mode in ["bptt", "online_spatial", "online_reservoir"]:
            raise ValueError("Upgrade gradient should not be called for this training mode")

        grad["seq"] = self.seq.update_gradients(grad["seq"])
        return grad

class CustomDense(nn.Module):
    is_dfa: bool
    d_model: int
    final_output_dim:  int
    def setup(self):
        self.dfa = RandomDenseLinearFA(features=self.d_model)
        self.dense = nn.Dense(self.d_model)
    def __call__(self, x):
        if self.is_dfa:
            assert(self.final_output_dim>0)
            return self.dfa(x)
        else:
            return self.dense(x)
        

class DFAPassthrough(nn.Module):
    passthrough: bool
    module: nn.module
    @nn.compact
    def __call__(self, *args):

        def f(module, *args):
            return module(*args)
        def fwd(module: nn.module, *args):
            primals, _ = nn.vjp(f, module, *args)
            _, variables = module.unbind()
            variables = jax.tree_map(lambda x: jax.numpy.zeros_like(x), variables)
            return primals, variables
        def bwd(variables, delta):
            return (variables, delta)
        if self.passthrough:
            custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
            return custom_f(self.module, *args)
        else:
            return self.module(*args)
