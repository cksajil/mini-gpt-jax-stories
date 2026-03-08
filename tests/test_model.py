import jax.numpy as jnp
import flax.nnx as nnx
from minigpt_jax.model import MiniGPT


def test_model_output_shape():
    model = MiniGPT(
        max_len=16,
        vocab_size=100,
        embed_dim=32,
        num_heads=4,
        mlp_dim=64,
        num_layers=2,
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 16), dtype=jnp.int32)
    logits = model(x)
    assert logits.shape == (2, 16, 100)
