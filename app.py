import json
from pathlib import Path

import gradio as gr
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import orbax.checkpoint as ocp
import tiktoken


BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "config.json", "r") as f:
    cfg = json.load(f)

MAX_LEN = cfg["max_len"]
VOCAB_NAME = cfg["vocab_name"]
VOCAB_SIZE = cfg["vocab_size"]
EMBED_DIM = cfg["embed_dim"]
NUM_HEADS = cfg["num_heads"]
NUM_LAYERS = cfg["num_layers"]
MLP_DIM = cfg["mlp_dim"]
PAD_ID = cfg["pad_id"]
EOT_ID = cfg["eot_id"]

tokenizer = tiktoken.get_encoding(VOCAB_NAME)


class CausalSelfAttention(nnx.Module):
    def __init__(self, embed_dim, num_heads, *, rngs):
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs,
        )

    def __call__(self, x, mask=None):
        return self.attn(x, mask=mask)


class MLP(nnx.Module):
    def __init__(self, embed_dim, mlp_dim, *, rngs):
        self.fc1 = nnx.Linear(embed_dim, mlp_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_dim, embed_dim, rngs=rngs)

    def __call__(self, x):
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, *, rngs):
        self.ln1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attn = CausalSelfAttention(embed_dim, num_heads, rngs=rngs)
        self.ln2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.mlp = MLP(embed_dim, mlp_dim, rngs=rngs)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, max_len, vocab_size, embed_dim, *, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(max_len, embed_dim, rngs=rngs)

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(token_ids) + self.pos_emb(positions)


class MiniGPT(nnx.Module):
    def __init__(
        self, max_len, vocab_size, embed_dim, num_heads, mlp_dim, num_layers, *, rngs
    ):
        self.max_len = max_len
        self.embedding = TokenAndPositionEmbedding(
            max_len, vocab_size, embed_dim, rngs=rngs
        )
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_dim, rngs=rngs)
            for _ in range(num_layers)
        ]
        self.ln_f = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.lm_head = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)

    def causal_attention_mask(self, seq_len):
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        return mask[None, None, :, :]

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_f(x)
        return self.lm_head(x)


model = MiniGPT(
    max_len=MAX_LEN,
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    num_layers=NUM_LAYERS,
    rngs=nnx.Rngs(42),
)

checkpoint_path = BASE_DIR / "best_model"
ckptr = ocp.PyTreeCheckpointer()

restored_state = ckptr.restore(checkpoint_path, item=nnx.state(model))
nnx.update(model, restored_state)


def generate_text(prompt, temperature=0.9, max_new_tokens=80, top_k=40):
    token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    token_ids = token_ids[:MAX_LEN]

    for _ in range(max_new_tokens):
        context = token_ids[-MAX_LEN:]
        x = [PAD_ID] * MAX_LEN
        x[: len(context)] = context
        x = jnp.asarray([x], dtype=jnp.int32)

        logits = model(x)
        next_logits = logits[0, len(context) - 1] / max(float(temperature), 1e-6)

        if top_k is not None and top_k < next_logits.shape[-1]:
            top_vals, top_idx = jax.lax.top_k(next_logits, int(top_k))
            probs = jax.nn.softmax(top_vals)
            sampled_local = int(
                jax.random.categorical(
                    jax.random.PRNGKey(len(token_ids)), jnp.log(probs)
                )
            )
            next_token = int(top_idx[sampled_local])
        else:
            probs = jax.nn.softmax(next_logits)
            next_token = int(
                jax.random.categorical(
                    jax.random.PRNGKey(len(token_ids)), jnp.log(probs)
                )
            )

        token_ids.append(next_token)

        if next_token == EOT_ID:
            break

    return tokenizer.decode(token_ids)


def app_generate(prompt, temperature, max_new_tokens, top_k):
    if not prompt.strip():
        return "Please enter a prompt."
    return generate_text(prompt, temperature, max_new_tokens, top_k)


demo = gr.Interface(
    fn=app_generate,
    inputs=[
        gr.Textbox(
            label="Prompt",
            lines=4,
            placeholder="Once upon a time in a quiet village...",
        ),
        gr.Slider(0.2, 1.5, value=0.9, step=0.1, label="Temperature"),
        gr.Slider(20, 200, value=80, step=10, label="Max new tokens"),
        gr.Slider(10, 100, value=40, step=5, label="Top-k"),
    ],
    outputs=gr.Textbox(label="Generated story", lines=18),
    title="MiniGPT JAX Story Generator",
    description="Short-story generation demo built in JAX/Flax NNX.",
)

if __name__ == "__main__":
    demo.launch()
