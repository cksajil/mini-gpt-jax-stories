import json
from pathlib import Path

import gradio as gr
import jax.numpy as jnp
import flax.nnx as nnx
import orbax.checkpoint as ocp
import tiktoken


# -------------------------
# Load config
# -------------------------
BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "config.json", "r") as f:
    cfg = json.load(f)

maxlen = cfg["maxlen"]
vocab_size = cfg["vocab_size"]
embed_dim = cfg["embed_dim"]
num_heads = cfg["num_heads"]
feed_forward_dim = cfg["feed_forward_dim"]
num_transformer_blocks = cfg["num_transformer_blocks"]

tokenizer = tiktoken.get_encoding(cfg.get("tokenizer_name", "gpt2"))
eot_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]


# -------------------------
# Model definition
# -------------------------
class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, *, rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            rngs=rngs,
        )
        self.ffn = nnx.Sequential(
            nnx.Linear(embed_dim, ff_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(ff_dim, embed_dim, rngs=rngs),
        )

    def __call__(self, x, mask=None):
        x = x + self.attention(x, mask=mask)
        x = x + self.ffn(x)
        return x


class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, token_ids):
        positions = jnp.arange(token_ids.shape[1])[None, :]
        return self.token_emb(token_ids) + self.pos_emb(positions)


class MiniGPT(nnx.Module):
    def __init__(
        self,
        maxlen=maxlen,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_transformer_blocks=num_transformer_blocks,
        *,
        rngs=nnx.Rngs(0),
    ):
        self.maxlen = maxlen
        self.embedding = TokenAndPositionEmbedding(
            maxlen, vocab_size, embed_dim, rngs=rngs
        )
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
            for _ in range(num_transformer_blocks)
        ]
        self.output_layer = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)

    def causal_attention_mask(self, seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)

        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        logits = self.output_layer(x)
        return logits


# -------------------------
# Load checkpoint
# -------------------------
model = MiniGPT()

checkpoint_path = BASE_DIR / "small_checkpoint.orbax"
checkpointer = ocp.PyTreeCheckpointer()

restored_state = checkpointer.restore(
    checkpoint_path,
    item=nnx.state(model),
)
nnx.update(model, restored_state)


# -------------------------
# Generation
# -------------------------
def generate_text(model, start_tokens, max_new_tokens=50, temperature=1.0):
    tokens = list(start_tokens)

    for _ in range(max_new_tokens):
        context = tokens[-model.maxlen :]
        actual_len = len(context)

        if actual_len < model.maxlen:
            context = context + [0] * (model.maxlen - actual_len)

        context_array = jnp.array(context)[None, :]
        logits = model(context_array)

        next_token_logits = logits[0, actual_len - 1, :] / max(temperature, 1e-6)
        next_token = int(jnp.argmax(next_token_logits))

        if next_token == eot_id:
            break

        tokens.append(next_token)

    return tokenizer.decode(tokens)


def create_story(story_prompt, temperature, max_new_tokens):
    if not story_prompt or not story_prompt.strip():
        return "Please enter a prompt."

    start_tokens = tokenizer.encode(story_prompt)[:maxlen]
    return generate_text(
        model,
        start_tokens,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
    )


# -------------------------
# Gradio app
# -------------------------
demo = gr.Interface(
    fn=create_story,
    inputs=[
        gr.Textbox(
            label="Story Prompt",
            lines=4,
            placeholder="A little fox found a glowing lantern in the forest...",
        ),
        gr.Slider(minimum=0.1, maximum=1.5, value=0.8, step=0.05, label="Temperature"),
        gr.Slider(minimum=10, maximum=200, value=50, step=5, label="Max Tokens"),
    ],
    outputs=gr.Textbox(label="Generated Story", lines=14),
    title="MiniGPT JAX Story Generator",
    description="Generate short story continuations with a MiniGPT model trained in JAX.",
)

if __name__ == "__main__":
    demo.launch()
