import json
from pathlib import Path

print("DEBUG 1: Starting app.py")

import gradio as gr
print("DEBUG 2: Imported gradio")

import jax
print("DEBUG 3: Imported jax")

import jax.numpy as jnp
print("DEBUG 4: Imported jax.numpy")

import flax.nnx as nnx
print("DEBUG 5: Imported flax.nnx")

import orbax.checkpoint as ocp
print("DEBUG 6: Imported orbax.checkpoint")

import tiktoken
print("DEBUG 7: Imported tiktoken")


# --------------------------------------------------
# Paths and config
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
print("DEBUG 8: BASE_DIR =", BASE_DIR)

CONFIG_PATH = BASE_DIR / "config.json"
print("DEBUG 9: CONFIG_PATH =", CONFIG_PATH, "| exists =", CONFIG_PATH.exists())

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

print("DEBUG 10: Config loaded successfully")
print("DEBUG 11: Config contents =", cfg)

MAX_LEN = cfg["max_len"]
VOCAB_NAME = cfg["vocab_name"]
VOCAB_SIZE = cfg["vocab_size"]
EMBED_DIM = cfg["embed_dim"]
NUM_HEADS = cfg["num_heads"]
NUM_LAYERS = cfg["num_layers"]
MLP_DIM = cfg["mlp_dim"]
PAD_ID = cfg["pad_id"]
EOT_ID = cfg["eot_id"]

print("DEBUG 12: Parsed config values")
print("  MAX_LEN =", MAX_LEN)
print("  VOCAB_NAME =", VOCAB_NAME)
print("  VOCAB_SIZE =", VOCAB_SIZE)
print("  EMBED_DIM =", EMBED_DIM)
print("  NUM_HEADS =", NUM_HEADS)
print("  NUM_LAYERS =", NUM_LAYERS)
print("  MLP_DIM =", MLP_DIM)
print("  PAD_ID =", PAD_ID)
print("  EOT_ID =", EOT_ID)


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------
print("DEBUG 13: Loading tokenizer...")
tokenizer = tiktoken.get_encoding(VOCAB_NAME)
print("DEBUG 14: Tokenizer loaded successfully")


# --------------------------------------------------
# Model definition
# --------------------------------------------------
class CausalSelfAttention(nnx.Module):
    def __init__(self, embed_dim, num_heads, *, rngs):
        print("DEBUG MODEL: Initializing CausalSelfAttention")
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
        print("DEBUG MODEL: Initializing MLP")
        self.fc1 = nnx.Linear(embed_dim, mlp_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_dim, embed_dim, rngs=rngs)

    def __call__(self, x):
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, *, rngs):
        print("DEBUG MODEL: Initializing TransformerBlock")
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
        print("DEBUG MODEL: Initializing TokenAndPositionEmbedding")
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(max_len, embed_dim, rngs=rngs)

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(token_ids) + self.pos_emb(positions)


class MiniGPT(nnx.Module):
    def __init__(
        self,
        max_len,
        vocab_size,
        embed_dim,
        num_heads,
        mlp_dim,
        num_layers,
        *,
        rngs
    ):
        print("DEBUG MODEL: Initializing MiniGPT")
        self.max_len = max_len
        self.embedding = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim, rngs=rngs)
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


# --------------------------------------------------
# Create model
# --------------------------------------------------
print("DEBUG 15: Creating model...")
model = MiniGPT(
    max_len=MAX_LEN,
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    num_layers=NUM_LAYERS,
    rngs=nnx.Rngs(42),
)
print("DEBUG 16: Model created successfully")


# --------------------------------------------------
# Check checkpoint files
# --------------------------------------------------
checkpoint_path = BASE_DIR / "best_model"
print("DEBUG 17: checkpoint_path =", checkpoint_path)
print("DEBUG 18: checkpoint exists =", checkpoint_path.exists())

if checkpoint_path.exists():
    print("DEBUG 19: Listing checkpoint contents...")
    all_ckpt_files = list(checkpoint_path.rglob("*"))
    print("DEBUG 20: Number of checkpoint items =", len(all_ckpt_files))
    for i, p in enumerate(all_ckpt_files[:50]):
        print(f"  CKPT ITEM {i+1}: {p}")
    if len(all_ckpt_files) > 50:
        print("  ... more files omitted ...")
else:
    print("DEBUG ERROR: best_model folder not found")


# --------------------------------------------------
# Restore checkpoint
# --------------------------------------------------
print("DEBUG 21: Creating Orbax checkpointer...")
ckptr = ocp.PyTreeCheckpointer()
print("DEBUG 22: Orbax checkpointer created")

print("DEBUG 23: Preparing abstract state and restore_args...")

state = nnx.state(model)

# Restore everything onto the current device.
single_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

restore_args = jax.tree_util.tree_map(
    lambda x: ocp.ArrayRestoreArgs(
        restore_type=jax.Array,
        sharding=single_sharding,
        global_shape=x.shape,
        dtype=x.dtype,
    ) if hasattr(x, "shape") and hasattr(x, "dtype") else ocp.RestoreArgs(),
    state,
)

print("DEBUG 23A: About to restore checkpoint...")
try:
    restored_state = ckptr.restore(
        checkpoint_path,
        item=state,
        restore_args=restore_args,
    )
    print("DEBUG 24: Checkpoint restore finished")
except Exception as e:
    print("DEBUG ERROR during restore:", repr(e))
    raise

print("DEBUG 25: About to update model state...")
try:
    nnx.update(model, restored_state)
    print("DEBUG 26: Model state updated successfully")
except Exception as e:
    print("DEBUG ERROR during nnx.update:", repr(e))
    raise


# --------------------------------------------------
# Text generation
# --------------------------------------------------
def generate_text(prompt, temperature=0.9, max_new_tokens=80, top_k=40):
    print("DEBUG GEN 1: generate_text called")
    print("DEBUG GEN 2: prompt =", prompt)
    print("DEBUG GEN 3: temperature =", temperature, "| max_new_tokens =", max_new_tokens, "| top_k =", top_k)

    token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    token_ids = token_ids[:MAX_LEN]

    print("DEBUG GEN 4: initial token count =", len(token_ids))

    if len(token_ids) == 0:
        return "Please enter a non-empty prompt."

    for step in range(max_new_tokens):
        context = token_ids[-MAX_LEN:]
        x = [PAD_ID] * MAX_LEN
        x[:len(context)] = context
        x = jnp.asarray([x], dtype=jnp.int32)

        if step == 0:
            print("DEBUG GEN 5: input tensor shape =", x.shape)

        logits = model(x)
        next_logits = logits[0, len(context) - 1] / max(float(temperature), 1e-6)

        if top_k is not None and int(top_k) < next_logits.shape[-1]:
            top_vals, top_idx = jax.lax.top_k(next_logits, int(top_k))
            probs = jax.nn.softmax(top_vals)
            sampled_local = int(
                jax.random.categorical(
                    jax.random.PRNGKey(len(token_ids)),
                    jnp.log(probs)
                )
            )
            next_token = int(top_idx[sampled_local])
        else:
            probs = jax.nn.softmax(next_logits)
            next_token = int(
                jax.random.categorical(
                    jax.random.PRNGKey(len(token_ids)),
                    jnp.log(probs)
                )
            )

        token_ids.append(next_token)

        if step < 3:
            print(f"DEBUG GEN step {step+1}: next_token =", next_token)

        if next_token == EOT_ID:
            print("DEBUG GEN: encountered EOT, stopping generation")
            break

    decoded = tokenizer.decode(token_ids)
    print("DEBUG GEN 6: generation complete, output length =", len(decoded))
    return decoded


def app_generate(prompt, temperature, max_new_tokens, top_k):
    print("DEBUG APP 1: app_generate called")

    if not prompt or not prompt.strip():
        print("DEBUG APP 2: empty prompt")
        return "Please enter a prompt."

    try:
        result = generate_text(
            prompt=prompt,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            top_k=int(top_k),
        )
        print("DEBUG APP 3: generation succeeded")
        return result
    except Exception as e:
        print("DEBUG APP ERROR:", repr(e))
        return f"Error during generation: {repr(e)}"


# --------------------------------------------------
# Gradio app
# --------------------------------------------------
print("DEBUG 27: Building Gradio interface...")

demo = gr.Interface(
    fn=app_generate,
    inputs=[
        gr.Textbox(
            label="Prompt",
            lines=4,
            placeholder="Once upon a time in a quiet village..."
        ),
        gr.Slider(0.2, 1.5, value=0.9, step=0.1, label="Temperature"),
        gr.Slider(20, 200, value=80, step=10, label="Max new tokens"),
        gr.Slider(10, 100, value=40, step=5, label="Top-k"),
    ],
    outputs=gr.Textbox(label="Generated story", lines=18),
    title="MiniGPT JAX Story Generator",
    description="Short-story generation demo built in JAX/Flax NNX.",
)

print("DEBUG 28: Gradio interface built successfully")
print("DEBUG 29: About to launch app...")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)