import jax
import jax.numpy as jnp
import flax.nnx as nnx


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
