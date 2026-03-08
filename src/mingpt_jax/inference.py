import numpy as np
import jax
import jax.numpy as jnp


def generate_text(
    model,
    tokenizer,
    prompt,
    max_len,
    pad_id,
    eot_id,
    max_new_tokens=80,
    temperature=1.0,
    top_k=40,
):
    token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    token_ids = token_ids[:max_len]

    for _ in range(max_new_tokens):
        context = token_ids[-max_len:]
        x = np.full((1, max_len), pad_id, dtype=np.int32)
        x[0, : len(context)] = np.asarray(context, dtype=np.int32)

        logits = model(jnp.asarray(x))
        next_logits = logits[0, len(context) - 1] / max(temperature, 1e-6)

        if top_k is not None and top_k < next_logits.shape[-1]:
            top_vals, top_idx = jax.lax.top_k(next_logits, top_k)
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

        if next_token == eot_id:
            break

    return tokenizer.decode(token_ids)
