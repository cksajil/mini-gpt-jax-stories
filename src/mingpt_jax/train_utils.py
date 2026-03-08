import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx


def cross_entropy_loss(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()


def create_optimizer(
    model, learning_rate, weight_decay, warmup_steps, total_steps, grad_clip_norm
):
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-5,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
    )
    optimizer = nnx.Optimizer(model, tx)
    return optimizer, lr_schedule


@nnx.jit(donate_argnames=("model", "optimizer"))
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        logits = model(x)
        return cross_entropy_loss(logits, y)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


@nnx.jit
def eval_step(model, x, y):
    logits = model(x)
    return cross_entropy_loss(logits, y)


def save_metrics(metrics: dict, out_path: str):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
