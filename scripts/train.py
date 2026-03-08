import os
import time
from pathlib import Path
import numpy as np
import flax.nnx as nnx
import orbax.checkpoint as ocp

from minigpt_jax.config import load_config
from minigpt_jax.data import (
    load_story_column,
    get_tokenizer,
    build_or_load_token_cache,
    batch_iterator,
)
from minigpt_jax.model import MiniGPT
from minigpt_jax.train_utils import (
    create_optimizer,
    train_step,
    eval_step,
    save_metrics,
)


os.environ["JAX_COMPILATION_CACHE_DIR"] = "/kaggle/working/jax_cache"


def run_validation(model, valid_tokens, batch_size, seed=42, max_batches=50):
    losses = []
    for i, (x_np, y_np) in enumerate(
        batch_iterator(valid_tokens, batch_size, shuffle=False, seed=seed)
    ):
        loss = eval_step(model, x_np, y_np)
        losses.append(float(loss))
        if i + 1 >= max_batches:
            break
    return float(np.mean(losses))


def main(config_path="configs/base.yaml"):
    cfg = load_config(config_path)

    train_stories = load_story_column(cfg.data.train_csv, cfg.data.max_train_stories)
    valid_stories = load_story_column(cfg.data.valid_csv, cfg.data.max_valid_stories)

    tokenizer, eot_id = get_tokenizer(cfg.model.vocab_name)
    vocab_size = tokenizer.n_vocab

    train_tokens = build_or_load_token_cache(
        train_stories,
        "train",
        cfg.data.max_len,
        tokenizer,
        eot_id,
        cfg.data.token_cache_dir,
    )
    valid_tokens = build_or_load_token_cache(
        valid_stories,
        "valid",
        cfg.data.max_len,
        tokenizer,
        eot_id,
        cfg.data.token_cache_dir,
    )

    rngs = nnx.Rngs(cfg.train.seed)

    model = MiniGPT(
        max_len=cfg.data.max_len,
        vocab_size=vocab_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        mlp_dim=cfg.model.mlp_dim,
        num_layers=cfg.model.num_layers,
        rngs=rngs,
    )

    steps_per_epoch = len(train_tokens) // cfg.train.batch_size
    total_steps = steps_per_epoch * cfg.train.num_epochs
    warmup_steps = max(1, int(cfg.train.warmup_frac * total_steps))

    optimizer, lr_schedule = create_optimizer(
        model=model,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        grad_clip_norm=cfg.train.grad_clip_norm,
    )

    history = {
        "train_loss": [],
        "valid_loss": [],
        "tokens_per_sec": [],
        "step_time_sec": [],
    }

    best_valid_loss = float("inf")
    global_step = 0

    for epoch in range(cfg.train.num_epochs):
        running_loss = 0.0
        running_steps = 0

        for step, (x_np, y_np) in enumerate(
            batch_iterator(
                train_tokens,
                cfg.train.batch_size,
                shuffle=True,
                seed=cfg.train.seed + epoch,
            ),
            start=1,
        ):
            start = time.time()
            loss = train_step(model, optimizer, x_np, y_np)
            loss_value = float(loss)
            step_time = time.time() - start

            tok_per_sec = x_np.size / max(step_time, 1e-8)
            history["tokens_per_sec"].append(tok_per_sec)
            history["step_time_sec"].append(step_time)

            running_loss += loss_value
            running_steps += 1
            global_step += 1

            if step % 100 == 0:
                avg_loss = running_loss / running_steps
                current_lr = float(lr_schedule(global_step - 1))
                print(
                    f"epoch={epoch+1} step={step}/{steps_per_epoch} "
                    f"loss={avg_loss:.4f} lr={current_lr:.2e} tok/s={tok_per_sec:,.0f}"
                )

        train_epoch_loss = running_loss / max(running_steps, 1)
        valid_loss = run_validation(
            model, valid_tokens, cfg.train.batch_size, seed=cfg.train.seed
        )

        history["train_loss"].append(train_epoch_loss)
        history["valid_loss"].append(valid_loss)

        print(
            f"Epoch {epoch+1} complete | train_loss={train_epoch_loss:.4f} | "
            f"valid_loss={valid_loss:.4f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            ckptr = ocp.PyTreeCheckpointer()
            save_path = Path(cfg.output.checkpoint_dir) / "best_model"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ckptr.save(save_path, nnx.state(model), force=True)

    save_metrics(history, str(Path(cfg.output.metrics_dir) / "metrics.json"))


if __name__ == "__main__":
    main()
