from minigpt_jax.config import load_config
from minigpt_jax.data import get_tokenizer, PAD_ID
from minigpt_jax.model import MiniGPT
from minigpt_jax.inference import generate_text
import flax.nnx as nnx
import orbax.checkpoint as ocp


def main(config_path="configs/base.yaml", prompt="Once upon a time"):
    cfg = load_config(config_path)

    tokenizer, eot_id = get_tokenizer(cfg.model.vocab_name)
    vocab_size = tokenizer.n_vocab

    model = MiniGPT(
        max_len=cfg.data.max_len,
        vocab_size=vocab_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        mlp_dim=cfg.model.mlp_dim,
        num_layers=cfg.model.num_layers,
        rngs=nnx.Rngs(cfg.train.seed),
    )

    ckptr = ocp.PyTreeCheckpointer()
    restored_state = ckptr.restore(
        f"{cfg.output.checkpoint_dir}/best_model", item=nnx.state(model)
    )
    nnx.update(model, restored_state)

    out = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_len=cfg.data.max_len,
        pad_id=PAD_ID,
        eot_id=eot_id,
        max_new_tokens=100,
        temperature=0.9,
        top_k=40,
    )
    print(out)


if __name__ == "__main__":
    main()
