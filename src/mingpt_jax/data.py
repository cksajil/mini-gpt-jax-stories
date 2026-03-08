from pathlib import Path
import numpy as np
import pandas as pd
import tiktoken


PAD_ID = 0


def load_story_column(csv_path: str, limit: int) -> list[str]:
    df = pd.read_csv(csv_path).dropna()
    return df["text"].astype(str).tolist()[:limit]


def get_tokenizer(vocab_name: str = "gpt2"):
    tokenizer = tiktoken.get_encoding(vocab_name)
    eot_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    return tokenizer, eot_id


def encode_story_fixed(text: str, max_len: int, tokenizer, eot_id: int) -> np.ndarray:
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    ids = ids[: max_len - 1]
    ids.append(eot_id)

    arr = np.full((max_len,), PAD_ID, dtype=np.uint16)
    arr[: len(ids)] = np.asarray(ids, dtype=np.uint16)
    return arr


def build_or_load_token_cache(
    stories: list[str],
    split_name: str,
    max_len: int,
    tokenizer,
    eot_id: int,
    token_cache_dir: str,
):
    cache_dir = Path(token_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    token_file = cache_dir / f"{split_name}_tokens_{len(stories)}_{max_len}.npy"

    if token_file.exists():
        return np.load(token_file, mmap_mode="r")

    arr = np.empty((len(stories), max_len), dtype=np.uint16)
    for i, story in enumerate(stories):
        arr[i] = encode_story_fixed(story, max_len, tokenizer, eot_id)

    np.save(token_file, arr)
    return np.load(token_file, mmap_mode="r")


def batch_iterator(token_array, batch_size: int, shuffle: bool = True, seed: int = 42):
    n = len(token_array)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(idx)

    for start in range(0, n - batch_size + 1, batch_size):
        batch_idx = idx[start : start + batch_size]
        x = np.asarray(token_array[batch_idx], dtype=np.int32)

        y = np.full_like(x, PAD_ID, dtype=np.int32)
        y[:, :-1] = x[:, 1:]
        y[:, -1] = PAD_ID

        yield x, y
