# MiniGPT-JAX for Short Story Generation

A MiniGPT-style autoregressive language model built in JAX/Flax NNX and trained on 200k short stories.

## Highlights
- Built a GPT-style decoder-only transformer from scratch in JAX
- Optimized training for Kaggle free-tier hardware
- Reduced training bottlenecks with offline tokenization and memory-mapped batches
- Logged throughput, validation loss, and generated samples
- Designed as a recruiter-facing portfolio project for practical model engineering

## Results
- Dataset: 200,000 short stories
- Context length: 128
- Model: 6-layer MiniGPT
- Hardware: Kaggle free-tier GPU
- Key improvement: pretokenization + streamlined JAX training loop

## Story Generation
```
```

## Sample Outputs
```
```

## Benchmarks
| Version | Batch Size | Tokens/sec | Epoch Time |
|---|---:|---:|---:|
| Baseline kaggle notebook | ... | ... | ... |
| Optimized kaggle notebook | ... | ... | ... |

## Lessons Learned
- Python-side tokenization can dominate wall-clock time
- JIT alone is not enough without an efficient input pipeline
- Throughput tracking is essential for practical training work

## Future Work
- Flash attention or scaled-dot-product attention optimization
- Better evaluation with perplexity and held-out prompts
- Deploy story generator as a Hugging Face Space or Streamlit app