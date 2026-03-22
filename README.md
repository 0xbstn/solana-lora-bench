# solana-lora-bench

Fine-tuned open-weight models evaluated on [Solana Bench](https://github.com/solana-foundation/solana-gym-env).

## Context

Solana Bench measures how well LLMs compose and execute real Solana transactions in a sandboxed environment (Surfpool). Current evaluations cover frontier models only:

| Model | Median Score | Max | Min |
|-------|-------------|-----|-----|
| Claude Sonnet 4 | 115 | 181 | 30 |
| GPT-5 | 60 | 66 | 57 |
| Gemini 2.5 Flash | 40 | 44 | 23 |
| gpt-oss-120b | 23 | 25 | 16 |

Open-weight and fine-tuned models are missing from the picture.

## Approach

1. Extract code-generation training pairs from 130 existing Solana Bench trajectories
2. Augment with Solana SDK docs and protocol examples
3. LoRA fine-tune on Qwen 3.5-35B-A3B (MoE, 3B active params)
4. Evaluate on Solana Bench under the same conditions as frontier models
5. Publish dataset, adapter, and results

## Repository layout

```
data/
  raw/            Trajectories and source material
  processed/      Training-ready JSONL
scripts/          Data processing and evaluation
models/           LoRA adapters
results/          Benchmark runs and analysis
```

## Related

- [Solana Bench](https://github.com/solana-foundation/solana-gym-env) — the benchmark
- [Solana Bench blog](https://solana.com/news/solana-bench) — methodology and results
- [Surfpool](https://github.com/txtx/surfpool) — sandboxed Solana validator

## License

MIT
