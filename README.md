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
2. Augment with synthetic generation (self-instruct, variations, error-recovery, targeted patterns)
3. Validate every example on Surfpool — only keep code that serializes correctly
4. LoRA fine-tune on Qwen 3.5-35B-A3B (MoE, 3B active params)
5. Evaluate on Solana Bench under the same conditions as frontier models

## Open-Source Baseline Results

Open-source model baseline on Solana Bench ([full report](results/BENCHMARK_REPORT.md)):

| Model | Median | Mean | Max | Success Rate |
|-------|--------|------|-----|-------------|
| Qwen3.5-35B-A3B | 18 | 51.2 | 108 | 28% |
| DeepSeek V3.2 | 20 | 28.4 | 37 | 22% |
| Llama 4 Maverick | 14 | 13.6 | 16 | 12% |
| Qwen3-235B-A22B | 12 | 10.6 | 15 | 9% |

## Dataset

1,439 validated training examples. Every example serializes correctly on Surfpool.

| Source | Count | Description |
|--------|-------|-------------|
| Self-instruct | 454 | Generated task descriptions + code |
| Solana cookbook (regen) | 489 | Cookbook prompts with regenerated offline code |
| Targeted patterns | 290 | Targeted at top 6 error categories from benchmark |
| Benchmark trajectories | 100 | Original successful code from Solana Bench runs |
| Variations | 97 | Variations of high-reward examples |
| Error-recovery pairs | 9 | Buggy code + fix pairs |

## Setup

```bash
# Clone
git clone https://github.com/0xbstn/solana-lora-bench.git
cd solana-lora-bench
uv sync

# Parse trajectories (requires solana-bench data in ../solana-bench/)
python scripts/parse_trajectories.py

# Generate synthetic data (requires OpenAI-compatible API on localhost:8318)
python scripts/generate_synthetic.py       # self-instruct + variations + error-recovery
python scripts/generate_synthetic_v2.py    # targeted error patterns

# Validate on Surfpool (requires surfpool running on localhost:8899)
python scripts/validate_dataset.py -c 20
```

## Repository Layout

```
data/
  processed/        Training dataset (dataset_base.jsonl)
  swap/             DeFi/swap examples (excluded from basic benchmark)
scripts/
  parse_trajectories.py      Extract training pairs from Solana Bench runs
  generate_synthetic.py      Self-instruct + variations + error-recovery generation
  generate_synthetic_v2.py   Targeted generation for top 6 error patterns
  validate_dataset.py        Validate examples on Surfpool (serialization check)
validator/
  run.ts            Bun runner for transaction serialization testing
  package.json      Dependencies for validation
models/             LoRA adapters (after training)
results/            Benchmark reports and error analysis
```

## Related

- [Solana Bench](https://github.com/solana-foundation/solana-gym-env) — the benchmark
- [Solana Bench blog](https://solana.com/news/solana-bench) — methodology and results
- [Surfpool](https://github.com/txtx/surfpool) — sandboxed Solana validator

## License

MIT
