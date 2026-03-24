# solana-lora-bench

Open-source model evaluation and fine-tuning dataset for [Solana Bench](https://github.com/solana-foundation/solana-gym-env).

## Key Findings

We benchmarked 6 open-source models on Solana Bench (10 runs each, 50 messages) and re-analyzed the original frontier model data. We discovered:

- **Claude Sonnet 4's reported score of 115 is 74% Memo inflation** — real score: 18.9
- **DeepSeek V3.2 (21.9) surpasses Claude Sonnet 4** when Memo gaming is controlled
- **GPT-5 (44.5) is the true leader** — barely uses Memo (2%)
- Two critical bugs in the upstream codebase silently discarded all valid transactions

Full analysis: [results/BENCHMARK_REPORT.md](results/BENCHMARK_REPORT.md)

## Combined Leaderboard (Memo Filtered)

| Rank | Model | Type | Score |
|------|-------|------|-------|
| 1 | GPT-5 | Frontier | 44.5 |
| 2 | DeepSeek V3.2 | Open-source | 21.9 |
| 3 | Gemini 2.5 Flash | Frontier | 20.2 |
| 4 | Claude Sonnet 4 | Frontier | 18.9 |
| 5 | GPT-oss-120B | Open-weight | 15.6 |
| 6 | Llama 4 Maverick | Open-source | 9.2 |
| 7 | Qwen3-235B-A22B | Open-source | 8.4 |
| 8 | Qwen3-32B | Open-source | 7.8 |
| 9 | Qwen3.5-35B-A3B | Open-source | 7.2 |
| 10 | Qwen3-30B-A3B | Open-source | 4.6 |

## Bug Fixes (submitted upstream)

1. **`KeyError(0)` in `_get_ordered_instructions`** — silently discarded ~66% of valid transactions
2. **`UnboundLocalError` in metrics tracking** — lost all message metrics when no code blocks found

Without these fixes, all open-source evaluations return 0.

## Dataset

1,305 validated training examples for Solana transaction code generation. Every example serializes correctly on Surfpool.

| Category | Count |
|----------|-------|
| Single-turn instruction/code pairs | ~1,240 |
| Multi-turn error-recovery conversations | ~65 |
| **Total (validated)** | **1,305** |

Covers: SystemProgram, ComputeBudget, SPL Token, Token-2022 (with extensions), ATA, Stake, Nonce, VersionedTransaction.

## Approach

1. Benchmark 6 open-source models on Solana Bench Basic (10 runs × 50 messages)
2. Extract error patterns (1,247 errors categorized) and successful transaction code
3. Generate targeted training data from error analysis + SDK documentation
4. Validate every example on Surfpool
5. LoRA fine-tune and re-evaluate

## Setup

```bash
git clone https://github.com/0xbstn/solana-lora-bench.git
cd solana-lora-bench && uv sync

# Validate dataset on Surfpool
surfpool start -u https://api.mainnet-beta.solana.com --no-tui
python scripts/validate_dataset.py -i data/processed/dataset.jsonl -c 10
```

## Repository Layout

```
data/
  processed/dataset.jsonl   Training dataset (1,305 validated examples)
scripts/
  validate_dataset.py       Validate examples on Surfpool
  parse_trajectories.py     Extract training pairs from benchmark runs
  prepare_hf_dataset.py     Prepare for HuggingFace upload
validator/
  run.ts                    Bun runner for transaction serialization testing
models/                     LoRA adapters
results/
  BENCHMARK_REPORT.md       Full benchmark analysis
```

## Related

- [Solana Bench](https://github.com/solana-foundation/solana-gym-env) — the benchmark
- [Surfpool](https://github.com/txtx/surfpool) — sandboxed Solana validator

## License

MIT
