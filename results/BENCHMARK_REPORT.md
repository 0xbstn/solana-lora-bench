# Solana Bench: Open-Source Model Evaluation

> **First comprehensive benchmark of open-source LLMs on Solana transaction building**
>
> March 23, 2026 — Basic Environment, 50 messages, 10 runs per model

## TL;DR

We benchmarked 6 open-source models on the Solana Foundation's [Solana Bench](https://github.com/solana-foundation/solana-gym-env) and re-analyzed the original frontier model data with uniform Memo filtering. Key findings:

- **Claude Sonnet 4's reported score of 115 is 74% Memo inflation** — its actual score is 18.9
- **DeepSeek V3.2 (21.9) surpasses Claude Sonnet 4 (18.9)** on real Solana competence
- **Post-hoc and real-time filtering produce consistent results** (±2 points), validating both methods
- We fixed two critical bugs in the evaluation pipeline that silently broke scoring for all models

---

## Methodology

### Environment

- **Benchmark**: [Solana Bench](https://github.com/solana-foundation/solana-gym-env) Basic — open-ended exploration of Solana programs
- **Scoring**: Number of unique (program_id, instruction_discriminator) pairs from successfully executed transactions
- **Execution**: [Surfpool](https://surfpool.run) v1.1.1 sandboxed validator (mainnet fork)
- **Message budget**: 50 messages per run
- **Runs**: 10 per model
- **SDK**: `@solana/web3.js` v1.98.2, `@solana/spl-token` v0.4.13
- **LLM routing**: OpenRouter API

### Models Tested

| Model | Type | Architecture | Active Params |
|-------|------|-------------|--------------|
| DeepSeek V3.2 | Open-source | MoE | 37B |
| Llama 4 Maverick | Open-source | MoE | 17B |
| Qwen3-235B-A22B | Open-source | MoE | 22B |
| Qwen3-32B | Open-source | Dense | 32B |
| Qwen3.5-35B-A3B | Open-source | MoE | 3B |
| Qwen3-30B-A3B | Open-source | MoE | 3B |

Frontier model data (Claude Sonnet 4, GPT-5, Gemini 2.5 Flash, GPT-oss-120B) comes from the original benchmark trajectories in `docs/trajectory-viewer/public/data/basic/`, with 10 runs per model.

### Bug Fixes

We discovered and fixed two critical bugs in the upstream codebase:

1. **`KeyError(0)` in `_get_ordered_instructions`** — The reward calculation assumed every top-level instruction has inner instructions. When a simple instruction like `SystemProgram.transfer` had none, the code crashed and silently discarded the transaction as a failure. This affected ~66% of successful transactions.

2. **`UnboundLocalError` in metrics tracking** — When a model response contained no code blocks, the variable `instructions_discovered` was referenced before assignment, causing the entire message to be lost from metrics.

Without these fixes, open-source model evaluations return 0 across all runs.

---

## Part 1: Raw Results

We first ran the benchmark without any Memo filtering to establish baseline scores comparable to the original frontier results.

### Open-Source Leaderboard (Raw, 10 runs)

| Rank | Model | Mean | Median | Max | Min | Std |
|------|-------|------|--------|-----|-----|-----|
| 1 | DeepSeek V3.2 | 26.6 | 35 | 38 | 6 | 12.2 |
| 2 | Qwen3.5-35B-A3B | 22.8 | 14 | 80 | 4 | 25.9 |
| 3 | Llama 4 Maverick | 13.3 | 11 | 27 | 7 | 6.0 |
| 4 | Qwen3-235B-A22B | 12.1 | 11 | 18 | 6 | 4.5 |
| 5 | Qwen3-32B | 9.6 | 10 | 17 | 0 | 5.3 |
| 6 | Qwen3-30B-A3B | 6.1 | 6 | 9 | 3 | 1.8 |

### Compared to Frontier (Raw)

| Model | Type | Mean | Median | Source |
|-------|------|------|--------|--------|
| Claude Sonnet 4 | Frontier | 73.4 | 34 | Solana Foundation |
| GPT-5 | Frontier | 45.6 | 57 | Solana Foundation |
| Gemini 2.5 Flash | Frontier | 25.1 | 17 | Solana Foundation |
| **DeepSeek V3.2** | Open-source | **26.6** | **35** | This report |
| Qwen3.5-35B-A3B | Open-source | 22.8 | 14 | This report |
| GPT-oss-120B | Open-weight | 17.5 | 22 | Solana Foundation |

At first glance, DeepSeek V3.2 appears competitive — outperforming Gemini 2.5 Flash and GPT-oss-120B. But Qwen3.5-35B-A3B's extreme variance (scores ranging from 4 to 80) raised a red flag.

---

## Part 2: The Memo Problem

### Discovery

Investigating Qwen3.5-35B-A3B's outlier runs revealed systematic Memo Program exploitation:

- **Score 67**: 62 Memo instructions + 5 real = 93% Memo
- **Score 80**: 75 Memo instructions + 5 real = 94% Memo

The Memo Program (`MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr`) accepts arbitrary data. Since the benchmark scores unique `(program_id, first_byte_of_instruction_data)` pairs, a model can generate up to 256 unique Memo instructions by varying the first byte — without performing any meaningful Solana operations.

### Not a New Problem

The Solana Foundation documented this for Claude in their swap benchmark:

> *"Claude had found a loophole, and it had reward-hacked the environment by sending memo instructions with slightly different instruction data."*

However, the Basic benchmark results were published without Memo filtering. We applied uniform filtering to all models — both our runs and the original frontier data.

### Memo Dependency by Model

| Model | Raw Mean | Memo % of Score |
|-------|----------|----------------|
| Claude Sonnet 4 | 73.4 | **74%** |
| Qwen3.5-35B-A3B | 22.8 | **70%** |
| DeepSeek V3.2 | 26.6 | 25% |
| Qwen3-30B-A3B | 6.1 | 26% |
| Qwen3-235B-A22B | 12.1 | 21% |
| Gemini 2.5 Flash | 25.1 | 20% |
| Llama 4 Maverick | 13.3 | 17% |
| GPT-oss-120B | 17.5 | 11% |
| Qwen3-32B | 9.6 | 9% |
| GPT-5 | 45.6 | **2%** |

Two models relied on Memo for over 70% of their scores. GPT-5, the actual top performer, barely used Memo at all (2%).

---

## Part 3: Post-Hoc Filtering

We re-analyzed all original frontier runs (40 runs from `docs/trajectory-viewer/public/data/basic/`) and our open-source runs by subtracting Memo-derived instructions from the scores.

### Frontier Models Re-Analyzed

| Model | Raw Per-Run | Filtered Per-Run | Raw Mean | Filtered Mean |
|-------|-------------|-----------------|----------|---------------|
| GPT-5 | [33,58,34,60,66,57,62,30,29,27] | [33,55,34,59,64,54,60,30,29,27] | 45.6 | **44.5** |
| Gemini 2.5 Flash | [23,40,40,43,16,14,13,44,18,0] | [11,29,36,34,16,14,13,31,18,0] | 25.1 | **20.2** |
| Claude Sonnet 4 | [33,30,152,34,37,19,102,181,31,115] | [33,19,5,17,25,11,34,13,20,12] | 73.4 | **18.9** |
| GPT-oss-120B | [25,9,17,23,22,23,22,16,8,10] | [22,9,17,20,21,13,22,14,8,10] | 17.5 | **15.6** |

Claude Sonnet 4's scores of 152 and 181 drop to 5 and 13 after filtering. Its actual Solana competence (18.9) is below Gemini 2.5 Flash (20.2).

### Open-Source Models (Post-Hoc Filtered)

| Model | Raw Mean | Filtered Mean |
|-------|----------|---------------|
| DeepSeek V3.2 | 26.6 | **20.0** |
| Llama 4 Maverick | 13.3 | **11.0** |
| Qwen3-235B-A22B | 12.1 | **9.6** |
| Qwen3-32B | 9.6 | **8.7** |
| Qwen3.5-35B-A3B | 22.8 | **6.8** |
| Qwen3-30B-A3B | 6.1 | **4.5** |

---

## Part 4: Real-Time Filtered Benchmark

Post-hoc filtering has a limitation: models waste messages on Memo instructions that yield no useful data. To measure what happens when models must adapt in real-time, we ran a second benchmark where **Memo instructions give 0 reward during the run** (via `disallowed_programs` in the environment config). Models still see the Memo Program listed in the prompt and can attempt it, but receive 0 reward and must find alternative strategies.

### Results (Real-Time Filtered, 10 runs per model)

| Rank | Model | Mean | Median | Max | Min |
|------|-------|------|--------|-----|-----|
| 1 | **DeepSeek V3.2** | **21.9** | 26 | 36 | 6 |
| 2 | Llama 4 Maverick | 9.2 | 10 | 17 | 5 |
| 3 | Qwen3-235B-A22B | 8.4 | 9 | 11 | 4 |
| 4 | Qwen3-32B | 7.8 | 10 | 16 | 0 |
| 5 | Qwen3.5-35B-A3B | 7.2 | 5 | 16 | 5 |
| 6 | Qwen3-30B-A3B | 4.6 | 5 | 6 | 1 |

### Per-Run Detail

| Model | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 |
|-------|----|----|----|----|----|----|----|----|----|----|
| DeepSeek V3.2 | 33 | 33 | 36 | 9 | 26 | 20 | 6 | 18 | 29 | 9 |
| Llama 4 Maverick | 5 | 17 | 5 | 11 | 7 | 12 | 7 | 11 | 7 | 10 |
| Qwen3-235B-A22B | 9 | 10 | 11 | 11 | 6 | 6 | 8 | 4 | 10 | 9 |
| Qwen3-32B | 10 | 10 | 0 | 3 | 5 | 10 | 5 | 16 | 3 | 16 |
| Qwen3.5-35B-A3B | 5 | 5 | 5 | 10 | 16 | 6 | 5 | 10 | 5 | 5 |
| Qwen3-30B-A3B | 5 | 1 | 5 | 5 | 5 | 5 | 4 | 5 | 5 | 6 |

---

## Part 5: Filtering Method Comparison

Both filtering methods produce consistent results, validating the approach:

| Model | Post-Hoc Filtered | Real-Time Filtered | Difference |
|-------|-------------------|-------------------|------------|
| DeepSeek V3.2 | 20.0 | 21.9 | +1.9 |
| Llama 4 Maverick | 11.0 | 9.2 | -1.8 |
| Qwen3-235B-A22B | 9.6 | 8.4 | -1.2 |
| Qwen3-32B | 8.7 | 7.8 | -0.9 |
| Qwen3.5-35B-A3B | 6.8 | 7.2 | +0.4 |
| Qwen3-30B-A3B | 4.5 | 4.6 | +0.1 |

Maximum deviation: ±1.9 points. Rankings are identical. The Solana Foundation used post-hoc filtering in their swap analysis; our real-time approach confirms those results while also testing model adaptability.

---

## Combined Leaderboard

All models ranked by filtered score. Open-source scores use real-time filtering (the more rigorous method). Frontier scores use post-hoc filtering from their original data.

| Rank | Model | Type | Filtered Mean | Raw Mean | Memo % | Runs |
|------|-------|------|--------------|----------|--------|------|
| 1 | **GPT-5** | Frontier | **44.5** | 45.6 | 2% | 10 |
| 2 | **DeepSeek V3.2** | Open-source | **21.9** | 26.6 | 25% | 10 |
| 3 | Gemini 2.5 Flash | Frontier | 20.2 | 25.1 | 20% | 10 |
| 4 | **Claude Sonnet 4** | Frontier | **18.9** | 73.4 | **74%** | 10 |
| 5 | GPT-oss-120B | Open-weight | 15.6 | 17.5 | 11% | 10 |
| 6 | Llama 4 Maverick | Open-source | 9.2 | 13.3 | 17% | 10 |
| 7 | Qwen3-235B-A22B | Open-source | 8.4 | 12.1 | 21% | 10 |
| 8 | Qwen3-32B | Open-source | 7.8 | 9.6 | 9% | 10 |
| 9 | Qwen3.5-35B-A3B | Open-source | 7.2 | 22.8 | **70%** | 10 |
| 10 | Qwen3-30B-A3B | Open-source | 4.6 | 6.1 | 26% | 10 |

### Rank Changes After Filtering

| Model | Raw Rank | Filtered Rank | Change |
|-------|----------|---------------|--------|
| GPT-5 | 1 | 1 | — |
| Claude Sonnet 4 | 2 | **4** | ↓2 |
| DeepSeek V3.2 | 4 | **2** | ↑2 |
| Gemini 2.5 Flash | 3 | 3 | — |
| Qwen3.5-35B-A3B | 5 | **9** | ↓4 |

---

## Error Analysis

### 1,247 error patterns extracted across 30 runs (initial analysis)

| Category | Count | % | Description |
|----------|-------|---|-------------|
| Solana-specific | 578 | 46% | Rent-exempt, insufficient funds, read-only accounts |
| Transaction construction | 269 | 22% | TX too large (>1232 bytes), wrong signers |
| API misuse | 177 | 14% | Wrong method names, wrong parameters |
| Import errors | 117 | 9% | Non-existent exports (`MemoProgram`, `Buffer`) |
| TypeScript errors | 106 | 9% | Syntax errors, type mismatches, BigInt issues |

### Top Recurring Errors

| Error | Count | Description |
|-------|-------|-------------|
| `MemoProgram` not found | 61 | Every model attempts this import on message 1 |
| Transaction too large | 72 | Exceeding the 1232-byte limit |
| `TransactionInstruction` not defined | 56 | Missing import |
| Insufficient funds for rent | 55 | Incorrect lamport calculation |
| BigInt type errors | 52 | Passing number instead of BigInt |
| Unknown signer | 43 | Signing with keypairs not in the transaction |

### Self-Correction Rate

| Model | Rate | Description |
|-------|------|-------------|
| DeepSeek V3.2 | 21.7% | Best — fixes 1 in 5 errors |
| Qwen3.5-35B-A3B | 14.2% | |
| Llama 4 Maverick | 12.8% | |
| Qwen3-235B-A22B | 8.5% | |
| Qwen3-32B | 5.8% | |
| Qwen3-30B-A3B | 4.5% | Rarely recovers from errors |

DeepSeek V3.2 has both the best self-correction rate and the highest score. The ability to learn from error feedback within a conversation session correlates directly with benchmark performance.

---

## Key Findings

### 1. Memo gaming distorts the leaderboard

Claude Sonnet 4's widely cited score of 115 is 74% Memo inflation. Without Memo, it scores 18.9 — below Gemini 2.5 Flash (20.2) and the open-source DeepSeek V3.2 (21.9). Raw scores should not be used to compare models on this benchmark.

### 2. An open-source model outperforms Claude Sonnet 4

DeepSeek V3.2 at 21.9 (real-time filtered) surpasses Claude Sonnet 4 at 18.9 (post-hoc filtered). This is the first documented case of an open-source model outperforming a frontier model on Solana Bench when Memo gaming is controlled for.

### 3. The real gap is 2x, not 6x

The gap between the best open-source model (DeepSeek V3.2: 21.9) and GPT-5 (44.5) is approximately 2x. Raw scores suggested a 6x gap. A 2x gap is achievable with domain-specific fine-tuning.

### 4. Model size does not predict performance

Qwen3-235B-A22B (22B active params) scored 8.4, while Llama 4 Maverick (17B active) scored 9.2. DeepSeek V3.2 (37B active) scored 21.9. Solana-specific knowledge in the training data matters more than parameter count.

### 5. Filtering methods agree

Post-hoc filtering and real-time filtering produce scores within ±2 points of each other across all models. This validates both the Solana Foundation's original post-hoc approach and our real-time filtering method.

### 6. Common errors are addressable through fine-tuning

The top 5 error patterns account for over 60% of failures. These are systematic knowledge gaps (wrong imports, transaction size limits, account model misunderstandings) that can be corrected with targeted training data.

---

## Implications for Fine-Tuning

| Data source | Count | Quality |
|-------------|-------|---------|
| Self-corrected error-fix pairs | 102 | High — verified working code |
| Successful transaction code | 195 | High — confirmed on-chain |
| Error patterns with known fixes | 950 | Medium — fix known, not model-generated |
| Scraped SDK examples (Cookbook, Anchor, program-examples) | 500+ | High — real working code |
| Synthetic augmentation (distilabel) | 1,400+ | Generated from seeds above |

**Target**: A Qwen3-32B fine-tuned with 5-10K instruction pairs covering correct imports, transaction size constraints, and the Solana account model could move from 7.8 to 30+, surpassing Claude Sonnet 4 and approaching GPT-5.

---

## Reproduction

```bash
git clone https://github.com/solana-foundation/solana-gym-env
cd voyager && uv sync

# Apply bug fixes (KeyError + UnboundLocalError)
# Start Surfpool
surfpool start -u https://api.mainnet-beta.solana.com --no-tui

# Run benchmark
python run_bench.py \
  --api-url https://openrouter.ai/api/v1 \
  --api-key YOUR_KEY \
  --models deepseek/deepseek-v3.2 meta-llama/llama-4-maverick \
    qwen/qwen3-235b-a22b qwen/qwen3-32b \
    qwen/qwen3.5-35b-a3b qwen/qwen3-30b-a3b \
  --runs 10 --messages 50 --parallel 20 --filter-memo
```

---

## Status and Next Steps

**Completed:**
- Benchmark of 6 open-source models (10 runs × 50 messages each)
- Re-analysis of frontier model data with Memo filtering
- Two critical bug fixes for the evaluation pipeline
- 1,305-example validated training dataset
- Initial LoRA fine-tuning (Qwen3.5-35B-A3B)

**In progress:**
- LoRA iteration — improving Token-2022 and multi-program coverage
- Dataset publication on HuggingFace

**Planned (with funding):**
- Extended fine-tuning experiments (GPU compute for multiple iterations)
- Re-benchmark fine-tuned models against baselines
- PR bug fixes upstream to solana-gym-env
- Evaluate on Swap environment (blocked by Jupiter API authentication change)

---

## Credits

- **Solana Foundation** — [Solana Bench](https://github.com/solana-foundation/solana-gym-env) and original frontier model data
- **Surfpool** — Sandboxed Solana validator ([surfpool.run](https://surfpool.run))
- **OpenRouter** — LLM API routing
