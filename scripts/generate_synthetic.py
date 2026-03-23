#!/usr/bin/env python3
"""
Generate synthetic Solana training data using Claude via VibeProxy.
Async parallel execution.

3 strategies:
1. Self-Instruct: Generate new task descriptions then code them
2. Variations: Take best examples and create variations
3. Error-recovery: Generate buggy+fixed pairs

Output: data/processed/synthetic_v1.jsonl
"""

import asyncio
import json
import os
import re
import hashlib
from openai import AsyncOpenAI

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FILE = os.path.join(BASE, "data/processed/clean_train.jsonl")
OUTPUT_FILE = os.path.join(BASE, "data/processed/synthetic_v1.jsonl")

SYSTEM_PROMPT = (
    "You are an expert Solana developer. Write TypeScript code using @solana/web3.js "
    "to build and execute Solana transactions. Always use the function signature: "
    "export async function executeSkill(blockhash: string): Promise<string>. "
    "Return base64 encoded serialized transactions.\n\n"
    "CRITICAL RULES — OFFLINE TRANSACTION BUILDING:\n"
    "- Do NOT import or use Connection. No network calls. Build transactions purely offline.\n"
    "- Use getAssociatedTokenAddressSync() NOT getAssociatedTokenAddress() (async needs Connection).\n"
    "- Do NOT call connection.getMinimumBalanceForRentExemption(). Use hardcoded rent values:\n"
    "  Mint (82 bytes): 1_461_600 lamports. Token account (165 bytes): 2_039_280 lamports.\n"
    "  Basic account (128 bytes): 1_461_600 lamports. Stake (200 bytes): 2_282_880 lamports.\n"
    "  Nonce (80 bytes): 1_447_680 lamports.\n"
    "- Wallet placeholder: new PublicKey('11111111111111111111111111111111') for any placeholder address.\n"
    "- tx.recentBlockhash = blockhash; tx.feePayer = walletPubkey;\n"
    "- return tx.serialize({requireAllSignatures: false, verifySignatures: false}).toString('base64');"
)

MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3
CONCURRENCY = 5

client = AsyncOpenAI(
    base_url="http://localhost:8318/v1",
    api_key="not-needed",
)

sem = asyncio.Semaphore(CONCURRENCY)
file_lock = asyncio.Lock()
seen = set()
total_generated = 0
total_errors = 0


async def chat(messages, temperature=0.7, max_tokens=4096):
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"    Retry {attempt+1}: {e}")
                    await asyncio.sleep(wait)
                else:
                    raise


def extract_code(text):
    m = re.search(r'```(?:typescript|ts)?\n(.*?)```', text, re.DOTALL)
    if m:
        raw = m.group(1).strip()
        return raw, f"```typescript\n{raw}\n```"
    return None, None


def code_hash(text):
    raw, _ = extract_code(text)
    if raw is None:
        raw = text
    return hashlib.md5(re.sub(r'\s+', ' ', raw.strip()).encode()).hexdigest()


def has_valid_code(raw_code):
    if "serialize" not in raw_code and "base64" not in raw_code:
        return False
    disallowed = ["@raydium-io/", "@orca-so/", "@jup-ag/", "axios", "node-fetch", "decimal.js"]
    return not any(d in raw_code for d in disallowed)


def load_existing():
    with open(TRAIN_FILE) as f:
        return [json.loads(l) for l in f if l.strip()]


async def save(entry):
    global total_generated
    async with file_lock:
        with open(OUTPUT_FILE, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        total_generated += 1


# =============================================================================
# Prompts
# =============================================================================

TASK_GEN_PROMPT = """Generate {n} UNIQUE and DIVERSE Solana TypeScript task descriptions.
Each task should describe building an OFFLINE transaction using @solana/web3.js v1.
The code must build and serialize a transaction WITHOUT any network calls (no Connection, no RPC).

ALLOWED packages ONLY: @solana/web3.js, @solana/spl-token, @coral-xyz/anchor, bs58, bn.js, buffer
FORBIDDEN: @raydium-io/*, @orca-so/*, @jup-ag/*, axios, node-fetch, Connection

DO NOT generate Memo-only tasks. Memo can be added as a secondary instruction but should NEVER be the main operation.
Focus on REAL Solana operations that demonstrate actual on-chain skills.

Cover these categories (mix them, be creative, vary complexity):
- SystemProgram: transfer, createAccount, assign, allocate, createAccountWithSeed
- ComputeBudgetProgram: setComputeUnitLimit, setComputeUnitPrice
- SPL Token: createMint, mintTo, transfer, burn, approve, revoke, closeAccount, freezeAccount, thawAccount
- Token-2022: transfer fee extension, metadata extension, immutable owner, permanent delegate, mint close authority
- PDA derivation with PublicKey.findProgramAddressSync
- Multi-instruction transactions (combine 3+ different operations)
- VersionedTransaction with MessageV0
- Stake: createStakeAccount, delegate, deactivate, withdraw, split, merge
- Nonce: createNonceAccount, nonceAdvance, nonceWithdraw
- Anchor: program.methods.xxx(), PDA accounts, initialize + call patterns
- Account seeds: createAccountWithSeed, findProgramAddressSync

Each task: 1-2 sentences, SPECIFIC (include amounts, addresses, program names).
Make batch {batch_id} different from previous batches — focus on: {focus}

Return ONLY a JSON array of strings."""

TASK_FOCUSES = [
    "SystemProgram operations with various amounts and multiple transfers",
    "SPL Token lifecycle (create mint, mint tokens, transfer, burn, close)",
    "ComputeBudget + Memo combinations with creative memo messages",
    "PDA derivation and Anchor program interactions",
    "Multi-instruction transactions combining 4+ different operations",
    "VersionedTransaction with AddressLookupTable and multiple instructions",
    "Stake operations (delegate, deactivate, withdraw, split)",
    "Token-2022 extensions (transfer fee, metadata, permanent delegate)",
    "Nonce accounts and offline transaction signing patterns",
    "Complex transactions: create accounts + tokens + transfers + memos in one tx",
]

CODE_GEN_PROMPT = """Write TypeScript code for this Solana task:

{task}

Requirements:
1. Signature: export async function executeSkill(blockhash: string): Promise<string>
2. Import from @solana/web3.js v1 (Transaction, PublicKey, Keypair, SystemProgram, etc.)
3. If needed: @solana/spl-token or @coral-xyz/anchor
4. DO NOT import @raydium-io, @orca-so, @jup-ag, axios, node-fetch
5. OFFLINE ONLY: no Connection, no network calls
6. Use getAssociatedTokenAddressSync (not async version)
7. Hardcode rent values, do NOT call getMinimumBalanceForRentExemption
8. Wallet placeholder: new PublicKey('11111111111111111111111111111111')
9. Build transaction and return base64:
   tx.recentBlockhash = blockhash;
   tx.feePayer = walletPubkey;
   return tx.serialize({{requireAllSignatures: false, verifySignatures: false}}).toString('base64');
10. Complete, self-contained code
11. Wrap in ```typescript ... ```"""

VARIATION_PROMPT = """Here is existing Solana TypeScript code:

```typescript
{code}
```

Create a DIFFERENT variation: {instruction}

Requirements:
- Signature: export async function executeSkill(blockhash: string): Promise<string>
- @solana/web3.js v1, @solana/spl-token, @coral-xyz/anchor ONLY
- OFFLINE ONLY: no Connection, no network calls, use getAssociatedTokenAddressSync
- Hardcode rent values, no getMinimumBalanceForRentExemption
- Placeholder: new PublicKey('11111111111111111111111111111111')
- Return base64 serialized transaction
- Complete and self-contained

First line: TASK: [1-sentence description]
Then the code in ```typescript ... ```"""

ERROR_PROMPT = """Generate a Solana TypeScript error-recovery training example.

Scenario: {scenario}

Write BUGGY code with a realistic error, then the FIXED version.

Format:
TASK: [what the code does]
BUGGY:
```typescript
[buggy code with executeSkill signature]
```
ERROR: [realistic error message that Solana/TypeScript would produce]
FIXED:
```typescript
[fixed code with executeSkill signature]
```

Both must use executeSkill(blockhash), @solana/web3.js v1, return base64.
OFFLINE ONLY: no Connection, no network calls, use getAssociatedTokenAddressSync, hardcode rent values.
Placeholder: new PublicKey('11111111111111111111111111111111')."""

VARIATION_INSTRUCTIONS = [
    "Add ComputeBudgetProgram instructions (setComputeUnitLimit + setComputeUnitPrice) before the main operations",
    "Add a Memo instruction with a descriptive message after the main operations",
    "Use multiple Keypair.generate() and include them as additional signers with partialSign",
    "Use VersionedTransaction with TransactionMessage.compile and MessageV0 instead of legacy Transaction",
    "Combine with an SPL token operation (create ATA + mint tokens)",
    "Add SystemProgram.createAccount to create a new account as part of the transaction",
    "Double the instructions by repeating core logic with different parameters and amounts",
    "Add stake-related instructions (create stake account and delegate to a validator)",
    "Use createAccountWithSeed instead of regular createAccount",
    "Add a nonce advance instruction at the beginning using SystemProgram.nonceAdvance",
]

ERROR_SCENARIOS = [
    "Transfer SOL but forget to set feePayer on the transaction",
    "Create SPL token mint but allocate wrong space (should be MINT_SIZE from @solana/spl-token)",
    "Transfer tokens but destination associated token account doesn't exist yet",
    "Set compute unit limit above maximum (1.4M units) causing transaction rejection",
    "Derive PDA with wrong seeds order causing account not found",
    "Create account with insufficient lamports for rent exemption",
    "Close token account that still has remaining token balance",
    "Build VersionedTransaction but forget to resolve address lookup table accounts",
    "Mint tokens but the signer is not the mint authority",
    "Create stake account with wrong StakeProgram program ID",
    "Use wrong number of bytes for u64 in instruction data Buffer",
    "Initialize mint with decimals as BigInt instead of number",
    "Add too many instructions exceeding 1232 byte transaction size limit",
    "Use deprecated Memo program ID instead of MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr",
    "Create ATA for wrong mint address causing token mismatch",
    "Forget partialSign for generated Keypair that is a required signer",
    "Set feePayer to a PublicKey that isn't included as a signer",
    "Try to assign account to a program but account already has data",
    "Use SystemProgram.transfer with zero lamports (not allowed)",
    "Create two accounts with same Keypair in one transaction causing duplicate key error",
]


# =============================================================================
# Phase 1: Self-Instruct
# =============================================================================

async def generate_tasks():
    """Generate task descriptions in parallel batches."""
    all_tasks = []

    async def gen_batch(batch):
        focus = TASK_FOCUSES[batch % len(TASK_FOCUSES)]
        try:
            text = await chat(
                [{"role": "user", "content": TASK_GEN_PROMPT.format(
                    n=50, batch_id=batch+1, focus=focus
                )}],
                temperature=0.9,
            )
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                tasks = json.loads(match.group())
                print(f"  Batch {batch+1}/10: {len(tasks)} tasks ({focus[:30]}...)")
                return tasks
        except Exception as e:
            print(f"  Batch {batch+1} failed: {e}")
        return []

    results = await asyncio.gather(*[gen_batch(i) for i in range(10)])
    for tasks in results:
        all_tasks.extend(tasks)

    return list(set(all_tasks))


async def code_one_task(task):
    """Generate code for a single task."""
    global total_errors
    try:
        text = await chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": CODE_GEN_PROMPT.format(task=task)},
        ])

        raw, wrapped = extract_code(text)
        if raw is None:
            raw = text.strip()
            wrapped = f"```typescript\n{raw}\n```"

        if not has_valid_code(raw):
            return

        h = code_hash(wrapped)
        if h in seen:
            return
        seen.add(h)

        await save({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
                {"role": "assistant", "content": wrapped},
            ],
            "metadata": {
                "type": "synthetic",
                "source": "distilabel/self-instruct",
                "model": MODEL,
                "protocol": "system",
            },
        })
    except Exception:
        total_errors += 1


# =============================================================================
# Phase 2: Variations
# =============================================================================

async def generate_variation(seed, var_inst):
    """Generate one variation from a seed example."""
    global total_errors
    raw_content = seed["messages"][2]["content"]
    raw, _ = extract_code(raw_content)
    if raw is None:
        raw = raw_content

    try:
        text = await chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": VARIATION_PROMPT.format(
                code=raw[:2000], instruction=var_inst
            )},
        ])

        task_m = re.search(r'TASK:\s*(.+?)(?:\n|$)', text)
        raw_var, wrapped_var = extract_code(text)
        if raw_var is None:
            return

        if not has_valid_code(raw_var):
            return

        h = code_hash(wrapped_var)
        if h in seen:
            return
        seen.add(h)

        await save({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task_m.group(1).strip() if task_m else f"Variation: {var_inst}"},
                {"role": "assistant", "content": wrapped_var},
            ],
            "metadata": {
                "type": "synthetic",
                "source": "distilabel/variation",
                "model": MODEL,
                "protocol": seed["metadata"].get("protocol", "system"),
            },
        })
    except Exception:
        total_errors += 1


# =============================================================================
# Phase 3: Error-recovery
# =============================================================================

async def generate_error_recovery(scenario):
    """Generate one error-recovery pair."""
    global total_errors
    try:
        text = await chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ERROR_PROMPT.format(scenario=scenario)},
        ])

        task_m = re.search(r'TASK:\s*(.+?)(?:\n|$)', text)
        buggy_m = re.search(r'BUGGY:\s*```(?:typescript|ts)?\n(.*?)```', text, re.DOTALL)
        error_m = re.search(r'ERROR:\s*(.+?)(?:\n|$)', text)
        fixed_m = re.search(r'FIXED:\s*```(?:typescript|ts)?\n(.*?)```', text, re.DOTALL)

        if not fixed_m:
            return

        raw_fixed = fixed_m.group(1).strip()
        wrapped_fixed = f"```typescript\n{raw_fixed}\n```"

        if not has_valid_code(raw_fixed):
            return

        h = code_hash(wrapped_fixed)
        if h in seen:
            return
        seen.add(h)

        task = task_m.group(1).strip() if task_m else scenario

        # Entry 1: Success example
        await save({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
                {"role": "assistant", "content": wrapped_fixed},
            ],
            "metadata": {
                "type": "synthetic",
                "source": "distilabel/error-recovery",
                "model": MODEL,
                "protocol": "system",
            },
        })

        # Entry 2: Error-fix pair with buggy code
        if buggy_m and error_m:
            raw_buggy = buggy_m.group(1).strip()
            error_msg = error_m.group(1).strip()

            fix_prompt = (
                f"The previous transaction failed with this error:\n"
                f"{error_msg}\n\n"
                f"Here is the code that failed:\n"
                f"```typescript\n{raw_buggy}\n```\n\n"
                f"Fix the code to make it work."
            )

            await save({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": fix_prompt},
                    {"role": "assistant", "content": wrapped_fixed},
                ],
                "metadata": {
                    "type": "synthetic",
                    "source": "distilabel/error-recovery",
                    "model": MODEL,
                    "protocol": "system",
                },
            })

    except Exception:
        total_errors += 1


# =============================================================================
# Main
# =============================================================================

async def main():
    global total_generated, total_errors

    existing = load_existing()
    for e in existing:
        seen.add(code_hash(e["messages"][2]["content"]))

    # Also load v2 targeted to avoid dupes
    targeted_path = os.path.join(BASE, "data/processed/synthetic_targeted.jsonl")
    if os.path.exists(targeted_path):
        with open(targeted_path) as f:
            for line in f:
                if line.strip():
                    try:
                        e = json.loads(line)
                        seen.add(code_hash(e["messages"][2]["content"]))
                    except:
                        pass

    open(OUTPUT_FILE, 'w').close()

    print(f"Loaded {len(existing)} existing, {len(seen)} unique hashes")
    print(f"Concurrency: {CONCURRENCY}")
    print()

    # === Phase 1: Self-Instruct ===
    print("=== Phase 1: Self-Instruct (generating tasks) ===")
    all_tasks = await generate_tasks()
    print(f"  Unique tasks: {len(all_tasks)}")

    print(f"\n=== Phase 1: Self-Instruct (coding {len(all_tasks)} tasks) ===")
    batch_size = 20
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i:i+batch_size]
        await asyncio.gather(*[code_one_task(t) for t in batch])
        print(f"  Progress: {min(i+batch_size, len(all_tasks))}/{len(all_tasks)} | generated: {total_generated} | errors: {total_errors}")

    si_count = total_generated
    print(f"  Self-instruct: {si_count} examples")

    # === Phase 2: Variations ===
    print("\n=== Phase 2: Variations ===")
    bench = sorted(
        [e for e in existing if e["metadata"].get("type") in ("success", "error_fix")],
        key=lambda x: -x["metadata"].get("reward", 0),
    )[:20]

    scraped_by_proto = {}
    for e in existing:
        if e["metadata"].get("type") == "scraped":
            p = e["metadata"].get("protocol", "?")
            if p not in scraped_by_proto:
                scraped_by_proto[p] = []
            if len(scraped_by_proto[p]) < 5:
                scraped_by_proto[p].append(e)

    scraped_seeds = []
    for proto_examples in scraped_by_proto.values():
        scraped_seeds.extend(proto_examples[:4])
    seeds = bench + scraped_seeds[:20]
    print(f"  Seeds: {len(seeds)}")

    var_tasks = []
    for j, seed in enumerate(seeds):
        start_idx = (j * 4) % len(VARIATION_INSTRUCTIONS)
        for k in range(4):
            idx = (start_idx + k) % len(VARIATION_INSTRUCTIONS)
            var_tasks.append((seed, VARIATION_INSTRUCTIONS[idx]))

    print(f"  Variation tasks: {len(var_tasks)}")
    before = total_generated
    for i in range(0, len(var_tasks), batch_size):
        batch = var_tasks[i:i+batch_size]
        await asyncio.gather(*[generate_variation(s, v) for s, v in batch])
        print(f"  Progress: {min(i+batch_size, len(var_tasks))}/{len(var_tasks)} | variations: {total_generated - before}")

    print(f"  Variations: {total_generated - before} examples")

    # === Phase 3: Error-recovery ===
    print("\n=== Phase 3: Error-recovery ===")
    before = total_generated
    await asyncio.gather(*[generate_error_recovery(s) for s in ERROR_SCENARIOS])
    print(f"  Error-recovery: {total_generated - before} examples")

    # === Summary ===
    print(f"\n{'='*50}")
    print(f"TOTAL GENERATED: {total_generated}")
    print(f"API errors: {total_errors}")
    print(f"Saved to {OUTPUT_FILE}")

    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        with open(OUTPUT_FILE) as f:
            final = [json.loads(l) for l in f if l.strip()]
        sources = {}
        for e in final:
            s = e["metadata"]["source"]
            sources[s] = sources.get(s, 0) + 1
        print(f"By source: {sources}")


if __name__ == "__main__":
    asyncio.run(main())
