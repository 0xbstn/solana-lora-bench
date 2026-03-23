#!/usr/bin/env python3
"""
Validate training examples by running them against Surfpool.
Single script: validates, shows progress, categorizes errors, outputs clean JSONL.

Usage:
  python scripts/validate_dataset.py                                    # all default files
  python scripts/validate_dataset.py -i data/processed/synthetic_v1.jsonl  # specific file
  python scripts/validate_dataset.py -c 20                              # more parallelism
  python scripts/validate_dataset.py --errors-only                      # only show failures
"""

import asyncio
import json
import os
import re
import base64
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
VALIDATOR_DIR = BASE / "validator"
RUNNER = VALIDATOR_DIR / "run.ts"
TMP_DIR = VALIDATOR_DIR / "tmp"
DEFAULT_INPUTS = [
    BASE / "data/processed/fixed_train.jsonl",
    BASE / "data/processed/fixed_valid.jsonl",
    BASE / "data/processed/synthetic_targeted.jsonl",
    BASE / "data/processed/synthetic_v1.jsonl",
]
OUTPUT_DIR = BASE / "data/processed"
RPC_URL = "http://localhost:8899"

# Known valid Solana program IDs — keep these, replace everything else
KNOWN_PROGRAM_IDS = {
    '11111111111111111111111111111111',
    'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA',
    'TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb',
    'ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL',
    'ComputeBudget111111111111111111111111111111',
    'MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr',
    'Memo1UhkJBfCR6MNB5bUiSVgSayR7NZNPbczGc5vHNUjH',
    'Stake11111111111111111111111111111111111111',
    'SysvarRent111111111111111111111111111111111',
    'SysvarC1ock11111111111111111111111111111111',
    'SysvarStakeHistory1111111111111111111111111',
    'StakeConfig11111111111111111111111111111111',
    'Vote111111111111111111111111111111111111111',
}


def get_blockhash() -> str:
    import urllib.request
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getLatestBlockhash"}
    req = urllib.request.Request(
        RPC_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=10)
    return json.loads(resp.read())["result"]["value"]["blockhash"]


def extract_code(text: str) -> str | None:
    m = re.search(r'```(?:typescript|ts)?\n(.*?)```', text, re.DOTALL)
    return m.group(1).strip() if m else None


def fix_pubkeys_for_test(raw: str) -> str:
    """Replace all non-program-ID pubkeys with valid placeholders for testing."""
    def _replace(match):
        pk = match.group(1)
        if pk in KNOWN_PROGRAM_IDS:
            return match.group(0)
        return "new PublicKey('11111111111111111111111111111111')"
    return re.sub(r"new PublicKey\(['\"]([^'\"]+)['\"]\)", _replace, raw)


def setup_validator():
    if not (VALIDATOR_DIR / "node_modules").exists():
        print("Installing validator dependencies...", flush=True)
        subprocess.run(["bun", "install"], cwd=str(VALIDATOR_DIR), check=True, capture_output=True)
        print("Done.", flush=True)
    TMP_DIR.mkdir(exist_ok=True)


def categorize_error(error: str) -> str:
    if not error:
        return "unknown"
    if "is not a function" in error:
        return "api_misuse"
    if "is not defined" in error:
        return "missing_import"
    if "not found in module" in error or "Export named" in error:
        return "bad_export"
    if "timeout" in error:
        return "timeout"
    if "too large" in error or "1232" in error:
        return "tx_too_large"
    if "SyntaxError" in error or "Unexpected" in error:
        return "syntax_error"
    if "invalid base64" in error:
        return "invalid_base64"
    if "Invalid public key" in error or "Non-base58" in error:
        return "invalid_pubkey"
    if "Cannot access" in error and "before initialization" in error:
        return "hoisting_error"
    if "unknown signer" in error:
        return "unknown_signer"
    if "undefined is not an object" in error:
        return "undefined_access"
    return "other"


async def run_one(idx: int, entry: dict, blockhash: str, sem: asyncio.Semaphore, timeout: int = 15) -> dict:
    """Validate a single entry. Returns validation result dict."""
    code = entry["messages"][2]["content"]
    raw = extract_code(code)
    if raw is None:
        raw = code.strip()

    raw = fix_pubkeys_for_test(raw)

    tmp_file = TMP_DIR / f"skill_{idx}.ts"
    tmp_file.write_text(raw)

    try:
        async with sem:
            proc = await asyncio.create_subprocess_exec(
                "bun", "run", str(RUNNER), str(tmp_file), blockhash,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(VALIDATOR_DIR),
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.communicate()
                except:
                    pass
                return {"serializes": False, "error": "timeout", "category": "timeout"}

        result = None
        if stdout:
            for line in reversed(stdout.decode(errors="replace").strip().split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        result = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        pass

        if result is None:
            err = stderr.decode(errors="replace")[:300] if stderr else "no output"
            return {"serializes": False, "error": err, "category": categorize_error(err)}

        serialized = result.get("serialized_tx")
        if not serialized:
            err = result.get("error", "unknown")
            return {"serializes": False, "error": err, "category": categorize_error(err)}

        # Validate base64
        try:
            decoded = base64.b64decode(serialized)
            if len(decoded) > 1232:
                return {"serializes": False, "error": f"tx too large: {len(decoded)} > 1232", "category": "tx_too_large"}
            return {"serializes": True, "tx_size": len(decoded)}
        except Exception:
            return {"serializes": False, "error": "invalid base64", "category": "invalid_base64"}

    finally:
        try:
            tmp_file.unlink()
        except:
            pass


async def validate_file(input_path: Path, blockhash: str, concurrency: int, errors_only: bool) -> tuple[list, dict]:
    if not input_path.exists():
        return [], {}

    with open(input_path) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    if not entries:
        return [], {}

    print(f"\n{'='*70}", flush=True)
    print(f"  {input_path.name} ({len(entries)} examples)", flush=True)
    print(f"{'='*70}", flush=True)

    sem = asyncio.Semaphore(concurrency)
    start = time.time()
    validations = [None] * len(entries)
    fails_detail = []

    batch_size = 20
    for batch_start in range(0, len(entries), batch_size):
        batch_end = min(batch_start + batch_size, len(entries))
        tasks = [
            run_one(i, entries[i], blockhash, sem)
            for i in range(batch_start, batch_end)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            idx = batch_start + i
            if isinstance(result, Exception):
                validations[idx] = {"serializes": False, "error": str(result), "category": "exception"}
            else:
                validations[idx] = result

            # Collect failure details
            v = validations[idx]
            if not v["serializes"]:
                fails_detail.append((idx, entries[idx]["metadata"].get("source", "?"), v["error"][:150], v.get("category", "other")))

        done = batch_end
        valid = sum(1 for v in validations[:done] if v and v.get("serializes"))
        pct = valid / done * 100
        elapsed = time.time() - start

        if not errors_only:
            print(f"  {done:4d}/{len(entries)} | valid: {valid} ({pct:.0f}%) | {elapsed:.1f}s", flush=True)

    # Stats
    total_valid = sum(1 for v in validations if v["serializes"])
    error_cats = Counter()
    for v in validations:
        if not v["serializes"]:
            error_cats[v.get("category", "other")] += 1

    pct = total_valid / len(entries) * 100
    print(f"\n  Result: {total_valid}/{len(entries)} OK ({pct:.1f}%)", flush=True)

    if error_cats:
        print(f"  Errors:", flush=True)
        for cat, count in error_cats.most_common():
            print(f"    {count:4d}  {cat}", flush=True)

    # Show individual failures
    if fails_detail:
        print(f"\n  Failures:", flush=True)
        for idx, source, error, cat in fails_detail:
            print(f"    {idx:4d} [{source:25s}] ({cat:15s}) {error[:100]}", flush=True)

    # Annotate entries
    annotated = []
    for entry, v in zip(entries, validations):
        entry_copy = json.loads(json.dumps(entry))
        entry_copy["metadata"]["validation"] = v
        annotated.append(entry_copy)

    stats = {
        "total": len(entries),
        "valid": total_valid,
        "rate": round(pct, 1),
        "duration": round(time.time() - start, 1),
        "errors": dict(error_cats.most_common()),
    }

    return annotated, stats


async def main():
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", type=str, nargs="*")
    parser.add_argument("--concurrency", "-c", type=int, default=10)
    parser.add_argument("--errors-only", action="store_true")
    parser.add_argument("--output-valid", type=str, default=None)
    args = parser.parse_args()

    setup_validator()

    print("Connecting to Surfpool...", flush=True)
    try:
        blockhash = get_blockhash()
        print(f"Blockhash: {blockhash}", flush=True)
    except Exception as e:
        print(f"ERROR: Cannot connect to Surfpool at {RPC_URL}", flush=True)
        print(f"  Start: surfpool start -u https://api.mainnet-beta.solana.com --no-tui", flush=True)
        sys.exit(1)

    input_files = [Path(p) for p in args.input] if args.input else [p for p in DEFAULT_INPUTS if p.exists()]

    if not input_files:
        print("No input files found.", flush=True)
        sys.exit(1)

    print(f"Files: {len(input_files)} | Concurrency: {args.concurrency}", flush=True)

    all_annotated = []
    all_stats = {}

    for input_path in input_files:
        annotated, stats = await validate_file(input_path, blockhash, args.concurrency, args.errors_only)
        all_annotated.extend(annotated)
        all_stats[input_path.name] = stats
        try:
            blockhash = get_blockhash()
        except:
            pass

    # Summary
    total = sum(s["total"] for s in all_stats.values())
    valid = sum(s["valid"] for s in all_stats.values())

    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    for name, s in all_stats.items():
        print(f"  {name:35s} {s['valid']:4d}/{s['total']:4d} ({s['rate']}%) in {s['duration']}s", flush=True)
    print(f"  {'─'*50}", flush=True)
    print(f"  {'TOTAL':35s} {valid:4d}/{total:4d} ({valid/total*100:.1f}%)" if total else "  TOTAL: 0", flush=True)

    # Aggregate errors
    all_errors = Counter()
    for s in all_stats.values():
        for k, v in s.get("errors", {}).items():
            all_errors[k] += v
    if all_errors:
        print(f"\n  All errors:", flush=True)
        for cat, count in all_errors.most_common():
            print(f"    {count:4d}  {cat}", flush=True)

    # Save validated only
    valid_entries = [e for e in all_annotated if e["metadata"]["validation"]["serializes"]]
    out_path = Path(args.output_valid) if args.output_valid else OUTPUT_DIR / "validated_train.jsonl"
    with open(out_path, "w") as f:
        for entry in valid_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n  Validated: {out_path} ({len(valid_entries)} entries)", flush=True)

    # Save report
    report_path = BASE / "results" / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": all_stats,
            "total": total,
            "valid": valid,
            "rate": round(valid / total * 100, 1) if total else 0,
            "errors": dict(all_errors.most_common()),
        }, f, indent=2)
    print(f"  Report: {report_path}", flush=True)

    try:
        TMP_DIR.rmdir()
    except:
        pass


if __name__ == "__main__":
    asyncio.run(main())
