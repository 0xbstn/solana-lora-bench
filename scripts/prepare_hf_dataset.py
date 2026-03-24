"""Prepare dataset for HuggingFace upload."""
import json
import random
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/hf_upload")

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def flatten_for_hf(rows):
    """Convert chat format to flat columns for HuggingFace."""
    flat = []
    for row in rows:
        msgs = row["messages"]
        meta = row.get("metadata", {})

        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        flat.append({
            "instruction": user,
            "response": assistant,
            "system_prompt": system,
            "source": meta.get("source", "unknown"),
            "type": meta.get("type", "unknown"),
            "protocol": meta.get("protocol", ""),
            "validated": bool(meta.get("validation", {}).get("serializes", False)),
            "messages": msgs,  # keep original chat format too
        })
    return flat

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all data
    base = load_jsonl(DATA_DIR / "dataset_base.jsonl")
    validated = load_jsonl(DATA_DIR / "validated_train.jsonl")
    fixes = load_jsonl(DATA_DIR / "benchmark_fixes.jsonl")

    # Merge and deduplicate
    all_rows = base + validated + fixes
    seen = set()
    unique = []
    for row in all_rows:
        key = json.dumps(row["messages"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(row)

    print(f"Total rows: {len(all_rows)}")
    print(f"After dedup: {len(unique)}")

    # Flatten
    flat = flatten_for_hf(unique)

    # Split train/test (90/10)
    random.seed(42)
    random.shuffle(flat)
    split = int(len(flat) * 0.9)
    train = flat[:split]
    test = flat[split:]

    print(f"Train: {len(train)}, Test: {len(test)}")

    # Stats
    sources = Counter(r["source"] for r in flat)
    types = Counter(r["type"] for r in flat)
    validated_count = sum(1 for r in flat if r["validated"])

    print(f"\nSources:")
    for s, c in sources.most_common(10):
        print(f"  {s}: {c}")
    print(f"\nTypes: {dict(types)}")
    print(f"Validated on Surfpool: {validated_count}/{len(flat)}")

    # Write JSONL (HuggingFace auto-converts to Parquet)
    for name, data in [("train", train), ("test", test)]:
        path = OUTPUT_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWritten {len(data)} rows to {path}")

    # Write stats
    stats = {
        "total": len(flat),
        "train": len(train),
        "test": len(test),
        "sources": dict(sources.most_common()),
        "types": dict(types),
        "validated_count": validated_count,
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
