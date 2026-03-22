"""
Parse Solana Bench trajectories into LoRA training data.

Extracts 3 types of training pairs:
1. SUCCESS: code that executed and earned rewards
2. ERROR_FIX: error message → corrected code that worked
3. PROGRESSIVE: multi-turn sequences showing iterative improvement

Output: JSONL files in data/processed/
"""

import json
import glob
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

TRAJECTORY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "solana-bench", "docs", "trajectory-viewer", "public", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def extract_code_block(content: str) -> str | None:
    """Extract TypeScript code block from assistant message."""
    pattern = r"```(?:typescript|ts)\s*\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    if matches:
        # Return the one with executeSkill if possible
        for match in matches:
            if "executeSkill" in match:
                return match.strip()
        return matches[0].strip()
    return None


def is_success(user_msg: str) -> bool:
    """Check if user feedback indicates success."""
    return "✅" in user_msg or "Transaction executed successfully" in user_msg


def is_error(user_msg: str) -> bool:
    """Check if user feedback indicates an error."""
    return "❌" in user_msg or "error" in user_msg.lower() or "failed" in user_msg.lower()


def extract_reward(user_msg: str) -> int:
    """Extract reward from success message."""
    match = re.search(r"Earned (\d+) reward", user_msg)
    if match:
        return int(match.group(1))
    return 0


def parse_conversation(conv_path: str, metrics_path: str) -> dict:
    """Parse a single conversation file into training data."""
    with open(conv_path) as f:
        messages = json.load(f)

    with open(metrics_path) as f:
        metrics = json.load(f)

    model = metrics.get("model", "unknown")
    final_reward = 0
    rewards = metrics.get("cumulative_rewards", [])
    if rewards:
        final_reward = rewards[-1]

    results = {
        "model": model,
        "final_reward": final_reward,
        "run_id": metrics.get("run_id", ""),
        "success_pairs": [],      # code that worked
        "error_fix_pairs": [],    # error → fix sequences
        "all_code_blocks": [],    # all code blocks for reference
    }

    system_prompt = None
    prev_error = None
    prev_code = None

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system_prompt = content
            continue

        if role == "assistant":
            code = extract_code_block(content)
            if code:
                results["all_code_blocks"].append({
                    "message_index": i,
                    "code": code,
                    "full_response": content,
                })
                prev_code = code

        elif role == "user" and i > 0:
            # Check if previous assistant message had code
            if prev_code:
                if is_success(content):
                    reward = extract_reward(content)
                    pair = {
                        "code": prev_code,
                        "reward": reward,
                        "message_index": i - 1,
                        "feedback": content[:500],
                    }

                    # If this was a fix for a previous error, record the error→fix pair
                    if prev_error:
                        results["error_fix_pairs"].append({
                            "error": prev_error,
                            "fix_code": prev_code,
                            "reward": reward,
                        })
                        prev_error = None

                    results["success_pairs"].append(pair)

                elif is_error(content):
                    prev_error = content[:500]
                else:
                    prev_error = None

            prev_code = None

    return results


def format_for_lora(results: list, system_prompt: str = None) -> list:
    """Format parsed results into JSONL training format."""
    training_data = []

    # Default system prompt for Solana coding
    default_system = (
        "You are an expert Solana developer. Write TypeScript code using @solana/web3.js "
        "to build and execute Solana transactions. Always use the function signature: "
        "export async function executeSkill(blockhash: string): Promise<string>. "
        "Return base64 encoded serialized transactions."
    )

    for result in results:
        # Type 1: Successful code blocks
        for pair in result["success_pairs"]:
            if pair["reward"] > 0:
                training_data.append({
                    "messages": [
                        {"role": "system", "content": default_system},
                        {"role": "user", "content": "Write TypeScript code to create a Solana transaction that explores new programs and instructions for maximum rewards."},
                        {"role": "assistant", "content": f"```typescript\n{pair['code']}\n```"},
                    ],
                    "metadata": {
                        "type": "success",
                        "reward": pair["reward"],
                        "model": result["model"],
                        "run_id": result["run_id"],
                    }
                })

        # Type 2: Error → Fix sequences
        for pair in result["error_fix_pairs"]:
            if pair["reward"] > 0:
                training_data.append({
                    "messages": [
                        {"role": "system", "content": default_system},
                        {"role": "user", "content": f"The previous transaction failed with this error:\n{pair['error']}\n\nFix the code to make it work."},
                        {"role": "assistant", "content": f"```typescript\n{pair['fix_code']}\n```"},
                    ],
                    "metadata": {
                        "type": "error_fix",
                        "reward": pair["reward"],
                        "model": result["model"],
                        "run_id": result["run_id"],
                    }
                })

    return training_data


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    all_results = []
    stats = defaultdict(int)

    # Process basic trajectories
    basic_dir = os.path.join(TRAJECTORY_DIR, "basic", "runs")
    swap_dir = os.path.join(TRAJECTORY_DIR, "swap", "runs")

    for env_name, env_dir in [("basic", basic_dir), ("swap", swap_dir)]:
        conv_files = sorted(glob.glob(os.path.join(env_dir, "*_conversation.json")))
        print(f"\n{'='*60}")
        print(f"Processing {env_name}: {len(conv_files)} conversations")
        print(f"{'='*60}")

        for conv_path in conv_files:
            run_id = os.path.basename(conv_path).replace("_conversation.json", "")
            metrics_path = conv_path.replace("_conversation.json", "_metrics.json")

            if not os.path.exists(metrics_path):
                print(f"  ⚠️  Missing metrics for {run_id}, skipping")
                continue

            result = parse_conversation(conv_path, metrics_path)
            all_results.append(result)

            success_count = len(result["success_pairs"])
            error_fix_count = len(result["error_fix_pairs"])
            total_code = len(result["all_code_blocks"])

            stats["total_runs"] += 1
            stats["total_success_pairs"] += success_count
            stats["total_error_fix_pairs"] += error_fix_count
            stats["total_code_blocks"] += total_code
            stats[f"{env_name}_runs"] += 1

            if result["final_reward"] > 0:
                stats["runs_with_reward"] += 1

            print(f"  {run_id} | {result['model']:<40} | reward={result['final_reward']:>3} | success={success_count:>2} | error_fix={error_fix_count:>2} | code_blocks={total_code:>2}")

    # Format for LoRA
    training_data = format_for_lora(all_results)

    # Split into train/valid
    # Use high-reward examples for training, lower for validation
    training_data.sort(key=lambda x: x["metadata"]["reward"], reverse=True)

    split_idx = int(len(training_data) * 0.9)
    train_data = training_data[:split_idx]
    valid_data = training_data[split_idx:]

    # Save
    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    valid_path = os.path.join(OUTPUT_DIR, "valid.jsonl")

    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(valid_path, "w") as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save raw parsed results
    raw_path = os.path.join(RAW_DIR, "parsed_trajectories.json")
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save stats
    stats_path = os.path.join(OUTPUT_DIR, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(dict(stats), f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total runs parsed: {stats['total_runs']}")
    print(f"  Basic: {stats.get('basic_runs', 0)}")
    print(f"  Swap: {stats.get('swap_runs', 0)}")
    print(f"Runs with reward > 0: {stats['runs_with_reward']}")
    print(f"Total code blocks found: {stats['total_code_blocks']}")
    print(f"Success pairs (code + reward): {stats['total_success_pairs']}")
    print(f"Error→Fix pairs: {stats['total_error_fix_pairs']}")
    print(f"")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(valid_data)}")
    print(f"")
    print(f"Files saved:")
    print(f"  {train_path}")
    print(f"  {valid_path}")
    print(f"  {raw_path}")
    print(f"  {stats_path}")


if __name__ == "__main__":
    main()
