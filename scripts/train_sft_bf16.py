"""
SFT Training for Qwen3.5-35B-A3B (MoE) on Solana Bench dataset.

CRITICAL:
- BF16 LoRA only (NOT QLoRA — BitsAndBytes doesn't support MoE nn.Parameter)
- gate_up_proj FUSED (NOT separate gate_proj/up_proj)
- r=16, alpha=32 (Unsloth recommended for MoE)
- VRAM: ~74 GB → fits on 1x H200

Usage:
    python3 train_sft_bf16.py [--dataset /root/dataset_base.jsonl] [--output /root/lora_output] [--merge /root/merged_model]
"""

import argparse
import json
import os

# Auto-detect best MoE backend for H200
# grouped_mm = H100+, unsloth_triton = A100, native_torch = fallback
# Let Unsloth auto-detect (will pick grouped_mm on H200)

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


def load_jsonl_dataset(path: str) -> Dataset:
    """Load JSONL dataset with messages format."""
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            # Keep only the messages field (what SFTTrainer needs)
            rows.append({"messages": row["messages"]})
    print(f"Loaded {len(rows)} examples from {path}")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/root/dataset_base.jsonl")
    parser.add_argument("--output", default="/root/lora_output")
    parser.add_argument("--merge", default="/root/merged_model")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--no-merge", action="store_true", help="Skip merge step")
    args = parser.parse_args()

    max_seq_length = args.max_seq_length

    # =========================================================================
    # 1. Load model — BF16, NOT QLoRA
    # =========================================================================
    print("=" * 60)
    print("Loading Qwen3.5-35B-A3B in BF16 (NOT QLoRA)...")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3.5-35B-A3B",
        max_seq_length=max_seq_length,
        load_in_4bit=False,      # NO QLoRA for MoE!
        load_in_16bit=True,      # BF16 LoRA
        # gpu_memory_utilization=0.9,  # for multi-GPU if needed
    )

    # =========================================================================
    # 2. Apply LoRA — fused gate_up_proj for MoE
    # =========================================================================
    print("Applying LoRA adapter (r={}, alpha={})...".format(args.rank, args.rank * 2))

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_up_proj", "down_proj",   # FUSED — NOT separate gate_proj/up_proj
        ],
        lora_alpha=args.rank * 2,          # *2 speeds up convergence
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    # Print trainable params
    model.print_trainable_parameters()

    # =========================================================================
    # 3. Load dataset
    # =========================================================================
    dataset = load_jsonl_dataset(args.dataset)

    # Split 90/10 for train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # =========================================================================
    # 4. Training config
    # =========================================================================
    training_args = SFTConfig(
        # Output
        output_dir=args.output,
        # Batch
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Schedule
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # Optimizer
        optim="adamw_8bit",
        weight_decay=0.01,
        # Precision
        fp16=False,
        bf16=True,
        # Sequence
        max_seq_length=max_seq_length,
        dataset_num_proc=4,
        packing=True,  # Pack multiple examples per sequence for efficiency
        # Logging
        logging_steps=1,
        logging_first_step=True,
        # Eval
        eval_strategy="steps",
        eval_steps=50,
        # Save
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        # Misc
        seed=3407,
        report_to="none",
    )

    # =========================================================================
    # 5. Train
    # =========================================================================
    print("=" * 60)
    print("Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max seq length: {max_seq_length}")
    print("=" * 60)

    # Qwen3.5 is a VLM — from_pretrained returns a Processor, not a Tokenizer.
    # Processor.apply_chat_template expects multimodal format (content as list of dicts).
    # We need the underlying tokenizer for text-only chat template.
    real_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    def formatting_func(example):
        """Format messages using the underlying tokenizer's chat template.
        Must always return a list of strings (Unsloth requirement)."""
        messages = example["messages"]
        # Handle both single example (list of dicts) and batch (list of lists)
        if isinstance(messages[0], dict):
            # Single example — wrap in list
            text = real_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return [text]
        else:
            # Batch
            return [
                real_tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                for msgs in messages
            ]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        formatting_func=formatting_func,
    )

    trainer.train()

    # Final eval
    eval_results = trainer.evaluate()
    print(f"\nFinal eval results: {eval_results}")

    # =========================================================================
    # 6. Save LoRA adapter
    # =========================================================================
    print(f"\nSaving LoRA adapter to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # =========================================================================
    # 7. Merge to full model (for vLLM serving)
    # =========================================================================
    if not args.no_merge:
        print(f"\nMerging LoRA into full model at {args.merge}...")
        model.save_pretrained_merged(
            args.merge,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"Merged model saved to {args.merge}")
        print("\nTo serve with vLLM:")
        print(f"  python3 -m vllm.entrypoints.openai.api_server \\")
        print(f"    --model {args.merge} \\")
        print(f"    --port 8000 \\")
        print(f"    --max-model-len 262144 \\")
        print(f"    --enforce-eager \\")
        print(f"    --gpu-memory-utilization 0.95 \\")
        print(f"    --language-model-only \\")
        print(f"    --tensor-parallel-size 2 \\")
        print(f"    --reasoning-parser qwen3")
    else:
        print("Skipping merge (--no-merge). Adapter saved only.")

    print("\nDone!")


if __name__ == "__main__":
    main()
