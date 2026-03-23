#!/usr/bin/env python3
"""
Generate targeted synthetic data for the 6 most common error patterns.
Async parallel — runs alongside generate_synthetic.py.

Output: data/processed/synthetic_targeted.jsonl
"""

import asyncio
import json
import os
import re
import hashlib
from openai import AsyncOpenAI

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FILE = os.path.join(BASE, "data/processed/clean_train.jsonl")
OUTPUT_FILE = os.path.join(BASE, "data/processed/synthetic_targeted.jsonl")

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
CONCURRENCY = 5  # parallel API calls

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


async def save(entry):
    global total_generated
    async with file_lock:
        with open(OUTPUT_FILE, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        total_generated += 1


# =============================================================================
# Targeted prompts based on real benchmark errors
# =============================================================================

TARGETED_TASKS = [
    # --- 1. Account writable (380 errors — biggest problem) ---
    # 50 tasks
    *[f"Write a Solana transaction that {op}. CRITICAL: ensure ALL accounts that are modified are marked as isWritable: true in AccountMeta. The fee payer must also be writable and a signer."
      for op in [
          "transfers 1 SOL from wallet to a new address",
          "creates a new SPL token mint and mints 1000 tokens",
          "creates an associated token account and transfers tokens into it",
          "delegates stake to a validator",
          "closes a token account and recovers rent",
          "burns tokens from a token account",
          "approves a delegate for 500 tokens on a token account",
          "creates a nonce account with 0.01 SOL",
          "splits a stake account into two",
          "freezes then thaws a token account in one transaction",
          "transfers SOL to 3 different recipients in one transaction",
          "creates a mint with Token-2022 and sets metadata",
          "withdraws from a stake account after deactivation",
          "revokes a token delegate approval",
          "creates account with seed and assigns to a program",
          "transfers 0.1 SOL and then 0.2 SOL to the same recipient",
          "creates two token mints in one transaction",
          "mints tokens to an existing ATA",
          "transfers tokens then burns the remainder",
          "creates an ATA for a Token-2022 mint",
          "initializes a multisig account with 3 signers",
          "sets a new authority on a token mint",
          "creates a token account and immediately approves a delegate",
          "transfers SOL to 5 different wallets in one tx",
          "deactivates then withdraws from a stake account",
          "merges two stake accounts into one",
          "creates a nonce account and immediately advances it",
          "allocates space and assigns an account to a program",
          "creates a PDA-derived account and transfers SOL to it",
          "creates a token mint with freeze authority set",
          "transfers tokens between two ATAs owned by different wallets",
          "burns all tokens from an account and closes it",
          "creates an account, transfers SOL to it, and assigns it",
          "sets compute budget then does a SOL transfer",
          "creates a mint, ATA, and mints tokens — 3 operations",
          "thaws a frozen token account",
          "transfers SOL using a nonce instead of recent blockhash",
          "creates a stake account with 10 SOL and delegates to validator",
          "splits 5 SOL from a 10 SOL stake account",
          "withdraws rent from a closed stake account",
          "creates two ATAs for different mints in one tx",
          "approves delegate then transfers via delegate authority",
          "creates a token account with explicit owner",
          "transfers SOL to a PDA address",
          "creates an account with createAccountWithSeed",
          "initializes a mint then freezes the mint authority",
          "burns tokens using delegate authority",
          "closes a token account and sends rent to a different wallet",
          "creates 3 keypairs and creates accounts for each",
          "transfers tokens and adds compute budget priority fee",
      ]],

    # --- 2. Correct imports (79 errors) ---
    # 50 tasks
    *[f"Write a Solana transaction that {op}. You MUST import TransactionInstruction from '@solana/web3.js'. Do NOT use MemoProgram — it doesn't exist. For memos use: new TransactionInstruction({{keys: [], programId: new PublicKey('MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr'), data: Buffer.from('message')}})"
      for op in [
          "sends 0.5 SOL with a memo 'payment for services'",
          "creates a token mint and adds a memo instruction",
          "transfers tokens between ATAs with a memo",
          "does a system transfer with compute budget and memo",
          "creates a stake account with a descriptive memo",
          "burns tokens and logs the reason in a memo",
          "does 3 SOL transfers each with a different memo message",
          "creates a nonce account with a memo recording the purpose",
          "transfers SPL tokens and adds a memo with the invoice number",
          "combines SystemProgram.transfer + ComputeBudget + Memo in one tx",
          "sends 1 SOL with memo 'rent payment March 2026'",
          "creates a mint and adds memo 'new token launch'",
          "sends 0.01 SOL to 3 addresses each with memo 'airdrop'",
          "transfers 100 tokens with memo containing the recipient name",
          "delegates stake with memo 'staking epoch 500'",
          "burns 50 tokens with memo 'token buyback burn'",
          "creates ATA with memo 'account setup for user X'",
          "does compute budget + transfer + 2 memo instructions",
          "sends SOL with a JSON-formatted memo string",
          "creates a token account and adds memo 'initialized'",
          "transfers SOL and adds a memo with a transaction reference ID",
          "approves token delegate with memo 'delegation approved'",
          "closes token account with memo 'account cleanup'",
          "freezes token account with memo 'compliance hold'",
          "thaws token account with memo 'hold released'",
          "creates nonce with memo 'offline signing nonce'",
          "mints tokens with memo 'batch mint round 3'",
          "transfers SOL with memo containing a UUID",
          "splits stake with memo 'rebalancing stake'",
          "withdraws stake with memo 'unstaking rewards'",
          "creates 2 accounts each with different memo messages",
          "sends SOL with a 200-character memo message",
          "creates mint + ATA + mint tokens + memo — 4 instructions",
          "sends 5 SOL with memo 'invoice #12345'",
          "burns tokens with memo containing the burn amount as string",
          "transfers to PDA with memo 'program deposit'",
          "creates account with seed and adds memo 'seeded account'",
          "revokes delegate with memo 'authorization revoked'",
          "sends SOL with empty memo (Buffer.from(''))",
          "transfers tokens with memo 'swap settlement'",
      ]],

    # --- 3. Correct API usage (46 errors) ---
    # 50 tasks
    *[f"Write a Solana transaction. {constraint}"
      for constraint in [
          "Use PublicKey.toBytes() NOT .toBuffer() — toBuffer does not exist on PublicKey. Transfer 2 SOL.",
          "Use tx.partialSign(keypair) NOT keypair.sign() — Keypair has no .sign() method. Create an account with a generated keypair.",
          "Use SystemProgram.transfer() — there is no SystemProgram.transferWithSeed(). Transfer SOL to a derived address.",
          "ComputeBudgetProgram must be imported from '@solana/web3.js'. Set compute unit limit to 200000 and price to 1.",
          "Import Keypair from '@solana/web3.js'. Generate a keypair, create an account for it, and transfer SOL to it.",
          "Connection is NOT needed for building transactions offline. Do NOT import Connection. Just build and serialize the tx.",
          "Use createAssociatedTokenAccountInstruction from '@solana/spl-token' — import it correctly. Create an ATA.",
          "PublicKey.generate() does not exist — use Keypair.generate().publicKey for random pubkeys. Create 3 accounts.",
          "Use LAMPORTS_PER_SOL from '@solana/web3.js' for SOL conversion — don't hardcode 1000000000. Transfer 5 SOL.",
          "Use TOKEN_PROGRAM_ID and ASSOCIATED_TOKEN_PROGRAM_ID from '@solana/spl-token'. Create ATA and mint tokens.",
          "Use PublicKey.toBytes() to get the byte representation. Derive a PDA using seeds with the wallet bytes.",
          "Use tx.partialSign(keypair) after setting blockhash and feePayer. Create a token mint with a generated keypair.",
          "SystemProgram.createAccount requires lamports, space, and programId. Create an account for SPL tokens.",
          "Import ComputeBudgetProgram from '@solana/web3.js'. Set unit limit to 400000 and unit price to 5000.",
          "Use Keypair.generate() — never new Keypair(). Generate 2 keypairs and create accounts for both.",
          "Do NOT use connection.getLatestBlockhash() — use the blockhash parameter. Transfer 0.5 SOL.",
          "Use getAssociatedTokenAddressSync from '@solana/spl-token' for synchronous ATA derivation. Transfer tokens.",
          "Use Keypair.fromSecretKey() NOT Keypair.fromSeed() for full 64-byte keys. Create an account.",
          "Use LAMPORTS_PER_SOL * 2 for 2 SOL — don't use BigInt or BN for lamport amounts in SystemProgram.transfer.",
          "Use TOKEN_2022_PROGRAM_ID from '@solana/spl-token' for Token-2022 operations. Create a Token-2022 mint.",
          "Use StakeProgram.createAccount() — it takes fromPubkey, stakePubkey, authorized, lamports. Create stake.",
          "Use StakeProgram.delegate() with stakePubkey, authorizedPubkey, votePubkey. Delegate stake.",
          "Use createInitializeMintInstruction from '@solana/spl-token' — not initializeMint. Create a 6-decimal mint.",
          "Use createMintToInstruction from '@solana/spl-token' — not mintTo() which needs Connection. Mint 1000 tokens.",
          "Use createTransferInstruction from '@solana/spl-token' for token transfers — not transfer(). Transfer 500 tokens.",
          "Use createBurnInstruction from '@solana/spl-token' — not burn(). Burn 100 tokens.",
          "Use createCloseAccountInstruction from '@solana/spl-token' — not closeAccount(). Close a token account.",
          "Use createApproveInstruction from '@solana/spl-token' — not approve(). Approve delegate for 200 tokens.",
          "Use createRevokeInstruction from '@solana/spl-token' — not revoke(). Revoke a delegate.",
          "Use createFreezeAccountInstruction and createThawAccountInstruction. Freeze then thaw.",
          "Use createSetAuthorityInstruction to change mint authority. Set new authority on a mint.",
          "Use SystemProgram.assign() — it takes accountPubkey and programId. Assign account to a program.",
          "Use SystemProgram.allocate() — it takes accountPubkey and space. Allocate 1024 bytes.",
          "Use SystemProgram.createAccountWithSeed() — it needs basePubkey, seed string, lamports, space, programId.",
          "Use SystemProgram.nonceAdvance() — it takes noncePubkey and authorizedPubkey. Advance a nonce.",
          "Use PublicKey.findProgramAddressSync([seeds], programId) — returns [pda, bump]. Derive a PDA.",
          "Use getAssociatedTokenAddress(mint, owner) — not getAssociatedTokenAccount. Get ATA address.",
          "Use MINT_SIZE from '@solana/spl-token' for mint account space — it's 82 bytes. Create a mint.",
          "Use createInitializeAccountInstruction from '@solana/spl-token' — not initializeAccount. Init token account.",
          "Transaction.serialize({requireAllSignatures: false, verifySignatures: false}) — both flags needed for offline.",
          "Use Buffer.from(string) for memo data — not new TextEncoder(). Add a memo to a transfer.",
          "Use new PublicKey(string) — PublicKey constructor takes base58 string. Reference a program ID.",
          "Use Keypair.generate().publicKey for random addresses — not PublicKey.unique() which doesn't exist.",
          "Import { Transaction } from '@solana/web3.js' — Transaction is a named export, not default.",
          "Use tx.add(instruction) to add instructions — not tx.instructions.push(). Add 3 instructions.",
          "Use StakeProgram.deactivate() then StakeProgram.withdraw() — they are separate operations.",
          "Use createSyncNativeInstruction from '@solana/spl-token' — not syncNative(). Sync wrapped SOL.",
          "Use getMint() only if you have Connection — for offline, use MINT_SIZE constant. Create mint offline.",
          "Use NONCE_ACCOUNT_LENGTH from '@solana/web3.js' for nonce account space. Create nonce account.",
          "Use SystemProgram.nonceWithdraw() — it takes noncePubkey, authorizedPubkey, toPubkey, lamports.",
      ]],

    # --- 4. Transaction size limit (75 errors) ---
    # 50 tasks
    *[f"Write a Solana transaction that {op}. IMPORTANT: keep total transaction size under 1232 bytes. Use at most 4-5 instructions. Do NOT add unnecessary operations."
      for op in [
          "transfers SOL to 3 recipients with a memo",
          "creates a token mint, creates an ATA, and mints tokens — 3 instructions max",
          "sets compute budget, transfers SOL, and adds a memo — exactly 3 instructions",
          "creates a stake account and delegates — 2 instructions only",
          "creates an account with seed and transfers SOL — 2 instructions",
          "does a token transfer with compute budget priority — 3 instructions",
          "burns tokens and closes the account — 2 instructions",
          "creates a nonce account — minimal instructions needed",
          "transfers tokens between 2 ATAs with compute budget — 3 instructions",
          "freezes a token account with compute budget — 2 instructions",
          "creates a mint + ATA + mints tokens — exactly 3 instructions",
          "transfers SOL to 2 recipients — 2 transfer instructions",
          "sets compute unit limit + price, then transfers SOL — 3 instructions",
          "creates account + assigns to program — 2 instructions",
          "burns tokens + memo — 2 instructions only",
          "creates ATA + transfers tokens into it — 2 instructions",
          "delegates stake + compute budget — 3 instructions",
          "transfers SOL + memo — 2 instructions only",
          "creates mint + sets authority — 2 instructions",
          "approves delegate + transfers tokens — 2 instructions",
          "closes token account + transfers remaining SOL — 2 instructions",
          "creates nonce + memo — 2 instructions",
          "transfers SOL to 4 recipients — keep under 1232 bytes",
          "creates account with seed + initializes as token account — 2 instructions",
          "sets compute budget + creates mint — 3 instructions",
          "transfers tokens + burns tokens — 2 instructions from same source",
          "creates 2 accounts — 2 createAccount instructions with minimal space",
          "freezes + thaws account — 2 instructions",
          "mints tokens + adds memo — 2 instructions",
          "revokes delegate + closes account — 2 instructions",
          "deactivates stake — 1 instruction only",
          "withdraws stake — 1 instruction only",
          "advances nonce + transfers SOL — 2 instructions",
          "creates ATA + memo — 2 instructions",
          "transfers tokens to 2 different ATAs — 2 instructions",
          "allocates space + assigns to program — 2 instructions",
          "initializes mint + creates ATA — 2 instructions",
          "creates mint + mints tokens (no ATA creation) — 2 instructions",
          "sets compute unit limit only — 1 instruction",
          "transfers SOL + compute budget + memo — exactly 3 instructions, no more",
          "creates account + transfers SOL to another address — 2 instructions",
          "burns tokens from 2 different accounts — 2 instructions",
          "mints to 2 different ATAs — 2 instructions",
          "approves 2 different delegates — 2 instructions",
          "creates PDA account + transfers SOL — 2 instructions",
          "token transfer + compute budget — 2 instructions",
          "creates account with exact minimum space — 1 instruction",
          "simple SOL transfer — 1 instruction, minimal tx size",
          "creates mint with freeze authority — 1 createAccount + 1 initMint",
          "transfers SOL to 3 recipients — exactly 3 instructions, verify size",
      ]],

    # --- 5. TypeScript correctness (251 errors) ---
    # 50 tasks
    *[f"Write clean, correct TypeScript for a Solana transaction that {op}. No syntax errors. All parentheses and braces must match. Use proper async/await. All variables must be declared with const/let."
      for op in [
          "transfers 1 SOL using SystemProgram.transfer",
          "creates a mint with createInitializeMintInstruction",
          "creates an ATA with getAssociatedTokenAddress and createAssociatedTokenAccountInstruction",
          "builds a VersionedTransaction with TransactionMessage.compile",
          "derives a PDA with PublicKey.findProgramAddressSync and uses it as an account",
          "creates multiple keypairs and uses partialSign for each",
          "uses BN from bn.js to encode a u64 amount in instruction data",
          "creates a token account with getMinimumBalanceForRentExemption hardcoded",
          "builds a transaction with StakeProgram.createAccount",
          "uses Buffer.alloc and writeUInt32LE for custom instruction data",
          "transfers SOL to a generated keypair and creates an account",
          "creates a mint, ATA, and mints tokens in one transaction",
          "uses PublicKey.findProgramAddressSync with multiple seeds",
          "builds a transaction with 3 SystemProgram.transfer instructions",
          "creates a stake account with Authorized and Lockup parameters",
          "uses createInitializeMintInstruction with freezeAuthority set to null",
          "transfers tokens using createTransferInstruction with amount as number",
          "burns tokens using createBurnInstruction",
          "closes an account using createCloseAccountInstruction",
          "approves a delegate using createApproveInstruction",
          "freezes an account using createFreezeAccountInstruction",
          "thaws an account using createThawAccountInstruction",
          "uses ComputeBudgetProgram.setComputeUnitLimit and setComputeUnitPrice",
          "creates a nonce account using SystemProgram.createNonceAccount",
          "advances a nonce using SystemProgram.nonceAdvance",
          "withdraws from nonce using SystemProgram.nonceWithdraw",
          "uses SystemProgram.allocate to allocate 512 bytes",
          "uses SystemProgram.assign to assign account to a program",
          "creates account with SystemProgram.createAccountWithSeed",
          "uses VersionedTransaction with MessageV0.compile",
          "transfers SOL with tx.serialize({requireAllSignatures: false})",
          "creates mint + ATA in one tx with proper import destructuring",
          "uses getAssociatedTokenAddressSync for synchronous ATA derivation",
          "builds TransactionInstruction with manual keys, programId, and data",
          "uses Buffer.from() to encode memo data",
          "creates 2 keypairs with Keypair.generate() and uses both",
          "uses LAMPORTS_PER_SOL constant for SOL amounts",
          "builds a transaction and returns base64 encoded string",
          "uses TOKEN_PROGRAM_ID from '@solana/spl-token'",
          "derives PDA with Buffer.from('seed') as seed",
          "creates a transaction with recentBlockhash from parameter",
          "uses tx.feePayer = walletPubkey before serialization",
          "builds tx with proper error-free TypeScript — no type assertions needed",
          "uses const for all variable declarations — no let or var",
          "creates mint with Keypair.generate() as mintKeypair",
          "sets up proper imports at the top: Transaction, PublicKey, SystemProgram, Keypair, LAMPORTS_PER_SOL",
          "uses async function with Promise<string> return type",
          "transfers tokens with proper u64 amount (no BigInt needed for small amounts)",
          "creates stake with StakeProgram.createAccount — proper Authorized import",
          "uses createAssociatedTokenAccountInstruction with all 4 params: payer, ata, owner, mint",
      ]],

    # --- 6. Rent exemption (59 errors) ---
    # 50 tasks
    *[f"Write a Solana transaction that {op}. For rent exemption, use a hardcoded safe value: const RENT_EXEMPT_MINIMUM = 1_461_600 for basic accounts (128 bytes), 2_039_280 for token accounts (165 bytes), 1_113_600 for mint accounts (82 bytes). Do NOT call connection.getMinimumBalanceForRentExemption — we're building offline."
      for op in [
          "creates a new account with SystemProgram.createAccount — allocate 128 bytes with correct rent",
          "creates a token mint account with MINT_SIZE (82 bytes) and correct lamports",
          "creates a token account with 165 bytes and correct rent-exempt lamports",
          "creates a stake account — stake accounts need 200 bytes, use 2_282_880 lamports",
          "creates a nonce account — nonce accounts need 80 bytes, use 1_447_680 lamports",
          "creates an account with seed and correct rent for 256 bytes",
          "creates multiple accounts in one tx, each with correct rent-exempt amounts",
          "creates a mint + token account in one tx with proper rent for both",
          "creates an account for a custom program with 1024 bytes of data space",
          "creates a token-2022 mint with extensions — allocate extra space for metadata",
          "creates an account with 64 bytes — use 1_238_400 lamports for rent",
          "creates an account with 0 bytes — use 890_880 lamports minimum",
          "creates a mint (82 bytes) and two token accounts (165 bytes each) in one tx",
          "creates a stake account with 200 bytes, fund with 2_282_880 + 5 SOL for staking",
          "creates a nonce account with 80 bytes, fund with 1_447_680 + extra for withdrawals",
          "creates an account with 512 bytes of data space for a custom program",
          "creates an account with 2048 bytes for storing large program state",
          "creates a token mint and immediately mints tokens — correct rent for mint",
          "creates two separate mints in one tx — 1_113_600 lamports each",
          "creates a token account and transfers tokens to it — correct 2_039_280 rent",
          "creates an account with seed for 128 bytes — 1_461_600 lamports",
          "creates 3 accounts: mint (82b) + 2 token accounts (165b each) with proper rents",
          "creates a stake account funded with exact rent + 1 SOL for delegation",
          "creates nonce account funded with exact rent for 80 bytes",
          "creates an account with 300 bytes for a custom data structure",
          "creates a mint with Token-2022 — needs extra 128 bytes for extensions (210 bytes total)",
          "creates an account with 100 bytes — calculate rent as ceil(128 * (100/128)) minimum",
          "creates 2 mints and 2 ATAs in one tx — 4 createAccount with proper rents",
          "creates a basic account (128 bytes) and transfers remaining SOL",
          "creates a program-owned account with 64 bytes of data",
          "creates a token mint then an ATA — correct rent for mint only (ATA is free via instruction)",
          "creates an account with 1 byte of data — minimum possible rent-exempt amount",
          "creates a stake account with authorized staker and withdrawer — 200 bytes rent",
          "creates nonce account and advances it — only nonce creation needs rent",
          "creates an account for storing a single public key (32 bytes)",
          "creates an account for storing 10 public keys (320 bytes)",
          "creates a mint with 9 decimals — same 82 bytes, same 1_113_600 rent",
          "creates a token account for a specific mint — 165 bytes, 2_039_280 lamports",
          "creates 2 stake accounts in one tx — each 200 bytes with proper rent",
          "creates an account with seed and 256 bytes — proper rent calculation",
          "creates a basic account and assigns it to a program — rent needed only for create",
          "creates a PDA-controlled account with 128 bytes — standard rent",
          "creates a frozen token account — same rent as regular token account (165 bytes)",
          "creates a multisig account — 355 bytes for 11-of-11 multisig",
          "creates a Token-2022 mint with permanent delegate — extra space needed",
          "creates a Token-2022 mint with transfer fee config — extra 108 bytes",
          "creates a Token-2022 mint with immutable owner — extra 1 byte",
          "creates an account with exactly 0 data and minimum rent-exempt lamports",
          "creates a token mint + funds it above rent so it can also pay fees",
          "creates 5 accounts of different sizes in one tx — each with correct rent",
      ]],
]


async def generate_one(task: str, idx: int):
    global total_errors
    try:
        text = await chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
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

        # Determine category from task content
        if "isWritable" in task:
            source = "targeted/writable"
        elif "MemoProgram" in task or "TransactionInstruction" in task:
            source = "targeted/imports"
        elif "toBuffer" in task or "partialSign" in task or "does not exist" in task:
            source = "targeted/api-usage"
        elif "1232 bytes" in task:
            source = "targeted/tx-size"
        elif "syntax" in task.lower() or "parentheses" in task.lower():
            source = "targeted/typescript"
        elif "rent" in task.lower() or "RENT_EXEMPT" in task:
            source = "targeted/rent"
        else:
            source = "targeted/other"

        await save({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
                {"role": "assistant", "content": wrapped},
            ],
            "metadata": {
                "type": "synthetic",
                "source": source,
                "model": MODEL,
                "protocol": "system",
            },
        })

    except Exception as e:
        total_errors += 1


async def main():
    global total_generated, total_errors

    # Load existing hashes to avoid dupes
    for path in [TRAIN_FILE, OUTPUT_FILE,
                 os.path.join(BASE, "data/processed/synthetic_distilabel.jsonl")]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        try:
                            e = json.loads(line)
                            seen.add(code_hash(e["messages"][2]["content"]))
                        except:
                            pass

    # Clear output
    open(OUTPUT_FILE, 'w').close()

    print(f"Existing hashes: {len(seen)}")
    print(f"Tasks to generate: {len(TARGETED_TASKS)}")
    print(f"Concurrency: {CONCURRENCY}")
    print()

    # Launch all tasks
    tasks = [generate_one(task, i) for i, task in enumerate(TARGETED_TASKS)]

    # Progress tracking
    done = 0
    batch_size = 10
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        await asyncio.gather(*batch)
        done += len(batch)
        print(f"  Progress: {done}/{len(tasks)} done, {total_generated} generated, {total_errors} errors")

    print(f"\n{'='*50}")
    print(f"TOTAL: {total_generated} generated, {total_errors} errors")
    print(f"Saved to {OUTPUT_FILE}")

    # Stats
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        sources = {}
        for e in entries:
            s = e["metadata"]["source"]
            sources[s] = sources.get(s, 0) + 1
        print(f"By source: {sources}")


if __name__ == "__main__":
    asyncio.run(main())
