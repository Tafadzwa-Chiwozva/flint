#!/usr/bin/env python3
"""Compare Flint logits with Hugging Face transformers.

Torch is loaded only for the dump step, then the process exits before Flint
loads the same weights. That keeps the two f32 copies from residing together.

    python scripts/compare_hf.py dump --model-dir models/Qwen2.5-0.5B-Instruct --out /tmp/hf_logits.json
    cargo run --release -p flint-server -- logits --model-dir models/Qwen2.5-0.5B-Instruct \\
        --prompt "Hello, my name is" --out /tmp/flint_logits.json
    python scripts/compare_hf.py compare --hf /tmp/hf_logits.json --flint /tmp/flint_logits.json

`dump` records eager float32 last-token logits, the final hidden state, and a
16-token greedy continuation. `compare` prints max absolute error and checks
that the argmax token matches.
"""

from __future__ import annotations

import argparse
import json
import sys

PROMPTS = (
    "Hello, my name is",
    "The capital of France is",
    "Rust is a systems programming language focused on safety and speed.",
)


def dump(model_dir: str, out: str, prompts: list[str]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    # Transformers 5 reads this flag when picking eager vs SDPA. Force eager
    # so the comparison is the reference math, not a fused kernel.
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    rows = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        ids = encoded["input_ids"]
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True, use_cache=True)
        prompt_logits = output.logits[0, -1].float().cpu()
        # Qwen2Model applies the final RMSNorm before appending the last hidden state.
        hidden = output.hidden_states[-1][0, -1].float().cpu()
        past = output.past_key_values
        greedy = []
        logits = prompt_logits
        for _ in range(16):
            next_id = int(logits.argmax())
            greedy.append(next_id)
            step = torch.tensor([[next_id]], dtype=torch.long)
            with torch.no_grad():
                output = model(step, past_key_values=past, use_cache=True)
            logits = output.logits[0, -1].float().cpu()
            past = output.past_key_values
        rows.append(
            {
                "prompt": prompt,
                "token_ids": ids[0].tolist(),
                "logits": prompt_logits.tolist(),
                "last_hidden": hidden.tolist(),
                "greedy_ids": greedy,
                "greedy_text": tokenizer.decode(greedy, skip_special_tokens=True),
            }
        )
        print(
            f"dumped {prompt!r} tokens={ids.shape[1]} greedy={rows[-1]['greedy_text']!r}",
            file=sys.stderr,
        )
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)
    print(out)


def compare(hf_path: str, flint_path: str, tol: float) -> int:
    with open(hf_path, encoding="utf-8") as handle:
        hf_rows = json.load(handle)
    with open(flint_path, encoding="utf-8") as handle:
        flint_rows = json.load(handle)
    by_prompt = {row["prompt"]: row for row in flint_rows}
    failed = 0
    for row in hf_rows:
        flint = by_prompt.get(row["prompt"])
        if flint is None:
            print(f"missing flint row for {row['prompt']!r}")
            failed += 1
            continue
        if flint["token_ids"] != row["token_ids"]:
            print(f"{row['prompt']!r}: token ids differ")
            print("  hf   ", row["token_ids"])
            print("  flint", flint["token_ids"])
            failed += 1
            continue
        hf_logits = row["logits"]
        our_logits = flint["logits"]
        if len(hf_logits) != len(our_logits):
            print(f"{row['prompt']!r}: logit width {len(our_logits)} != {len(hf_logits)}")
            failed += 1
            continue
        max_abs = max(abs(a - b) for a, b in zip(hf_logits, our_logits))
        hf_top = max(range(len(hf_logits)), key=hf_logits.__getitem__)
        our_top = flint.get("argmax", max(range(len(our_logits)), key=our_logits.__getitem__))
        hidden_abs = None
        if row.get("last_hidden") and flint.get("last_hidden"):
            hidden_abs = max(
                abs(a - b) for a, b in zip(row["last_hidden"], flint["last_hidden"])
            )
        print(
            f"{row['prompt']!r}: max_abs_logit={max_abs:.6f} "
            f"hidden={hidden_abs} hf_top={hf_top} flint_top={our_top}"
        )
        if our_top != hf_top or max_abs > tol:
            failed += 1
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    dump_p = sub.add_parser("dump")
    dump_p.add_argument("--model-dir", required=True)
    dump_p.add_argument("--out", required=True)
    dump_p.add_argument("--prompt", action="append")
    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("--hf", required=True)
    cmp_p.add_argument("--flint", required=True)
    cmp_p.add_argument("--tol", type=float, default=0.5)
    args = parser.parse_args()
    if args.cmd == "dump":
        dump(args.model_dir, args.out, args.prompt or list(PROMPTS))
        return 0
    return 1 if compare(args.hf, args.flint, args.tol) else 0


if __name__ == "__main__":
    raise SystemExit(main())
