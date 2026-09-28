"""G1 runner for forced PyTorch SDPA paths on AMD ROCm.

One invocation runs one kernel path in a fresh process, preventing cached model
state from silently retaining a previous SDPA selection.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path


def supported_backend_names() -> tuple[str, ...]:
    """The explicit SDPA choices tested by the W7900D G1 experiment."""
    return ("FLASH_ATTENTION", "MATH", "EFFICIENT_ATTENTION")


def run(model_id: str, prompt_file: Path, backend_name: str, cache_dir: str | None = None) -> dict:
    """Generate once with an explicitly forced PyTorch SDPA backend."""
    import os
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")

    backend = getattr(SDPBackend, backend_name)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, attn_implementation="sdpa", cache_dir=cache_dir
    ).to("cuda").eval()
    messages = [{"role": "user", "content": prompt_file.read_text(encoding="utf-8")}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, return_tensors="pt").to("cuda")

    torch.cuda.synchronize()
    t_gen_start = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel(backend):
        generated = model.generate(
            **encoded, do_sample=False, max_new_tokens=64, return_dict_in_generate=True,
            output_scores=True, use_cache=True,
        )
    torch.cuda.synchronize()
    gen_seconds = time.perf_counter() - t_gen_start

    prefix = encoded.input_ids.shape[1]
    token_ids = generated.sequences[0, prefix:].tolist()
    logprobs = [
        float(torch.log_softmax(scores[0], dim=-1)[token_id].item())
        for scores, token_id in zip(generated.scores, token_ids)
    ]
    return {
        "backend_requested": backend_name,
        "backend_observed": f"PyTorch SDPBackend.{backend_name}",
        "model": model_id,
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "ok": True,
        "text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "token_ids": token_ids,
        "chosen_logprobs": logprobs,
        "generation_seconds": gen_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=supported_backend_names())
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    result = {"backend_requested": args.backend, "backend_observed": None, "ok": False,
              "text": "", "token_ids": [], "chosen_logprobs": []}
    try:
        result = run(args.model, args.prompt_file, args.backend, cache_dir=args.cache_dir)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("backend_requested", "ok", "error", "wall_seconds")}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
