"""Lazy Transformers producer using a fresh process-local PyTorch SDPA choice.

Observed backend attribution comes only from profiler operator evidence. A
requested context-manager setting is never treated as proof of execution.
"""
from __future__ import annotations

import contextlib
import io
import traceback
import time
from pathlib import Path
from typing import Iterable

from silentpath.matrix import Cell
from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord
from silentpath.telemetry import repeat_timed


def setup_gpu_device(torch):
    """Require and return PyTorch's ROCm/HIP device (exposed as cuda)."""
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch reports no CUDA/HIP GPU device")
    return torch.device("cuda")


def map_profiler_operators(operators: Iterable[str]) -> tuple[str | None, dict[str, str]]:
    """Map an unambiguous PyTorch SDPA operator set to a backend."""
    names = sorted({str(op) for op in operators if "scaled_dot_product" in str(op).lower()})
    found = set()
    for name in names:
        low = name.lower()
        if "flash_attention" in low or "flashattention" in low:
            found.add("FLASH_ATTENTION")
        elif "efficient_attention" in low or "efficientattention" in low:
            found.add("EFFICIENT_ATTENTION")
        elif "attention_math" in low or "math_attention" in low:
            found.add("MATH")
    mapped = next(iter(found)) if len(found) == 1 else None
    signals = {"profiler": " | ".join(names)} if mapped is not None else {}
    return mapped, signals


def parse_observed_backend(operators: Iterable[str]) -> str | None:
    """Pure mapping helper; unresolved or conflicting traces return None."""
    return map_profiler_operators(operators)[0]


_BACKENDS = {
    "FLASH_ATTENTION": "FLASH_ATTENTION",
    "SDPA_FLASH": "FLASH_ATTENTION",
    "MATH": "MATH",
    "SDPA_MATH": "MATH",
    "EFFICIENT_ATTENTION": "EFFICIENT_ATTENTION",
    "SDPA_EFFICIENT": "EFFICIENT_ATTENTION",
}


class SdpaProducer:
    def __init__(self, artifact_dir: Path | str | None = None) -> None:
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None

    def run(self, cell: Cell) -> RunRecord:
        operators: list[str] = []
        logs: list[str] = []
        trace_path: Path | None = None
        sample = CostSample(wall_seconds=0.0, samples=1)
        output = Output(text="")
        error = None
        ok = False
        requested = cell.backend
        choice = _BACKENDS.get(requested.upper())
        cell_started = time.perf_counter()
        stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
        stdout_redirect = contextlib.redirect_stdout(stdout_capture)
        stderr_redirect = contextlib.redirect_stderr(stderr_capture)
        stdout_redirect.__enter__()
        stderr_redirect.__enter__()
        try:
            if choice is None:
                raise ValueError(f"unsupported PyTorch SDPA backend: {requested}")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = setup_gpu_device(torch)
            seed = int(cell.config.get("seed", 0))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            from torch.nn.attention import SDPBackend, sdpa_kernel
            backend = getattr(SDPBackend, choice)
            model_name = cell.config["model"]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation="sdpa", torch_dtype="auto"
            ).to(device)
            model.eval()
            encoded = tokenizer(cell.prompt, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            generated = None

            @contextlib.contextmanager
            def selected_backend():
                with sdpa_kernel(backend):
                    yield

            # Warm up kernels before collecting timing and profiler evidence.
            with selected_backend():
                model.generate(**encoded,
                               max_new_tokens=int(cell.config.get("max_tokens", 64)),
                               do_sample=False)

            # The profiler wraps generation so the resulting operator list is
            # runtime evidence from the actual inference call.
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False,
            ) as prof:
                with selected_backend():
                    sample, generated = repeat_timed(
                        lambda: model.generate(
                            **encoded,
                            max_new_tokens=int(cell.config.get("max_tokens", 64)),
                            do_sample=False,
                        ),
                        repetitions=max(1, int(cell.config.get("repetitions", 1))),
                    )
            operators = [event.key for event in prof.key_averages()]
            prompt_len = encoded["input_ids"].shape[-1]
            token_ids = generated[0, prompt_len:].detach().cpu().tolist()
            output = Output(text=tokenizer.decode(token_ids, skip_special_tokens=True),
                             token_ids=token_ids)
            ok = True
            if self.artifact_dir:
                self.artifact_dir.mkdir(parents=True, exist_ok=True)
                trace_path = self._artifact_path(cell, "trace.json")
                try:
                    prof.export_chrome_trace(str(trace_path))
                except Exception as trace_exc:
                    logs.append(f"Could not export profiler trace: {trace_exc}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logs.append(traceback.format_exc())
        finally:
            stderr_redirect.__exit__(None, None, None)
            stdout_redirect.__exit__(None, None, None)
            if stdout_capture.getvalue():
                logs.insert(0, "STDOUT:\n" + stdout_capture.getvalue())
            if stderr_capture.getvalue():
                logs.insert(0, "STDERR:\n" + stderr_capture.getvalue())

        observed, signals = map_profiler_operators(operators)
        path = ObservedPath(requested=requested, observed=observed,
                            confidence=Confidence.REPORTED if observed else Confidence.UNKNOWN,
                            signals=signals)
        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            log_text = "\n".join(logs)
            if operators:
                log_text += ("\n" if log_text else "") + "PROFILER_OPERATORS:\n" + "\n".join(operators)
            (self._artifact_path(cell, "log.txt")).write_text(
                log_text or "No profiler operators captured.", encoding="utf-8")
        sample = sample.model_copy(update={"cell_wall_seconds": time.perf_counter() - cell_started})
        return RunRecord.build(config=cell.config, workload_id=cell.workload_id,
                               path=path, cost=sample, output=output, ok=ok, error=error)

    def _artifact_path(self, cell: Cell, suffix: str) -> Path:
        safe = f"{cell.backend}.{cell.workload_id}".replace("/", "_").replace("\\", "_")
        return self.artifact_dir / f"{safe}.{suffix}"
