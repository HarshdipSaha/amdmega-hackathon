"""Transformers engine for Qwen3-VL / Qwen3.5 (BF16, greedy, no repetition penalty, thinking off)."""
import time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from .engine_fake import Read

class QwenEngine:
    def __init__(self, path: str, dtype=torch.bfloat16, attn: str = "sdpa"):
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm GPU not visible: refusing silent CPU fallback")
        self.name = path.rstrip("/").split("/")[-1]
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, attn_implementation=attn)
        self.model.to("cuda").eval()

    @torch.inference_mode()
    def read(self, image, instruction, max_new_tokens):
        t = time.monotonic()
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        inputs = self.proc(text=[prompt], images=[image], return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)
        new = out[0, inputs["input_ids"].shape[1]:]
        text = self.proc.decode(new, skip_special_tokens=True)
        return Read(text, truncated=len(new) >= max_new_tokens, seconds=time.monotonic() - t)

    def peak_gib(self) -> float:
        return round(torch.cuda.max_memory_reserved() / 2**30, 2)
