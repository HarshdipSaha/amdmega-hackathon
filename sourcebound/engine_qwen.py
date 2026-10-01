"""GPU engine: one Qwen VLM (reader + transcriber) and one Qwen3 embedding model, BF16, resident (spec §8).

GPU only. Loaded from local directories with local_files_only=True; never touches the network.
"""
from __future__ import annotations

import os
import tempfile
import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

TRANSCRIBE = ("Transcribe all text printed in this image exactly as written, keeping every character, hyphen, "
              "underscore and # sign. Put each label on the same line as the thing it labels, for example "
              "'B14: THERM_ALERT#' for a pin and its signal, or 'BOARD REVISION: REV-C2' for a field and its value. "
              "Finish with one line that starts with 'IMAGE:' and says what the image shows. Output plain text only.")
QUERY_TASK = "Given a question about internal technical documents, retrieve the passage that answers it"
MAX_PIXELS = int(os.environ.get("SB_MAX_PIXELS", str(1600 * 1200)))


def load_image(path: str, max_pixels: int = MAX_PIXELS) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im.seek(0)
        im = im.convert("RGB")
    if im.width * im.height > max_pixels:
        s = (max_pixels / (im.width * im.height)) ** 0.5
        im = im.resize((max(28, int(im.width * s)), max(28, int(im.height * s))), Image.LANCZOS)
    return im


class QwenEngine:
    def __init__(self, reader: str, embedder: str, dtype=torch.bfloat16, attn: str = "sdpa"):
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm GPU not visible: refusing a silent CPU fallback")
        self.name = os.path.basename(os.path.realpath(reader.rstrip("/")))     # resolve the mc3-reader symlink
        self.proc = AutoProcessor.from_pretrained(reader, local_files_only=True)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            reader, dtype=dtype, attn_implementation=attn, local_files_only=True).to("cuda").eval()
        self.etok = AutoTokenizer.from_pretrained(embedder, padding_side="left", local_files_only=True)
        self.emb = AutoModel.from_pretrained(embedder, dtype=dtype, attn_implementation=attn,
                                             local_files_only=True).to("cuda").eval()
        self.lock = threading.Lock()

    def _inputs(self, prompts: list[str], images: list[list[str]]):
        texts, flat = [], []
        for prompt, imgs in zip(prompts, images):
            msgs = [{"role": "user", "content": [*({"type": "image"} for _ in imgs), {"type": "text", "text": prompt}]}]
            texts.append(self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                       enable_thinking=False))
            flat += [load_image(p) for p in imgs]
        return self.proc(text=texts, images=flat or None, padding=True, return_tensors="pt").to("cuda")

    @torch.inference_mode()
    def _generate(self, prompts, images, max_new_tokens) -> list[str]:
        with self.lock:
            inputs = self._inputs(prompts, images)
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)
            new = out[:, inputs["input_ids"].shape[1]:]
            return [self.proc.decode(row, skip_special_tokens=True).strip() for row in new]

    def generate(self, task, prompt, images=(), max_new_tokens=200, question=""):
        return self._generate([prompt], [list(images)], max_new_tokens)[0]

    def transcribe(self, paths, batch: int = int(os.environ.get("SB_OCR_BATCH", "4"))):
        out = []
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            out += self._generate([TRANSCRIBE] * len(chunk), [[p] for p in chunk], 384)
        return out

    @torch.inference_mode()
    def embed(self, texts, kind="document", batch: int = 32):
        if kind == "query":
            texts = [f"Instruct: {QUERY_TASK}\nQuery:{t}" for t in texts]
        rows = []
        with self.lock:
            for i in range(0, len(texts), batch):
                tok = self.etok(texts[i:i + batch], padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
                h = self.emb(**tok).last_hidden_state[:, -1]          # last-token pooling (left padding)
                rows.append(F.normalize(h.float(), dim=-1).cpu().numpy())
        return np.concatenate(rows) if rows else np.zeros((0, 1), dtype=np.float32)

    def warm_up(self):
        img = os.path.join(tempfile.gettempdir(), "sourcebound_warmup.png")
        Image.new("RGB", (256, 128), "white").save(img)
        self.transcribe([img])
        self.generate("read", "Reply with OK.", (), 4)
        self.embed(["warm up"], kind="query")

    def peak_gib(self) -> float:
        return round(torch.cuda.max_memory_reserved() / 2**30, 2)
