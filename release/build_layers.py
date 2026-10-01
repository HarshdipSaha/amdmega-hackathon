"""Build the layer tarballs that `crane append` puts on top of the mandated base (spec §10).

    python release/build_layers.py app  --out layers          # /app code, dirs, requirements.txt
    python release/build_layers.py deps --out layers          # /app/pylib from mc3/requirements.lock (cp314 wheels)
    python release/build_layers.py push-weights --out layers --repo Qwen/Qwen3-VL-8B-Instruct --rev <sha> \
        --slot reader --image <registry/repo:wip-tag>         # /models/reader, one <= 4 GiB group at a time

push-weights downloads ONE group of files, tars it, `crane append`s it and deletes it before the next group, so
the runner never holds more than one group plus its tar (spec Further Notes F). A single file larger than the
group limit (some safetensors shards are ~4.3-4.9 GB) becomes a layer of its own.
Tar entries are root-relative ("app/app.py"), owned by root, with fixed mtimes, so rebuilding gives identical layers.
"""
from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAYER_MAX = 4 * 2**30
MTIME = 1_790_000_000


def _info(name: str, size: int = 0, is_dir: bool = False) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = "root"
    ti.mtime = MTIME
    if is_dir:
        ti.type, ti.mode = tarfile.DIRTYPE, 0o755
    else:
        ti.size, ti.mode = size, 0o644
    return ti


def add_tree(tar: tarfile.TarFile, src: Path, dest: str) -> None:
    tar.addfile(_info(dest, is_dir=True))
    for p in sorted(src.rglob("*")):
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        name = f"{dest}/{p.relative_to(src).as_posix()}"
        if p.is_dir():
            tar.addfile(_info(name, is_dir=True))
        else:
            with open(p, "rb") as f:
                tar.addfile(_info(name, p.stat().st_size), f)


def build_app(out: Path) -> Path:
    path = out / "app.tar"
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
        tar.addfile(_info("app", is_dir=True))
        for d in ("app/corpus", "app/output", "app/index", "models"):
            tar.addfile(_info(d, is_dir=True))
        data = (ROOT / "mc3" / "app.py").read_bytes()
        tar.addfile(_info("app/app.py", len(data)), io.BytesIO(data))
        lock_path = ROOT / "mc3" / "requirements.lock"
        lock = (lock_path if lock_path.exists() else ROOT / "mc3" / "requirements.txt").read_bytes()
        tar.addfile(_info("app/requirements.txt", len(lock)), io.BytesIO(lock))
        add_tree(tar, ROOT / "sourcebound", "app/sourcebound")
    return path


def build_deps(out: Path) -> Path:
    stage = out / "stage-deps"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:", "--python-version", "3.14",
                    "--implementation", "cp", "--platform", "manylinux_2_28_x86_64", "--platform", "manylinux2014_x86_64",
                    "--target", str(stage), "-r", str(ROOT / "mc3" / "requirements.lock")], check=True)
    banned = [p.name for p in stage.iterdir() if p.name.split("-")[0].lower() in ("torch", "torchvision", "numpy", "pil", "pillow", "triton")]
    if banned:
        raise SystemExit(f"refusing to ship packages that shadow the base image: {banned}")
    path = out / "deps.tar"
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
        add_tree(tar, stage, "app/pylib")
    return path


def group_files(items: list[tuple[str, int]], limit: int = LAYER_MAX) -> list[list[str]]:
    """Pack (name, size) items, in name order, into groups whose total stays <= limit (a larger file goes alone)."""
    groups, cur, size = [], [], 0
    for name, s in sorted(items):
        if cur and size + s > limit:
            groups.append(cur)
            cur, size = [], 0
        cur.append(name)
        size += s
    if cur:
        groups.append(cur)
    return groups


def tar_group(out: Path, src: Path, names: list[str], slot: str, k: int) -> Path:
    """Tar the given files (relative to src) as models/<slot>/<name>."""
    path = out / f"weights-{slot}-{k:02d}.tar"
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
        tar.addfile(_info("models", is_dir=True))
        tar.addfile(_info(f"models/{slot}", is_dir=True))
        for rel in names:
            for parent in reversed(Path(rel).parents[:-1]):
                tar.addfile(_info(f"models/{slot}/{parent.as_posix()}", is_dir=True))
            p = src / rel
            with open(p, "rb") as f:
                tar.addfile(_info(f"models/{slot}/{rel}", p.stat().st_size), f)
    return path


def build_weights(out: Path, src: Path, slot: str) -> list[Path]:
    """Local variant (all files already on disk): one tar per group."""
    files = [(p.relative_to(src).as_posix(), p.stat().st_size) for p in src.rglob("*")
             if p.is_file() and ".cache" not in p.parts]
    return [tar_group(out, src, g, slot, k) for k, g in enumerate(group_files(files, LAYER_MAX))]


def push_weights(out: Path, repo: str, rev: str, slot: str, image: str) -> None:
    """Download, tar, crane-append and delete one group at a time (bounded runner disk)."""
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(repo, revision=rev, files_metadata=True)
    items = [(s.rfilename, s.size or 0) for s in info.siblings]
    stage = out / f"stage-{slot}"
    for k, group in enumerate(group_files(items, LAYER_MAX)):
        for name in group:
            hf_hub_download(repo, name, revision=rev, local_dir=stage)
        tar = tar_group(out, stage, group, slot, k)
        shutil.rmtree(stage)
        subprocess.run(["crane", "append", "-b", image, "-f", str(tar), "-t", image], check=True,
                       stdout=subprocess.DEVNULL)
        tar.unlink()
        print(f"{slot}: layer {k} appended ({len(group)} files)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["app", "deps", "push-weights"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo")
    ap.add_argument("--rev")
    ap.add_argument("--slot", choices=["reader", "embedder"])
    ap.add_argument("--image")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.what == "app":
        print(build_app(a.out))
    elif a.what == "deps":
        print(build_deps(a.out))
    else:
        push_weights(a.out, a.repo, a.rev, a.slot, a.image)


if __name__ == "__main__":
    main()
