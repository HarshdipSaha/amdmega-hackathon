# AMD runtime and packaging feasibility

Research date: **22 September 2026 (UTC)**. Sources and HTTP metadata are cached in [amd/](amd/). This is a feasibility audit, not a GPU benchmark: no notebook session was launched, no model weights or container layers were downloaded, and no application was implemented.

**Recommendation:** start with the supplied ROCm PyTorch installation and a native Transformers model, using GPU SDPA with an explicitly tested eager/math fallback. Keep the model in one resident process and expose the prescribed per-image CLI. Treat vLLM, custom attention kernels, quantization, and Qwen3.5's hybrid kernels as subsequent measured alternatives. An AMD-compatible model card alone does not establish compatibility with this exact image and the unknown allocated GPU.

## What the challenge actually requires

The PDF is the controlling source for these gates; [`COMPUTE.md`](../../COMPUTE.md) independently records the user's confirmed notebook access and session/storage limits. [B1, B2]

| Requirement | Consequence for the spec |
|---|---|
| Final image starts from `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`; lower-layer identity checked | Keep the complete base-layer prefix. Do not flatten, squash, or replace the final stage with a vLLM image. |
| At most **60 GiB uncompressed** | Gate at **64,424,509,440 bytes**; measure base plus dependencies plus shipped weights, including retained lower layers. Registry transfer size is a different quantity. |
| **1–48 GiB VRAM**, upper tolerance **1%**, peak sampled every **3 seconds** | Target a measured peak below 48 GiB, with operational headroom; do not rely on the tolerance. A small CPU OCR pipeline is insufficient. Resident, useful model weights should naturally clear the 1 GiB lower gate. |
| One GPU, possibly **CDNA or RDNA** | Discover the GPU and its `gfx` architecture before selecting attention kernels, precision or concurrency. Do not assume MI300X. |
| Startup **600 seconds**; each invocation **30 seconds**; all ten images after startup **600 seconds** | Include decoding, resizing, model calls, retries, IPC and output writing in each measured invocation. Warm model state must survive CLI process exits. |
| `python3 /app/app.py --input-image <path>` | Require no caller-side server request, extra CLI option, model download command or setup step. |
| `/app/output/<input-stem>_output.json` with a required string `text` | Write one UTF-8 JSON file atomically per call. Optional confidence is recorded but not scored. |
| PNG, JPEG and TIFF; no specified dimensions | Include actual decoding and orientation tests, compressed TIFF support, high-resolution inputs and unusual aspect ratios. |
| Publicly pullable final image; no secrets; do not advertise its reference in a public repo | Verify anonymous pullability when publishing, and keep the submission reference out of committed research/spec files. The required public **base** reference is not the submission reference. |

The PDF says the upper VRAM tolerance is 1%, yielding a mathematical boundary of 48.48 GiB, but **48 GiB is the engineering cap**. External AMD memory measurements are authoritative for this gate; PyTorch allocated/reserved memory is supplementary and omits some runtime allocations. [B1; A6]

## Exact base: available, with size still unmeasured

The tag in the PDF is **real and publicly resolvable**; it must not be described as a typo or silently replaced. [A1–A3]

| Observation | Evidence captured on 2026-09-22 |
|---|---|
| Docker Hub exact-tag request | HTTP **200**, 12:31:58 UTC |
| Docker registry manifest request | Initial unauthenticated **401**, anonymous token exchange **200**, authenticated anonymous pull request **200**, 12:32:00 UTC |
| Manifest digest | `sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c` |
| Platform | `linux/amd64` |
| Image config digest | `sha256:75287c3f4d5eba32d8b91797639a20df134e8d1a8167e9abed77d78f330fd0f3`; SHA-256 of captured config bytes matches |
| Base layer identity | **11** ordered `rootfs.diff_ids`, saved in the captured config |
| Config labels | Python **3.14**, ROCm **10.0.0**, Ubuntu **26.04**, AMD family `device-all` |
| Initial command | `CMD ["/bin/bash"]`, no entrypoint; provide an application lifecycle that remains running |
| Compressed layer sum | **20,492,259,842 bytes**; every captured layer has gzip media type |

The compressed sum is **not** the base's uncompressed size or the available room for weights. Pulling/unpacking the base on a suitable build machine is still required. A read-only local check found Docker CLI **28.3.2**, context `desktop-linux`, but no reachable Docker Desktop Linux engine. It was not started. [A3; local evidence in `amd/local-docker-status.json`]

Pin the observed base digest for a reproducible build and recheck the organizer's required tag before submission: tags can change. Preserve the same ordered lower layers; adding application layers does not preserve the base's whole-image config hash. [A3, A10]

### Correct packaging verification

Save `docker image inspect` JSON for both the pulled base and the final `linux/amd64` image. Assert that **the entire ordered base `RootFS.Layers` array is a prefix of the candidate's array**. Docker's image specification defines these as hashes of uncompressed layers, ordered bottom to top; a Dockerfile name or a visually plausible `docker history` listing is not equivalent evidence. [A10, A11]

For the classic Docker image store, inspect `.Size` as the machine-readable total image size and compare bytes with `60 * 1024**3`. Keep `docker history --no-trunc` as an additional diagnostic. On a modern containerd image store, also record the selected manifest's **`ImageData.Size.Unpacked`**, which the Engine API expressly defines as uncompressed content; `Size.Total` can include **both** compressed content and unpacked snapshots. Record the Docker version/storage backend and do not confuse shared/unique host disk usage, compressed blobs, or total disk consumption with the grading quantity. Leave sufficient margin if measurement methods differ. [A9–A12]

Deleting a large file in a later layer does not remove its bytes from earlier layers. Install and clean temporary artifacts in the same application layer; preserve all mandated base layers. A 25 GB persistent notebook volume is not a place to keep a complete unpacked base, a second image archive and multiple model caches. [B2; A12]

## Runtime options and their actual evidence

| Option | Primary-source finding | Decision |
|---|---|---|
| Native PyTorch + Transformers | ROCm PyTorch deliberately uses `torch.cuda` and `device="cuda"`; `"rocm"`/`"hip"` are not device strings. Transformers exposes `attn_implementation="sdpa"` and `"eager"`, including separate multimodal backbones. [A6–A8] | First path to test. Preserve the supplied PyTorch installation and pin compatible Python dependencies. |
| Built-in SDPA | PyTorch supplies a math implementation and selects among supported kernels. Its ROCm backend explicitly exposes AOTriton/CK Flash Attention selection; built-in SDPA does **not** imply installing the separate `flash-attn` package. [A7, A8] | Let tested built-in kernels operate first. Try math/eager when needed, but benchmark their memory and latency at the selected resolution. |
| vLLM | Current public docs list selected MI200/300/350, Radeon RX 7900/9000 and Ryzen AI families, not every AMD device. Docs warn that ROCm wheels are coupled to PyTorch/ROCm builds. [A5] | Optional only after the exact GPU, model, Python and ABI combination passes. “vLLM supports AMD” is too broad a compatibility claim. |
| Current vLLM ROCm wheel | The release index links `vllm-0.30.0+rocm723-cp312-cp312-manylinux_2_39_x86_64.whl`. The v0.30.0 ROCm recipe uses ROCm **7.2.3**, Python **3.12**, and a PyTorch **2.12** branch. [A13, A14] | This is not a verified drop-in wheel for ROCm 10.0 / Python 3.14 / supplied PyTorch 2.13. Source builds add dependencies, build time and size risk. |
| vLLM Python-version conflict | Its public installation page says Python **3.10–3.13**, while live v0.30.0 PyPI metadata says `>=3.10,<3.15` and pins `torch==2.13.0` in the public package. That public package also declares NVIDIA-specific dependencies. [A5, A15] | Do not call Python 3.14 categorically unsupported. Do not run unqualified `pip install vllm` and assume it selects the needed ROCm build. Treat exact-stack support as unverified. |
| Standalone FlashAttention | The live upstream README describes both CK and Triton AMD backends and now lists RDNA 3/4 as well as Instinct support for CK. It still has backend/version/input requirements. [A16] | Avoid stale claims that FlashAttention is universally CUDA-only or CK universally excludes Radeon; still do not make it a first-day dependency. |
| Qwen3.5 hybrid path | Transformers **v5.17.0** contains pure-PyTorch `causal_conv1d_fn/update`, `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule`, with optional kernel decorators. `Qwen3_5GatedDeltaNet` is distinct from ordinary full attention. [A17] | Optional FLA/causal-conv packages are not unconditional requirements at this version. Setting SDPA does not validate the hybrid blocks; test fallback correctness and worst-case latency separately. |

Use an ordinary Qwen3-VL-style attention architecture as the initial general VLM candidate if supported by the selected checkpoint and pinned Transformers release. Compare the small OCR specialist independently. Weight-file sizes are only storage/parameter evidence, not measured peak VRAM; vision activations, caches and runtime buffers still count. Avoid introducing quantization solely to fit a model that already fits, until the unquantized path is measured. These are engineering recommendations, not benchmark findings.

### Why the linked AMD notebook is illustrative

The exact tutorial linked in the PDF was developed on **MI300X, Ubuntu 22.04, ROCm 6.2/6.3**. It launches **`rocm/vllm-dev:main`**, uses gated **Llama-3.2-11B-Vision-Instruct**, builds a Gradio interface, and its example generation uses `temperature=0.7`, `max_tokens=512`, `max_num_seqs=16`. [A4]

Those choices illustrate VLM OCR but do not satisfy this challenge's final-base identity, prescribed CLI, registration-only transcription, deterministic short answer, or proven RDNA performance. The model-choice menu is not evidence that every listed model passes the same runtime on both GPU families. Use the notebook to understand ROCm device exposure and multimodal inference, then validate the chosen model on the mandated image. No token is needed for an ungated public model, and no HF token belongs in a public image. [B1; A4]

## Load once while keeping the prescribed CLI

**Document ambiguity:** the PDF explicitly says “There is no service to run and no endpoint to expose,” starts a new script for each image inside an already-running container, and also says “Load your model once, not once per image.” It does not describe how startup readiness is signalled or whether the harness overrides the image entrypoint. Docker documents `exec` as launching a new command inside the running container; a Python module-level singleton cannot survive across those separate invocations. [B1; A18]

Proposed internal design, pending validation of the harness lifecycle:

1. The image entrypoint starts one foreground worker, loads the model once onto the GPU, performs a bounded warm-up, and publishes readiness only after a real inference succeeds. The worker owns all GPU objects. It remains the container's main process or has a signal-forwarding supervisor.
2. `/app/app.py --input-image ...` remains the sole external interface. A lightweight standard-library client sends a request ID, the file path and deadline over an **AF_UNIX socket inside the container**. No HTTP server, listening TCP port or caller-visible API is introduced. Python supports Unix-domain sockets. [A19]
3. The worker serializes requests, decodes/preprocesses the image, performs bounded inference and returns structured text. Every request's deadline includes queueing, decode and all fallback passes. Keep any additional view/model call inside that deadline.
4. The CLI validates the response and atomically writes `/app/output/<stem>_output.json` in UTF-8. Give each request a unique temporary filename, then replace the final file. Do not return a stale output after worker failure or timeout.
5. Use a lock and worker-liveness/readiness checks so concurrent CLI invocations cannot allocate duplicate models. A dead or hung worker must fail visibly; starting a replacement while the original still owns VRAM can breach the cap. Cold direct inference is a separately tested fallback only when no live worker exists and the entire load-plus-inference call fits the same 30-second limit.

Add a Docker healthcheck and a readiness record for local validation, but do **not** assume a healthcheck forces the organizer to wait. Docker health status is additional to the normal running state. The spec must record this unconfirmed assumption and test the organizer's actual startup method before submission. If entrypoints are overridden or internal workers are forbidden, either a genuinely measured cold CLI path must pass 30 seconds on every image, or the lifecycle must be clarified; repeated reloads are not an established solution to the “load once” requirement. [A20; B1]

## Acceptance checks before calling the build feasible

| Check | Concrete acceptance evidence |
|---|---|
| Hardware/runtime probe | Record `amd-smi`/`rocm-smi` output, `rocminfo` architecture, `sys.executable`, Python version, `torch.__version__`, `torch.version.hip`, GPU name and total VRAM. Confirm `torch.cuda.is_available()` and a real GPU tensor operation. Test supported precision; do not silently fall back to CPU. |
| Dependency lock | Resolve/install the exact pinned Transformers/model dependencies with `python3 -m pip`; record versions before and after to ensure the supplied ROCm torch was preserved. Test imports and `pip check` under Python 3.14. |
| Attention compatibility | Run the full chosen model, including vision backbone, on the actual GPU. Exercise the intended SDPA path and the fallback. For hybrid models, separately exercise GatedDeltaNet prefill and cached decoding. A toy attention operator is insufficient. |
| Memory | Sample externally at **1 second** locally, retain a log sampled every **3 seconds** to mirror grading, and also record PyTorch peak counters. Confirm persistent useful GPU state clears 1 GiB and the full pipeline remains below 48 GiB through the hardest image and all fallback passes. |
| Format coverage | Test PNG RGB/RGBA/palette/grayscale, JPEG including EXIF rotation, and TIFF including compressed and 16-bit/grayscale cases. Pillow needs libtiff support for compressed TIFF. Import `features` from `PIL` and verify `features.check("libtiff")` in the final image. Apply orientation exactly once, including decoder behaviour for TIFF, and scale 16-bit data deliberately before RGB conversion. [A21, A22] |
| Resolution | Test large images, long narrow plates and multi-line signs, including text smaller than the first resized view. Bound model pixels/tokens while preserving a route to inspect a crop. Measure decode time as well as GPU time. The PDF does not define a multi-page TIFF convention; first-frame-only handling remains an explicit assumption to resolve. |
| Reuse and lifecycle | Start the final container as the harness will. Invoke `python3 /app/app.py --input-image ...` in **ten distinct processes**. Verify one model load/worker PID and correct unique outputs. Also test first invocation before readiness, worker death, missing worker, restart and an existing output file. |
| Timing | Startup-to-real-ready **≤600 s**; each command's wall time **≤30 s**; all ten after startup **≤600 s**. Engineering targets of ≤540 s startup and ≤25 s per call leave margin, but are targets rather than observed results. Run the same checks on the published image pulled anonymously. |
| Packaging | Exact base-layer prefix matches; uncompressed image below 60 GiB with recorded measurement method; required paths exist; no hardcoded sample answers or secrets; final image is publicly pullable. |

## Fit the user's available compute

Do model/card/paper review, prompt/data preparation, dependency planning and CPU format checks before launching the notebook. The quota is **three hours of pod lifetime per day**, including idle time, with no rollover. Turn off the session explicitly after work; closing the browser is not the documented stop action. Enter through a fresh `https://notebooks.amd.com/hackathon` page rather than a saved temporary redirected URL. [B1, B2]

At the beginning of every session, identify the URL family and set all model/data/result caches accordingly: **`/persistent` for `jupyter-hack-***`**, **`/workspace` for `rgapi-hackathon-***`**. Other paths are ephemeral. Keep results, package locks and benchmark logs alongside the cached checkpoint; session pip installs must be repeated after reset. Treat **25 GB** as an approximate maximum, not 25 GiB of guaranteed free space. [B1, B2]

Keep one principal checkpoint and at most one small challenger resident on persistent storage; reserve room for a compact labelled evaluation subset, wheels and results. Measure actual directory bytes before adding another model. Do not assume Docker-in-Docker, root privileges or enough image-build storage exist in the notebook. A build host with sufficient storage and a running Linux Docker engine is a separate prerequisite; local Docker is currently installed but unavailable. These build/storage decisions can be prepared without spending the user's GPU quota.

## Sources

All web entries below were retrieved on **2026-09-22**. Each cached response has a matching `.metadata.json` recording request URL, final URL, status and retrieval time. The initial PyTorch `stable` URLs returned HTML redirects; the audit followed them by fetching **2.13-specific documentation** matching the required base version. The older Radeon PyTorch URL redirected to a documentation landing page and is **not used as proof of exact package support**.

- **B1:** [Challenge PDF](../../LabLab_AMD%20AI%20Challenge%20-%20Mini%20Challenge.pdf), pp. 2–8; [extracted text](challenge-extracted.txt).
- **B2:** [COMPUTE.md](../../COMPUTE.md).
- **A1:** [Docker Hub exact-tag API](https://hub.docker.com/v2/repositories/rocm/pytorch/tags/rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0); `amd/docker-hub-exact-tag.json`.
- **A2:** [Docker registry exact manifest](https://registry-1.docker.io/v2/rocm/pytorch/manifests/rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0); `amd/docker-registry-exact-tag.response.json` and metadata. Registry access uses anonymous repository-scoped auth; no token was saved.
- **A3:** [Registry image config](https://registry-1.docker.io/v2/rocm/pytorch/blobs/sha256:75287c3f4d5eba32d8b91797639a20df134e8d1a8167e9abed77d78f330fd0f3); `amd/docker-registry-config.response.json`.
- **A4:** [AMD: OCR with vision-language models with vLLM](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/ocr_vllm.html).
- **A5:** [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/).
- **A6:** [PyTorch 2.13 HIP semantics](https://docs.pytorch.org/docs/2.13/notes/hip.html).
- **A7:** [PyTorch 2.13 SDPA](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html).
- **A8:** [Transformers attention backends](https://huggingface.co/docs/transformers/en/attention_interface) and [PyTorch 2.13 backend controls](https://docs.pytorch.org/docs/2.13/backends.html).
- **A9:** [Docker image inspect](https://docs.docker.com/reference/cli/docker/image/inspect/).
- **A10:** [Docker image specification](https://github.com/moby/docker-image-spec/blob/main/spec.md), `Layer DiffID`, `rootfs`, `history`; [OCI image config](https://github.com/opencontainers/image-spec/blob/main/config.md).
- **A11:** [Docker Engine API v1.53](https://docs.docker.com/reference/api/engine/version/v1.53.yaml), `ImageInspect` and `ImageManifestSummary` schemas.
- **A12:** [Docker image ls size](https://docs.docker.com/reference/cli/docker/image/ls/) and [Docker storage drivers](https://docs.docker.com/engine/storage/drivers/).
- **A13:** [vLLM ROCm release wheel index](https://wheels.vllm.ai/rocm/vllm/); [ROCm nightly variants](https://wheels.vllm.ai/rocm/nightly/).
- **A14:** [vLLM v0.30.0 ROCm Dockerfile](https://github.com/vllm-project/vllm/blob/v0.30.0/docker/Dockerfile.rocm_base); [ROCm requirements](https://github.com/vllm-project/vllm/blob/v0.30.0/requirements/rocm.txt).
- **A15:** [vLLM PyPI metadata](https://pypi.org/pypi/vllm/json) and [Transformers PyPI metadata](https://pypi.org/pypi/transformers/json); use the dated cached responses because these latest-version endpoints change.
- **A16:** [FlashAttention upstream README](https://github.com/Dao-AILab/flash-attention/blob/main/README.md), AMD ROCm support section.
- **A17:** [Transformers v5.17.0 Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
- **A18:** [Docker container exec](https://docs.docker.com/reference/cli/docker/container/exec/).
- **A19:** [Python 3.14 socket](https://docs.python.org/3.14/library/socket.html), `AF_UNIX` support.
- **A20:** [Dockerfile reference](https://docs.docker.com/reference/dockerfile/), `ENTRYPOINT` and `HEALTHCHECK`.
- **A21:** [Pillow file formats](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html), JPEG/PNG/TIFF sections.
- **A22:** [Pillow ImageOps](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html), `exif_transpose`.
