# OCR literature evidence for the AMD mini-challenge

Research date: **2026-09-22**. This review used the requested global Claude [arXiv skill](C:/Users/HARSHDIP/.claude/skills/arxiv/SKILL.md) and the research skill. The arXiv Atom API and all nine shortlisted PDF downloads returned HTTP 200. Full PDFs were downloaded and the methods, evaluation, and limitations sections identified below were read; this is not an abstract-only review. Original metadata, retrieval timestamps, PDFs, and page-marked text are in [papers/](papers/), with [retrieval-log.jsonl](papers/retrieval-log.jsonl) recording the requests. None of these models has been run on the challenge GPU in this research step.

## What the evidence supports

**Build a measured OCR pipeline for short scene text, with explicit plate/sign handling. Start with a compact general VLM, benchmark one compact OCR specialist, and add extra passes only when they improve held-out normalized whole-string accuracy within the time limit.** Qwen3.5-9B has useful independent current benchmark evidence; Qwen3-VL-4B/8B-Instruct has well-documented bilingual OCR results. PaddleOCR-VL-1.6 has a genuinely relevant text-spotting mode and is a stronger specialist candidate than a document-only headline suggests. These are candidates to measure, not an established ranking on the hidden challenge images. [P1, P2, P5]

The challenge's target-selection rules come from the [provided challenge text](challenge-extracted.txt), pages 6–7 and 9–13: retain the entire US registration while excluding surrounding jurisdiction/slogan text; retain Chinese province characters and the following letter; retain all sign text in reading order; preserve numbers without adding inferred units. A model can recognize every visible character correctly and still fail by returning the wrong subset. None of the reviewed benchmark scores directly measures that contract.

The recommendations below are engineering deductions from the papers and challenge rules, rather than published results of this proposed system:

- Use the original image for an initial bounded, deterministic read. Preserve source resolution long enough to find small text; compare an original crop and a geometrically rectified crop when perspective is strong. The Qwen architecture supports dynamic-resolution input, and OCRBench v2 identifies input resolution as a consequential variable. [P1, P6]
- Separate registration selection from transcription. A plate region and text-line geometry can help remove a small banner without deleting a Chinese registration prefix. Do not use a global list of words/provinces to strip arbitrary text, and do not apply language-model spelling correction to registrations.
- Test an independent crop recognizer, such as a suitable PP-OCRv5 variant or PaddleOCR-VL-1.6, only if the runtime is practical on the mandated ROCm/Python image. Use disagreement to trigger a bounded second look; agreement is evidence of consistency, not proof of correctness. [P2, P3, P7]
- Keep mild contrast/brightness adjustments and rectification as alternatives alongside the original. Do not make generic generative super-resolution or deblurring a default prerequisite. Real plate restoration can help, but a restoration model's attractive output is not evidence that it preserved the registration, especially its Chinese prefix. [P7, P8]
- Select models by normalized **complete-answer exact match**, with results sliced by US plates, Chinese plates, worded signs, number-only plaques, blur, glare/low light, noise, and perspective. Include unseen random registrations, repeated digits, and visually similar character pairs. Keep synthetic variants of a source image in the same split. Report latency and peak VRAM alongside accuracy.
- Respect [COMPUTE.md](../../COMPUTE.md): a three-hour daily session and approximately 25 GB persistent storage favor a short, staged comparison, not training a new recognizer or retaining many large checkpoints. Paper speed numbers measured on NVIDIA hardware are not AMD latency guarantees. Compatibility and the challenge's 1–48 GiB VRAM requirement require separate runtime verification.

## Evidence ledger

| ID | Primary paper and verified arXiv dates | Material actually read | Decision value |
|---|---|---|---|
| P1 | [Qwen3-VL Technical Report, 2511.21631v2](https://arxiv.org/abs/2511.21631v2). Submitted 2025-11-26; v2 2025-11-27. | §2, §3.2.3, §5.4, Table 4, evaluation appendix. [Local text](papers/2511.21631v2.txt). | General VLM baseline, bilingual recognition, Instruct versus Thinking. |
| P2 | [PaddleOCR-VL-1.6: Expanding the Frontier of Document Parsing with Under-Optimized Region Refinement and Progressive Post-Training, 2606.03264v1](https://arxiv.org/abs/2606.03264v1), 2026-06-02; plus predecessor [PaddleOCR-VL-1.5, 2601.21957v2](https://arxiv.org/abs/2601.21957v2), submitted 2026-01-29, v2 2026-04-03. | 1.6 §2–3, §5.1, §5.2.3/Table 6; 1.5 §2.1–2.2, §4.2.1/Table 4. [1.6 text](papers/2606.03264v1.txt), [1.5 text](papers/2601.21957v2.txt). | Compact scene text spotting, low-quality images, quadrilateral localization. |
| P3 | [PP-OCRv5: A Specialized 5M-Parameter Model Rivaling Billion-Parameter Vision-Language Models on OCR Tasks, 2603.24373v1](https://arxiv.org/abs/2603.24373v1), 2026-03-25. | §3, §4.1–4.3, Tables 3–4, footnote 1. [Local text](papers/2603.24373v1.txt). | Small visual recognizer, deployment-sensitive alternative, data diversity. |
| P4 | [GLM-OCR Technical Report, 2603.10910v2](https://arxiv.org/abs/2603.10910v2). Submitted 2026-03-11; v2 2026-03-16. | §2.1–2.2, §3/Tables 3–5, §5.3.1, §6. [Local text](papers/2603.10910v2.txt). | Compact direct text-recognition candidate; document benchmark caveats. |
| P5 | [CC-OCR V2: Fine-Grained Attribution of LMM Failures in Real-World Visual Document Understanding, 2605.03903v2](https://arxiv.org/abs/2605.03903v2). Submitted 2026-05-05; v2 2026-08-10. | §2–4, Tables 2–3, Appendices A.5–A.8, model table and recognition prompt. [Local text](papers/2605.03903v2.txt). | Current Qwen3.5-9B evidence and acquisition-condition failure taxonomy. |
| P6 | [OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning, 2501.00321v2](https://arxiv.org/abs/2501.00321v2). Submitted 2024-12-31; v2 2025-06-05. | §4, Appendix A.1, A.5, A.8–A.9, Tables 6–8 and resolution discussion. [Local text](papers/2501.00321v2.txt). | Bilingual scene-text evaluation; separates reading from spotting/reasoning. |
| P7 | [Visual Merit or Linguistic Crutch? A Close Look at DeepSeek-OCR, 2601.03714v2](https://arxiv.org/abs/2601.03714v2). Submitted 2026-01-07; v2 2026-01-08. | §2–4, Tables 1–7, §7/conclusion. [Local text](papers/2601.03714v2.txt). | Non-semantic strings expose failures concealed by fluent natural text. |
| P8 | [A Dataset and Model for Realistic License Plate Deblurring, 2404.13677v2](https://arxiv.org/abs/2404.13677v2). Submitted 2024-04-21; v2 2024-04-23. | §3–5, Tables 1–3, §6. [Local text](papers/2404.13677v2.txt). | Real versus synthetic blur; limitations of restoration for Chinese plates. |

## P1 — Qwen3-VL: useful baseline, especially Instruct

The architecture combines a SigLIP-2-derived vision encoder, a two-layer MLP merger, and a Qwen3 decoder. It accepts dynamic image resolutions and uses DeepStack to inject visual features from three levels into early language-model layers (§2, PDF pp. 2–4). OCR training includes real-world images, OCR-specialist pseudo-labels refined by Qwen2.5-VL, and synthetic plus real multilingual material (§3.2.3, p. 6). This supports testing it for both scene understanding and transcription; it does not establish accuracy on degraded registrations.

Table 4 (PDF p. 18) reports the following values for the **Instruct** variants:

| Model | OCRBench v2 English | OCRBench v2 Chinese | Original CC-OCR |
|---|---:|---:|---:|
| Qwen3-VL-2B-Instruct | 56.3 | 53.0 | 72.8 |
| Qwen3-VL-4B-Instruct | 63.7 | 57.6 | 76.2 |
| Qwen3-VL-8B-Instruct | 65.4 | 61.2 | 79.9 |

For comparison, the 4B Thinking variant records 61.8 / 55.8 / 73.8 on those rows. This is a defensible reason to start with bounded Instruct decoding for transcription. It is not proof that every deterministic prompt beats every reasoning setup. OCRBench v2 is a mixed-task benchmark; CC-OCR here is **v1**, not P5's v2. Do not turn these values into expected plate accuracy or compare them directly to document edit-distance scores.

Qwen3.5 is newer and deserves screening. The arXiv discovery pass did not establish an official base Qwen3.5 report; it found an Omni report and downstream Qwen3.5 work. The present review therefore uses the direct independent Qwen3.5-9B evaluation in P5, and leaves checkpoint/configuration details to current official model cards.

## P2 — PaddleOCR-VL-1.6: test the spotting path

The 0.9B model retains a native-resolution visual encoder, adaptive MLP connector, and ERNIE-4.5-0.3B decoder. Its **document parsing** path uses PP-DocLayoutV3 and region recognition; its **text spotting** path directly uses the VLM for detection plus recognition, including signboards and advertising posters (§2, PDF pp. 4–5). A full document-layout pipeline is consequently not required just to test scene spotting.

The 1.5 paper explains the spotting output: text followed by eight dedicated location tokens encoding a quadrilateral's four corners, normalized to 0–1000 (§2.2.2, PDF p. 8). This gives a useful way to separate a registration line from surrounding banners, provided the model actually localizes them correctly. The 1.6 paper describes mining examples unstable under small pixel shifts, compression, noise, blur, and scaling, then improving data and post-training (§3.2, pp. 7–8).

Table 6 in 1.6 (PDF p. 19) is a **private in-house end-to-end spotting benchmark**:

| Model | Overall, mean of nine categories | Blur | Common scenes |
|---|---:|---:|---:|
| PaddleOCR-VL-1.5 | 86.21 | 84.22 | 77.13 |
| PaddleOCR-VL-1.6 | 87.47 | 90.59 | 77.28 |

These are relevant screening signals, not public plate-test results. The table does not include Qwen3.5 or GLM-OCR. Its prose claims superiority across all nine dimensions, but the actual table contains regressions from 1.5 in Chinese handwriting (89.52 → 85.90) and printed English (86.89 → 86.51); rely on the table rather than repeating that blanket claim.

The headline **96.33 on OmniDocBench v1.6** (§5.1/Table 2) concerns document parsing. It must not be compared directly with the predecessor's **94.50 on OmniDocBench v1.5**, since the benchmark version differs, or treated as traffic-sign/plate accuracy. Runtime, package compatibility, prompts, and checkpoint license must be checked from official current deployment sources.

## P3 — PP-OCRv5: independent visual evidence, with variant discipline

The model uses a DB-based detector with a PP-LCNetV3 backbone and an SVTR/LCNet recognizer, trained with an attention-guided CTC scheme (§3.1, PDF p. 3). Text-line detection followed by crop recognition is a natural fit for small registration strings. The report emphasizes diverse difficult training examples, not a need to reproduce large-scale training during this challenge.

Footnote 1 (PDF p. 2) explicitly says the paper primarily studies **PP-OCRv5_mobile, 5M parameters**; a higher-capacity server variant also exists. Do not apply the 5M claim to every PP-OCRv5 release. In-house weighted recognition accuracy is 80.1 versus 53.0 for PP-OCRv4 (Table 3, p. 7). Table 4 (p. 8) reports normalized edit distance **0.067 overall**, **0.058 English**, **0.076 Chinese** on **component-level OmniDocBench OCR**, where lower is better. Those are neither scene-only results nor complete-string accuracy.

The practical role is a cheap independent recognizer/detector if an AMD-compatible runtime is available. A tiny OCR model used by itself might also fail the challenge's minimum GPU-memory gate; measure real GPU use instead of inferring compliance from parameter count. P7 offers independent evidence favoring a PaddleOCR-v5 pipeline on random strings, but its evaluated variant is listed as 0.07B, so its scores must not be attributed to this report's mobile model.

## P4 — GLM-OCR: compact challenger, not a proven plate specialist

GLM-OCR combines a 400M CogViT encoder with a 500M GLM decoder. The report describes multi-token prediction and a PP-DocLayoutV3-plus-region-recognition document pipeline (§2.1, PDF pp. 4–5). Crucially, §5.3.1 (p. 10) also documents a standalone prompt, `Text Recognition:`, for plain-text transcription, including a menu-board example with perspective and background clutter. Therefore a direct crop-recognition trial is reasonable without adopting its entire SDK.

Table 3 (p. 7) reports **94.0 on OCRBench (Text)** and **94.6 on OmniDocBench v1.5**. The former is explicitly a **text subset**; it is not the full OCRBench score, OCRBench v2, or plate accuracy. The latter aggregates document elements. A reported throughput advantage from its serving/MTP setup is not guaranteed by ordinary `transformers.generate` or by an AMD port.

The report acknowledges error propagation from layout mistakes, weaknesses with very low resolution or heavy distortion, and possible output-format variation (§6, p. 13). Nothing in the inspected tables establishes banner exclusion or Chinese registration exact match. Compare against the chosen baseline only after a short compatibility check.

## P5 — CC-OCR V2: current Qwen3.5-9B evidence and the right failure categories

This benchmark contains 7,093 samples, 16 subtasks, 32 recognition languages, and ten diagnostic factors spanning acquisition and document conditions (§2–3). Table 2 (PDF p. 5) provides the following directly relevant independent comparison:

| Model | Recognition | Grounding | Average across five operations |
|---|---:|---:|---:|
| Qwen3.5-9B | 83.89 | 47.06 | 67.35 |
| MiniCPM-o 4.5-8B | 61.09 | 13.75 | 51.97 |
| InternVL3.5-8B | 46.36 | 11.37 | 48.30 |
| Step3-VL-10B | 40.95 | 3.37 | 40.96 |

The public model identifier for the first row is `Qwen/Qwen3.5-9B` (Appendix A.8, Table 5, p. 17). There is **no Qwen3.5-4B row** in this comparison; do not extrapolate 9B numbers to the smaller checkpoint. §3 reports common task prompts and temperature zero; the local models used vLLM with FlashAttention, without establishing challenge hardware performance.

The recognition metric is **multi-set micro-F1** (Appendix A.7, pp. 13–14), deliberately reducing reading-order effects. Grounding is macro-averaged IoU, and the overall column mixes five different task metrics. A reordered string can receive credit here while failing the challenge. These results support candidate selection, not a claim of “83.89% plate accuracy.”

The failure analysis is highly relevant: blur, low resolution, shadows, rotation, dense small text, and perspective cause differing degradation; visually ambiguous content can induce plausible hallucinations or dominant-language drift (§4.4, Appendix A.5–A.6). Use these factors to stratify the challenge validation set instead of relying on a single average. The dataset remains document-centric despite including information boards and diverse image acquisition.

## P6 — OCRBench v2: bilingual capability, not the submission metric

OCRBench v2 covers 10,000 public English/Chinese image-question samples across 23 subtasks, eight capability groups, and 31 scenarios; the paper also evaluates a private set (§4, PDF pp. 5–6). Text recognition, localization, spotting, relation extraction, document parsing, calculation, and reasoning are intentionally separated. Its public data includes 7,400 English and 2,600 Chinese images (§4.3).

Appendix A.1 (pp. 21–22) compares representative text recognizers and VLMs, and makes clear that strong recognition does not imply strong spotting. Appendix A.9 (p. 29) discusses a resolution study using InternVL2-8B: increasing 448 to 896 improves the reported performance, motivating a **bounded resolution/crop ablation**, not an assumption that unbounded image enlargement always helps.

Metrics vary by task, including recognition scores, IoU, F1, normalized edit similarity, and question-answer rules (Appendix A.5). Hence OCRBench v2 and its predecessor, OCRBench Text, CC-OCR, and OmniDocBench should occupy separate columns in a model screen. For this challenge, the decisive metric is the evaluator's normalization followed by exact complete-string equality.

## P7 — Random strings expose language-prior failure

This controlled study uses **112 English Fox document pages**, with 600–2,500 ground-truth tokens, then renders natural text and semantically corrupted versions (§2.1, PDF p. 2). Its strongest perturbation generates random mixed-case words of 2–10 letters (§3.1, p. 3). This is relevant to the absence of natural language context in registrations, but it is **not a traffic-plate benchmark**.

Tables 6–7 (PDF p. 6) report OCR precision:

| Evaluated configuration | Natural rendered text | Random rendered text |
|---|---:|---:|
| DeepSeek-OCR Tiny | 88.00 | 19.84 |
| DeepSeek-OCR Small | 95.23 | 42.12 |
| Qwen2.5-VL-7B | 98.10 | 46.86 |
| PaddleOCR-v5, listed as 0.07B | 94.44 | 89.53 |

Table 5 gives examples of misspelled visible strings being changed into plausible words. This is strong evidence to include random strings and avoid semantic autocorrection; it is **not** evidence that Qwen3.5, PaddleOCR-VL-1.6, or DeepSeek-OCR 2 has the same numerical failure. The tested model versions, page length, rendering, compression configuration, and metric all matter. No broad claim that all generative OCR fails plates follows from this experiment.

DeepSeek-OCR and DeepSeek-OCR 2 were discovered through official arXiv metadata ([2510.18234v1](https://arxiv.org/abs/2510.18234v1), [2601.20552v1](https://arxiv.org/abs/2601.20552v1)); their full reports were not shortlisted for this read. Their optical-compression/document emphasis alone supplies no stronger scene-plate evidence than the candidates above. P7 critiques the first model, not the second.

## P8 — LPBlur: restoration needs registration-level validation

LPBlur contains paired real short/long-exposure captures, with denoising, geometric alignment, and plate cropping (§3, PDF pp. 3–4). The split has 9,288 training pairs and 1,000 test pairs, with 500 normal-light and 500 low-light test pairs (§5.1, p. 5). LPDGAN uses multi-scale processing, text reconstruction supervision, and partition discriminators (§4).

Table 2 (PDF p. 7) compares training on synthetic blur with training on real LPBlur pairs. Its **Text Levenshtein Distance**, lower better, is 1.68 versus 0.57 in normal light and 2.65 versus 0.81 in low light. This supports the value of real degradation data. However, §5.1/§5.3 defines the comparison using CRNN text recognized from generated and sharp images. It is not the challenge's full-string accuracy against human ground truth; even an average distance below one can hide many failed registrations.

The conclusion explicitly identifies improving restoration of spatially complex characters such as **Chinese characters** as future work (§6, p. 7). This is a concrete reason not to place that restoration model unconditionally ahead of a Chinese-plate recognizer. A sensible ablation would compare original, mildly enhanced, and restored crops against the same independent character labels, retaining a learned restoration path only if complete-registration accuracy improves without unacceptable latency or new character substitutions.

## Boundaries of this review

This is a reproducible candidate/evaluation rationale as of the access date, not a universal SOTA declaration. Manufacturer papers, private benchmarks, independent benchmarks, and challenge-specific engineering deductions are distinguished above. No reviewed result proves performance on the hidden test set, current AMD kernels, the mandated Python/ROCm image, or the supplied samples. The root study separately checks live model cards, licensing, installation compatibility, and deployment constraints before turning these recommendations into the implementation specification.
