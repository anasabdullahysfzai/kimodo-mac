# Kimodo on Apple Silicon Mac

[![Views](https://visitor-badge.laobi.icu/badge?page_id=anasabdullahysfzai.kimodo-mac&left_text=views)](https://github.com/anasabdullahysfzai/kimodo-mac)
[![Kimodo](https://img.shields.io/badge/NVIDIA-Kimodo-76B900)](https://github.com/nv-tlabs/kimodo)
[![Platform](https://img.shields.io/badge/macOS-Apple%20Silicon-black)](#requirements-check-these-first)


> **How to use:** tell Claude Code (or another coding agent running natively on your Mac):
> *"Follow https://github.com/anasabdullahysfzai/kimodo-mac to install Kimodo on this Mac."*
> Or follow it by hand. Verified on 2026-11-03 on an M5 Pro MacBook Pro (48 GB), macOS 26, Python 3.11, torch 2.15. Kimodo's interactive web demo (`kimodo_demo`) works, and text prompts generate motion in the browser.
>
> This is an unofficial community guide, not affiliated with NVIDIA.

---

## Goal

Install NVIDIA Kimodo (https://github.com/nv-tlabs/kimodo), a text-to-3D-human-motion diffusion model, and run its interactive web demo at http://localhost:7860 on this Mac. The upstream repo targets Linux + NVIDIA CUDA. The steps below include every Mac-specific fix needed. Follow them in order, run commands natively on the Mac, and stop and report the first unexpected error before patching anything not listed here.

## Requirements (check these first)

1. **Apple Silicon Mac** (`uname -m` → `arm64`). **32 GB+ unified memory** recommended: the text encoder alone is a 16 GB Llama-3-8B in bf16. Only tested with 48 GB.
2. **~25 GB free disk**: about 16 GB for Llama-3-8B-Instruct, about 1.1 GB per Kimodo checkpoint, plus the Python env.
3. **Homebrew tools:** `uv` and `git` (`brew install uv`).
4. **Hugging Face account with access to `meta-llama/Meta-Llama-3-8B-Instruct`.** This is a gated model:
   - Request access at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct ("Expand to review and access", real name/details). Approval took a few hours.
   - Create a **Read** token at https://huggingface.co/settings/tokens and run `hf auth login`. Check with `hf auth whoami`.
   - Verify access (should print `200`, not `403`):
     ```bash
     curl -s -o /dev/null -w "%{http_code}\n" -I -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/resolve/main/config.json
     ```
   - OpenRouter or any chat API **cannot** replace this: Kimodo reads Llama's internal hidden states through LLM2Vec, so the weights must run locally.

## Important for AI agents: sandboxed shells

Claude Code's Bash sandbox allows `huggingface.co` but **blocks Hugging Face's file CDN** (`us.aws.cdn.hf.co`, `cas-bridge.xethub.hf.co`, …). Large downloads stall (e.g. stuck at exactly 1,024,000,000 bytes) or fail with `No route to host`. Also, the two servers must keep running and talk to each other on localhost.

**So: run the big download (step 4) and both servers (step 6) in a real terminal**, either the user's own terminal or Claude Code's Terminal panel (`run_in_terminal`). Don't run them through the sandboxed Bash tool.

## Step 1: Clone and create the environment

```bash
mkdir -p ~/kimodo-local && cd ~/kimodo-local
git clone https://github.com/nv-tlabs/kimodo.git
cd kimodo
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"   # expect: True
```

## Step 2: Install Kimodo WITHOUT the MotionCorrection C++ extension

The bundled `MotionCorrection` C++ extension is written with **x86 SSE/AVX intrinsics** (`immintrin.h`, `-msse4.1 -mavx`), so it cannot compile for arm64. It is only used by the optional "Post Processing" step, so skip it:

```bash
SKIP_MOTION_CORRECTION_IN_SETUP=1 uv pip install -e ".[all]"
python -c "import kimodo, viser; print('imports ok')"
```

(If you ever try building it anyway: an x86_64 `cmake`, e.g. from MacPorts at `/opt/local/bin/cmake`, silently produces x86_64 binaries that won't load in arm64 Python. Check with `file $(which cmake)`.)

## Step 3: Turn off post-processing by default

The demo enables Post Processing by default, which needs the extension skipped above and would raise an error on every generation. Turn the default off:

```bash
sed -i '' 's/^INIT_POSTPROCESSING = True/INIT_POSTPROCESSING = False  # macOS: motion_correction not built on arm64/' kimodo/demo/config.py
grep -n "INIT_POSTPROCESSING =" kimodo/demo/config.py
```

Users must keep the **Post Processing** checkbox off in the UI.

## Step 4: Download the models (real terminal, see the sandbox note)

```bash
source .venv/bin/activate
hf download meta-llama/Meta-Llama-3-8B-Instruct --exclude "original/*"   # ~16 GB; re-run to resume if it stops
hf download nvidia/Kimodo-SOMA-RP-v1.2                                    # demo default, ~1.1 GB (optional; auto-downloads on launch)
```

The LLM2Vec adapters (`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` and `…-mntp-supervised`, ~330 MB, public) download automatically on the first encoder start.

## Step 5: (Only if Llama access is still pending) use an ungated mirror

You can skip this if step 4 worked. While waiting for Meta's approval, the encoder can use the identical re-upload `NousResearch/Meta-Llama-3-8B-Instruct`. Meta's Llama 3 license still applies. Download the two adapter repos locally and repoint the base model:

```bash
cd ~/kimodo-local
for r in McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised; do hf download $r --local-dir text_encoders/$r; done
sed -i '' 's#"meta-llama/Meta-Llama-3-8B-Instruct"#"NousResearch/Meta-Llama-3-8B-Instruct"#' text_encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/adapter_config.json
hf download NousResearch/Meta-Llama-3-8B-Instruct --exclude "original/*"
```

Then add `TEXT_ENCODERS_DIR="$HOME/kimodo-local/text_encoders"` in front of **both** commands in step 6. The prompt formatting is unaffected because the adapter's `config.json` still names the Meta model.

## Step 6: Launch (two real terminals, encoder first)

**Terminal A: text encoder service** (runs Llama on the Mac GPU via MPS, port 9550):
```bash
cd ~/kimodo-local/kimodo && source .venv/bin/activate && PYTORCH_ENABLE_MPS_FALLBACK=1 TEXT_ENCODER_DEVICE=mps TEXT_ENCODER_TMP_FOLDER="$TMPDIR/text_encoder" kimodo_textencoder
```
Wait for `Running on local URL: http://0.0.0.0:9550` (first start takes a few minutes).

**Why `TEXT_ENCODER_TMP_FOLDER`:** the default `/tmp/text_encoder` resolves to `/private/tmp` on macOS. Gradio rejects it with `InvalidPathError: Cannot move /tmp/text_encoder/....npy to the gradio cache dir`. The demo then reports `Text encoder service is unreachable, falling back to local LLM2Vec encoder` and loads a second 16 GB copy of Llama on CPU. If you ever see that fallback message, stop the demo and fix the encoder first.

**Terminal B: demo** (port 7860):
```bash
cd ~/kimodo-local/kimodo && source .venv/bin/activate && PYTORCH_ENABLE_MPS_FALLBACK=1 kimodo_demo
```
Wait for `Model kimodo-soma-rp loaded successfully` (and **no** "unreachable" line), then open **http://localhost:7860**.

## Step 7: Verify

1. Close the welcome dialog. The default prompt box reads "A person walks forward." (6 s).
2. Click **Generate**. Terminal B shows a 100-step progress bar.
3. Scrub the timeline (e.g. frame 105). The character should be mid-stride, well forward of the start.

**Expected speed (M5 Pro):** ~60 s for a 6 s / 180-frame clip. The motion model runs on **CPU**, because the demo picks `cuda` if available, else `cpu`. The prompt encoding on MPS is fast.

## Harmless warnings (ignore)

- `torch.jit.script is deprecated`
- Encoder load report: `lm_head.weight | UNEXPECTED`, and a table of `lora_A/lora_B … UNEXPECTED / MISSING`. That's the pinned transformers/peft versions auto-loading the adapter with a key-prefix mismatch. Kimodo then loads it properly via PEFT, hence `Already found a peft_config attribute in the model`. Same on Linux. Output quality is fine.
- `The parameters have been moved from the Blocks constructor to the launch() method in Gradio 6.0`
- `To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone()`

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `403 … not in the authorized list` | Llama access not approved yet, or token not set. See requirements, or use step 5. |
| `Repository not found` on `meta-llama/...` | Same as above (gated repo, no access). |
| Download stuck at 1.0 GB / `No route to host` | Running inside a sandboxed agent shell. Run it in a real terminal. Plain re-run resumes. |
| `unknown type name 'FORCE_INLINE'`, `undeclared identifier 'labs'`, `unsupported option '-msse4.1'` | You're building MotionCorrection. Use `SKIP_MOTION_CORRECTION_IN_SETUP=1` (step 2). |
| `incompatible architecture (have 'x86_64', need 'arm64')` | x86 `cmake` built the extension. Skip the extension (step 2). |
| `motion_correction package is not installed` on Generate | Post Processing checkbox is on. Turn it off (step 3). |
| `Text encoder service is unreachable, falling back…` | Encoder not running, or the `/tmp` path issue. See step 6. |

## Prompting tips (what worked)

Kimodo generates **one body with no props or other people**. Describe what the body does, not objects or targets.

- Start with "A person …". Name sides/limbs ("right leg", "left hand"), direction, and speed/style.
- Pattern that worked best: **stance/style + intensity word + body landmark**. For example, *"A person in a fighting stance kicks their right leg extremely high, lifting their foot above their head."* reached head height. *"…kicks to the head of the other person"* did not, because there is no other person.
- Replace objects with poses: "holding a gun" → *"walks forward slowly with both arms extended in front of their chest, hands together as if aiming."*
- One action per prompt box. Use several boxes on the timeline for sequences (click empty track space to add, drag the right edge to resize, double-click to edit).
- Knobs: **Model Parameters → Seed** (different takes; note the seed of a good one), **Classifier-Free Guidance → Text Weight** (default 2.0; ~3 follows wording more strictly, >4 can get jittery), **Num Samples** (several at once).
- Rare moves (e.g. spinning head-height kicks) may not come out from text alone. Use the timeline constraints: at the peak frame add a **Right Foot** (or **Full-Body**) constraint, drag it into place in **Constraints → Enter Editing Mode**, raise **Constraint Weight**, and regenerate.

## Not done / possible next steps

- Running the motion model on MPS (would need a small patch to `kimodo/demo/app.py`, which hard-codes `cuda:0` or `cpu`). Untested.
- Porting MotionCorrection to arm64 (e.g. via `sse2neon`). Untested.
- `kimodo-mlx` (https://github.com/NomaDamas/kimodo-mlx) is a CLI-only MLX port with an ungated text bundle, if you don't need the UI.
