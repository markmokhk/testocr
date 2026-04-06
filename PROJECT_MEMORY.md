# testocr — codebase recall (compact)

**Purpose:** Experiment repo (MNIST digit training + older PDF/chord preview scripts). Path: `C:\Users\markm\projects\testocr`.

**Environment:** Conda env **`testocr`**, Python **3.11**, `pip install -r requirements.txt` → **torch 2.11.0**, **torchvision 0.26.0**. On first `conda create`, **Anaconda channel Terms of Service** may need to be accepted (`conda tos accept ...` for default channels).

**Git:** Remote **`https://github.com/markmokhk/testocr`**, branch **`main`**.

**Ignored (not in Git):** `data/` (MNIST downloads), `checkpoints/` (`.pt` from training), `chrome-tmp/`, usual Python junk.

**Tracked assets:** Multiple **`preview_*.png`**, **`scores_samples.pdf`**, **`samples/`** MNIST PNG exports — can bulk up the repo over time.

---

## ML pipeline (MNIST)

| Piece | Location / note |
|--------|------------------|
| Download MNIST | `scripts/download_mnist.py` → `data/mnist` |
| Export 30 sample PNGs | `scripts/export_mnist_samples.py` → `samples/` |
| Train | `scripts/train_mnist.py` — small **CNN**, Adam, CE loss, `num_workers=0` (**Windows**), saves **`checkpoints/mnist_cnn.pt`** (dict with `model_state_dict` + hparams) |

**Gaps / follow-ups:** No dedicated **inference** script (must rebuild `MnistCNN` and `load_state_dict`). No **tests/CI**. Optional official-style tweaks: `pin_memory` on GPU, AMP, `torch.compile` — not required for MNIST correctness.

**Product reality:** MNIST = **10 digits**, **28×28**; does **not** generalize to full-page or Chinese OCR without more data and pipeline.

---

## Other scripts (different domain)

- `scripts/color_preview5_chords.py`, `color_preview5_chords_pymupdf.py`, `color_preview10_chords_manual.py` — PDF/chord coloring experiments; **not** tied to MNIST training.

---

## Operational notes

- **Always activate the ML env before training:** PyTorch is installed in conda env **`testocr`**, not in **`(base)`**. If you see `ModuleNotFoundError: No module named 'torch'`, you are on the wrong environment.

  ```powershell
  conda activate testocr
  cd C:\Users\markm\projects\testocr
  python scripts\train_mnist.py
  ```

- **Console:** Traditional Chinese `print` in `train_mnist.py` may **garble** in default PowerShell encoding; file comments are fine in the editor.
- **Context7:** Not wired in this workspace by default; add MCP if you want doc-backed checks.
