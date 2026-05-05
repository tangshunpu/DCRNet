# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

PyTorch reference implementation of **DCRNet** — a dilated-convolution autoencoder for CSI feedback compression in massive MIMO-OFDM systems (paper: "Dilated Convolution Based CSI Feedback Compression for Massive MIMO Systems"). It trains on the **COST2100** dataset (indoor / outdoor scenarios) to compress and reconstruct a `2 × 32 × 32` angular-delay-domain CSI tensor.

## Environment

- A project-local virtualenv lives at `./.venv` — activate it (`source .venv/bin/activate`) before running anything; the system has no `pip`/`conda` and Python tools are managed via `uv` (`uv tool install` / `uvx`).
- `requirements.txt` pins `torch==1.7.0`, but `main.py` has been patched to the modern complex-FFT API (`torch.view_as_real(torch.fft.fft(torch.view_as_complex(...), dim=-1))`), so it runs on newer PyTorch. **`statics.py` still uses the removed `torch.fft(x, signal_ndim=1)` call** — keep this in mind if you touch it.

## Dataset

- The training script expects the COST2100 `.mat` files under `./COST2100/` (path overridable with `--data`). In this checkout, `./COST2100` is a symlink to `~/Documents/dataset/COST2100`, where the files are already extracted.
- Required files per scenario `<s>` ∈ {`in`, `out`}: `DATA_Htrain<s>.mat`, `DATA_Hval<s>.mat`, `DATA_Htest<s>.mat`, `DATA_HtestF<s>_all.mat` (the last one provides full-bandwidth complex CSI used only for the ρ metric in `evaluator`).
- The Google Drive link in `README.md` is dead; a working Dropbox mirror exists. Do **not** redownload — use the existing extracted files.

## Common commands

Train (paper recipe; logs/checkpoints land in `./outputs/`):

```bash
python main.py --gpu 0 --lr 2e-3 -v 5 --cr 4 --scenario in --expansion 1
```

Evaluate a checkpoint without training:

```bash
python main.py --gpu 0 --cr 4 --scenario in --expansion 1 \
               --pretrained outputs/checkpoints/1X-cr4-in --evaluate
```

Resume training from a checkpoint (restores epoch/optimizer/scheduler too):

```bash
python main.py --gpu 0 --lr 2e-3 -v 5 --cr 4 --scenario in --expansion 1 \
               --resume outputs/checkpoints/1X-cr4-in
```

CPU run: omit `--gpu` (disables `pin_memory` and the CUDA `PreFetcher`).

There is **no test suite, linter, or build step** — `main.py` is the only entry point that gets exercised.

## Code architecture

The runtime path is small and linear:

- **`main.py`** — argument parsing, logger setup, device selection, model instantiation (with `thop` FLOP/param profiling on a dummy `1×2×32×32` input), training loop, periodic test, best-NMSE checkpointing. Also defines `WarmUpCosineAnnealingLR` (linear warmup over `T_warmup` steps then cosine to `eta_min=5e-5`) and a self-contained `evaluator` that computes both NMSE on the angular-delay tensor and ρ on the full-bandwidth FFT-reconstructed CSI.
- **`models/dcrnet.py`** — defines `DCRNet(in_channels=2, reduction, expansion)` and the factory `dcrnet(reduction, expansion)`. Architecture: `ConvBN` 5×5 → `DCREncoderBlock` (asymmetric 3×1/1×3 dilated convs at d=1/2/3 in series concatenated with a parallel 3×3 branch, fused by 1×1, residual) → flatten → FC down to `2048/reduction` → FC up to 2048 → reshape → 5×5 ConvBN → two `DCRDecoderBlock`s (two parallel grouped+shuffled paths with dilation, fused 1×1, residual) → Sigmoid. `--expansion` scales decoder width via `width = 8 * expansion`.
- **`dataset/cost2100.py`** — `Cost2100DataLoader` loads the four `.mat` files into `TensorDataset`s; the test set yields `(sparse_gt, raw_gt)` pairs because `evaluator` needs full-band complex CSI for ρ. When `pin_memory=True` (i.e. GPU mode) each `DataLoader` is wrapped in `PreFetcher`, which uses a CUDA stream to overlap host→device copies with compute. The exported package surface is `from dataset import Cost2100DataLoader` and `from models import dcrnet`.

CLI knobs to know:

- `--cr N` — compression ratio is `1/N` (codeword size = `2048 // N`).
- `--scenario {in,out}` — selects indoor (5.3 GHz) or outdoor (300 MHz) `.mat` files.
- `--expansion {1,10,...}` — decoder width multiplier (`1×` is the lightweight model, `10×` is the high-capacity variant from the paper).
- `--val-freq / -v` — runs `test()` (and saves on best NMSE) every N epochs; there is no separate val invocation in `main.py` despite the name.
- Output naming: logs `outputs/log/<MMDDHHMM>-<expansion>X-<cr>-<scenario>.log`, checkpoint `outputs/checkpoints/<expansion>X-cr<cr>-<scenario>` (no extension, single best-NMSE file).

## Vestigial code — do not extend

- **`solver.py`** is an older `Trainer`/`Tester` pipeline that imports `wandb` and references undefined `logger`/`args`. It is **not** wired into `main.py`. Don't add features there; modify `main.py` instead. Same for **`statics.py`** — `main.py` has its own inline `AverageMeter` and `evaluator`, and the `statics.py` copy still calls the removed `torch.fft` API.
