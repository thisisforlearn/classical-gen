# MAESTRO — classical music composer bot

**Author: Vaibhav** (sole author; final authority on this project)
**License: GPL-3.0-only** — see [LICENSE](LICENSE). Never MIT.

A two-model system, the proven NotaGen (IJCAI 2025) recipe:

1. **Generator** — a RoPE/RMSNorm/SwiGLU GPT trained from scratch on **MAESTRO v3**
   (1,283 classical piano works: Bach, Mozart, Beethoven, Chopin, Liszt, Rachmaninoff...).
2. **Critic (grader)** — a second transformer trained to tell real compositions from
   generated ones, then used two ways:
   - **best-of-N**: scores candidate sections while composing, keeps the best,
   - **REINFORCE RL loop**: pushes the critic's judgment back into the generator.

Output per run: **MIDI + WAV/MP3 audio + MusicXML sheet music (+ PDF if MuseScore exists)**.
Target length: **16 minutes**, composed hierarchically (form plan → sections → stitch),
each section chosen by the critic out of N candidates.

---

## Quick start (this machine, WSL2 Debian, i5-1335U 12 threads, AVX2)

```bash
source ~/ml_env/bin/activate
cd ~/music-bot

python -m musicbot check      # environment report
python -m musicbot kaggle     # writes models/kaggle_cell.py + prints instructions
```

## Train on Kaggle (free GPU) — one paste, max quality

1. `python -m musicbot kaggle` → prints exact steps and writes `models/kaggle_cell.py`
2. kaggle.com → New Notebook → Accelerator **GPU T4 x2** → Internet **On**
3. Paste the whole `kaggle_cell.py` content into **one cell** → Run.
   It automatically: installs deps → downloads MAESTRO v3 + **Mutopia archive** →
   tokenizes (~40M+ tokens) → pretrains the **~85M-param generator on both T4s**
   (DataParallel, bf16, FlashAttention, 6.5 h time budget) → trains the **critic** →
   runs **REINFORCE** RL fine-tuning → writes **`maestro_bundle.zip`** + preview MIDIs
   to Output. Fits one 12 h session; if interrupted, re-running the cell resumes
   from `ckpt.pt`.
4. Download `maestro_bundle.zip`, unzip, put `maestro_bundle.pt` into `~/music-bot/models/`

## Compose + render locally (fast CPU: int8 oneDNN/AVX2, 12 threads)

```bash
python -m musicbot all                 # 16 min piece: MIDI + WAV(fluidsynth) + MP3 + sheet
python -m musicbot generate --minutes 16 --candidates 6 --seed 7
python -m musicbot render              # re-render last MIDI
```

Audio uses **FluidSynth + FluidR3 concert-grand soundfont** when available
(installed here), falling back to the built-in numpy piano synth.

Local CPU fallback training (no Kaggle needed; measured 0.4k tok/s here vs
~50-100x on T4 x2):

```bash
python -m musicbot prepare --mutopia   # download + tokenize MAESTRO (+ Mutopia)
python -m musicbot train --steps 2000  # CPU pretrain
python -m musicbot rl-train            # critic + REINFORCE
python -m musicbot bench               # measure this machine's train/gen speed
```

## Windows exe (Iris Xe iGPU via DirectML)

1. Copy this folder to Windows (e.g. `C:\music-bot`) and `maestro_bundle.pt` into `models\`
2. In `windows\`, run `build_exe.bat` (one-time: installs torch-directml, builds exe)
3. Run `dist\MaestroComposer.exe [seed]` → composes 16 min on the iGPU, MIDI in `output\`

## Design notes (why it's fast here)

- int8 dynamic quantization (oneDNN) + 12 threads on the AVX2 CPU for generation
- KV-cache batched sampling, best-of-N candidates composed in parallel
- packed uint16 memmap corpus, zero-padding training windows
- SDPA (FlashAttention on Kaggle T4), bf16 autocast, RoPE + SwiGLU + RMSNorm
- chunked additive piano synth — no system packages touched, everything in `~/ml_env`
