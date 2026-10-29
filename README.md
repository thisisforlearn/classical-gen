# MAESTRO — classical music composer bot

**Author: Vaibhav Agarwal** — [@thisisforlearn](https://github.com/thisisforlearn) (sole author; final authority on this project)
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

## Train on Kaggle (free GPU) — GitHub flow (recommended)

1. kaggle.com → New Notebook → Accelerator **GPU T4 x2** → Internet **On**
2. Paste **one cell**:

   ```
   !test -d classical-gen || git clone https://github.com/thisisforlearn/classical-gen
   !pip install -q mido rich
   !python classical-gen/kaggle/train_kaggle.py
   ```

   It automatically: installs deps → downloads MAESTRO v3 + **Mutopia archive** (parallel
   FTP crawler) → tokenizes (~28M tokens) → pretrains the **~85M-param generator on both
   T4s** (DataParallel, bf16, FlashAttention, EMA weights, label smoothing, dropout,
   live progress bars, 6.5 h time budget) → trains the **critic** → runs **REINFORCE** RL
   → writes **`maestro_bundle.zip`** + 3 preview MIDIs to Output.

3. **Never lose progress**: checkpoints (`ckpt.pt`, incl. EMA state) are saved every
   150 steps to `/kaggle/working`. If the session dies: click **Save Version**, then in a
   new session attach the previous notebook Output as an **Input** — the script
   auto-detects `ckpt.pt` / `maestro_bundle.pt` there and resumes exactly where it
   stopped (also reuses the tokenized corpus).

### Fallback: one-paste cell (no GitHub)

Run `python -m musicbot kaggle` → copy `models/kaggle_cell.py` content into one Kaggle
cell → Run. Same training, self-contained.

4. Download `maestro_bundle.zip`, unzip, put `maestro_bundle.pt` into `~/music-bot/models/`

## Compose + render locally (fast CPU: int8 oneDNN/AVX2, 12 threads)

```bash
python -m musicbot all                 # 16 min piece: MIDI + WAV(fluidsynth) + MP3 + sheet
python -m musicbot generate --minutes 16 --candidates 6 --seed 7
python -m musicbot render              # re-render last MIDI
```

Audio uses **FluidSynth** when available, falling back to the built-in numpy piano synth.

### Soundfont setup (one-time, ~470 MB — not stored in the repo)

The orchestral renderer uses the free **Sonatina Symphonic Orchestra** soundfont,
downloaded on demand (kept out of git so the repo stays small):

```bash
mkdir -p ~/music-bot/soundfonts && curl -L -o \
  ~/music-bot/soundfonts/Sonatina_Symphonic_Orchestra.sf2 \
  "https://huggingface.co/datasets/mileslilly/soundfonts/resolve/main/Sonatina_Symphonic_Orchestra.sf2"
```

Without it, rendering falls back to `MuseScore_General.sf2` or the system FluidR3
piano font.

Local CPU fallback training (no Kaggle needed; measured 0.4k tok/s here vs
~50-100x on T4 x2):

```bash
python -m musicbot prepare --mutopia   # download + tokenize MAESTRO (+ Mutopia)
python -m musicbot train --steps 2000  # CPU pretrain
python -m musicbot rl-train            # critic + REINFORCE
python -m musicbot bench               # measure this machine's train/gen speed
```

## Experiment Lab — tiny fast A/B training (find what actually works)

Instead of one 6.5 h run, the lab trains **tiny variants (~55 min each, one per GPU)**
with a **fixed token budget** so A/B comparisons are fair. Samples use **fixed seeds**
(same 3 seeds for every variant = paired listening), rendered with the same soundfont.
Everything is published to the public repo
**[hf.co/datasets/neko-i-am-a-cat/maestro-experiments](https://huggingface.co/datasets/neko-i-am-a-cat/maestro-experiments)**
(`samples/` zips, `results/` JSON, `ckpt/` weights).

Variants: `control` (base recipe) · `muon` (Muon optimizer on hidden layers —
modded-nanoGPT speedrun trick) · `data` (+GiantMIDI-Piano, 10,855 pieces) · `both`.

**Session 1** (GPU 0 builds data packs, GPU 1 trains the control):

```
!test -d classical-gen || git clone https://github.com/thisisforlearn/classical-gen
!pip install -q mido rich gdown huggingface_hub
!apt-get install -y -qq fluidsynth >/dev/null 2>&1
!curl -sL --create-dirs -o classical-gen/soundfonts/MuseScore_General.sf3 \
  "https://huggingface.co/datasets/mileslilly/soundfonts/resolve/main/MuseScore_General.sf3"
!cd classical-gen && { CUDA_VISIBLE_DEVICES=0 python kaggle/train_experiment.py prep > prep.log 2>&1 & \
  CUDA_VISIBLE_DEVICES=1 python kaggle/train_experiment.py run control > control.log 2>&1 & wait; }
!tail -3 classical-gen/*.log
```

**Session 2**: same cell but the last line runs `run muon` on GPU 0 and `run data` on GPU 1.
**Session 3**: `run both` + `run control 3000` (scaling check). Winners get stacked in later rounds.

Judging is **blind**: download the sample zips, listen, rate 1–10, then check `results/`
to see which variant won.

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
