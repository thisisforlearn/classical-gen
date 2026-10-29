# ============================================================
#  Maestro synth — chunked additive piano renderer (numpy, zero system deps)
#  Author: Vaibhav (sole author; final authority on this project)
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
from __future__ import annotations
import subprocess, shutil, wave
from pathlib import Path

import numpy as np
from mido import MidiFile

SR = 44100


def _note_events(midi_path):
    mid = MidiFile(midi_path)
    tempo_map = [(0, 500000)]
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo":
                tempo_map.append((t, msg.tempo))
    tempo_map.sort()

    def tick2sec(tick):
        sec, last_t, last_tp = 0.0, 0, 500000
        for tt, tp in tempo_map:
            if tt >= tick:
                break
            sec += (tt - last_t) * last_tp / 1e6 / mid.ticks_per_beat
            last_t, last_tp = tt, tp
        sec += (tick - last_t) * last_tp / 1e6 / mid.ticks_per_beat
        return sec

    notes = []
    for tr in mid.tracks:
        t = 0
        open_n = {}
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                open_n.setdefault(msg.note, []).append((t, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if open_n.get(msg.note):
                    on_t, vel = open_n[msg.note].pop(0)
                    notes.append((tick2sec(on_t), tick2sec(t), msg.note, vel))
    notes.sort()
    return notes


def _find_sf2():
    for p in (
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/sounds/sf2/default-GM.sf2",
        "/usr/share/sounds/sf2/TimGM6mb.sf2",
    ):
        if Path(p).exists():
            return p
    return None


def render_wav(midi_path, wav_path, log=print):
    fs = shutil.which("fluidsynth")
    sf2 = _find_sf2()
    if fs and sf2:
        log(f"fluidsynth render ({Path(sf2).name})...")
        subprocess.run(
            [fs, "-ni", "-g", "0.7", "-r", str(SR), "-F", str(wav_path), sf2, str(midi_path)],
            check=True, capture_output=True, timeout=180,
        )
        log(f"wrote {wav_path}")
        return wav_path
    log("fluidsynth/soundfont not found — using built-in additive piano synth")
    return _render_wav_builtin(midi_path, wav_path, log)


def _render_wav_builtin(midi_path, wav_path, log=print):
    notes = _note_events(midi_path)
    if not notes:
        raise RuntimeError("no notes in midi")
    total = notes[-1][1] + 3.0
    tmp = Path(str(wav_path) + ".f32.tmp")
    mono = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(int(total * SR) + SR,))
    rng = np.random.default_rng(0)
    pbar = _progress(len(notes), "synthesizing piano")
    for i, (t0, t1, pitch, vel) in enumerate(notes):
        f = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
        veln = vel / 127.0
        dur = max(0.08, t1 - t0)
        tail = min(2.5, 0.4 + 2.0 * (220.0 / f) ** 0.5)
        n = int((dur + tail) * SR)
        if n < 8:
            continue
        s0 = int(t0 * SR)
        if s0 >= len(mono):
            continue
        n = min(n, len(mono) - s0)
        t = np.arange(n, dtype=np.float32) / SR
        tau = float(np.clip(4.2 * (220.0 / f) ** 0.6, 0.5, 9.0))
        env = np.minimum(t / 0.006, 1.0) ** 2
        env *= np.exp(-t / tau)
        rel = t > dur
        env[rel] *= np.exp(-(t[rel] - dur) / 0.18)
        K = int(min(10, 18000.0 / f))
        bright = 0.35 + 0.65 * veln
        wavef = np.zeros(n, dtype=np.float32)
        ph = rng.uniform(0, 2 * np.pi, K)
        for k in range(1, K + 1):
            fk = f * k * np.sqrt(1.0 + (0.0004 * k) ** 2)
            if fk > 19000:
                break
            wavef += (k ** (-1.5 - (1.0 - bright) * 1.3)) * np.sin(2 * np.pi * fk * t + ph[k - 1])
        pan = np.clip((pitch - 60) / 48.0, -1, 1)
        gain = 0.32 * veln ** 1.3
        mono[s0 : s0 + n] += gain * env * wavef * (1.0 - 0.5 * max(0.0, pan))
        pbar.update(1)
    peak = float(np.abs(mono[: int(total * SR)]).max()) + 1e-9
    scale = 0.92 / peak if peak > 0.92 else 1.0
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        CH = SR * 10
        for off in range(0, int(total * SR), CH):
            seg = np.clip(np.asarray(mono[off : off + CH], dtype=np.float32) * scale, -1.0, 1.0)
            st = np.stack([seg, seg], axis=1)
            w.writeframes((st * 32767).astype(np.int16).tobytes())
    del mono
    tmp.unlink()
    pbar.close()
    log(f"wrote {wav_path}")
    return wav_path


def _progress(total, desc):
    class _P:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

        def update(self, n=1):
            if self.task is not None:
                self.prog.update(self.task, advance=n)

        def close(self):
            if self.prog is not None:
                self.prog.stop()

    try:
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

        p = _P()
        p.prog = Progress(TextColumn("[cyan]" + desc), BarColumn(bar_width=40), TimeElapsedColumn())
        p.task = None
        p.prog.start()
        p.task = p.prog.add_task(desc, total=total)
        return p
    except Exception:
        p = _P()
        p.prog = None
        p.task = None
        return p


def maybe_mp3(wav_path, out_dir, log=print):
    mp3 = Path(out_dir) / (Path(wav_path).stem + ".mp3")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-q:a", "2", str(mp3)],
                       check=True, capture_output=True)
        log(f"wrote {mp3}")
        return mp3
    except Exception:
        return None
