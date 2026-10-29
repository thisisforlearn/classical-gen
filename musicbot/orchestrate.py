# ============================================================
#  Maestro orchestrator — piano reduction -> General MIDI orchestra
#  Auto-harmonize + instrument assignment + humanized performance
#  Author: Vaibhav (sole author; final authority on this project)
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
from __future__ import annotations
import random
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage

VIOLIN, VIOLA, CELLO, BASS = 40, 41, 42, 43
FLUTE, OBOE, HORN, TIMPANI = 73, 68, 60, 47
RANGES = {  # (low, high) comfortable GM ranges
    VIOLIN: (55, 96), VIOLA: (48, 81), CELLO: (36, 76), BASS: (28, 60),
    FLUTE: (60, 93), OBOE: (58, 88), HORN: (48, 74), TIMPANI: (36, 60),
}
MAJOR, MINOR = [0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10]
MAJOR_TRIADS = [[0, 4, 7], [2, 5, 9], [4, 7, 11], [5, 9, 12], [7, 11, 14], [9, 12, 16], [11, 14, 17]]
MINOR_TRIADS = [[0, 3, 7], [2, 5, 8], [3, 7, 10], [5, 8, 12], [7, 10, 14], [8, 12, 15], [10, 14, 17]]
K_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
K_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
SEATING = {  # CC10 pan (0=hard left, 64=center, 127=hard right) + CC91 reverb send
    "Violin": (24, 26), "Flute": (64, 40), "Viola": (48, 62), "Horn": (78, 72),
    "Cello": (98, 48), "Contrabass": (110, 55), "Timpani": (64, 82),
}
GM_NAME = {"Violin": VIOLIN, "Flute": FLUTE, "Viola": VIOLA, "Horn": HORN,
           "Cello": CELLO, "Contrabass": BASS, "Timpani": TIMPANI}
PRESETS = {
    "gm": {"Violin": 40, "Flute": 73, "Viola": 41, "Horn": 60, "Cello": 42, "Contrabass": 43, "Timpani": 47},
    "sonatina": {"Violin": 27, "Flute": 52, "Viola": 41, "Horn": 3, "Cello": 38, "Contrabass": 35, "Timpani": 23},
}


def _load_notes(midi_path):
    mid = MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    tempo, ts = 500000, (4, 4)
    on, off, notes = {}, {}, []
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo" and t == 0:
                tempo = msg.tempo
            elif msg.type == "time_signature" and t == 0:
                ts = (msg.numerator, msg.denominator)
            elif msg.type == "note_on" and msg.velocity > 0:
                on.setdefault(msg.note, []).append(t)
            elif msg.type in ("note_off", "note_on"):
                if on.get(msg.note):
                    s = on[msg.note].pop(0)
                    notes.append((s, t, msg.note, 80))
    notes.sort()
    return notes, tpb, tempo, ts


def _detect_key(pitches):
    hist = np.bincount([int(p) % 12 for p in pitches], minlength=12).astype(float)
    if hist.sum() == 0:
        return 0, False
    hist /= hist.sum()
    best = (-2, 0, True)
    for rot in range(12):
        for minor, prof in ((False, K_MAJ), (True, K_MIN)):
            r = np.corrcoef(hist, np.roll(prof, rot))[0, 1]
            if r > best[0]:
                best = (r, rot, minor)
    return best[1], best[2]


def _fit(pitch, lo, hi):
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return pitch


def orchestrate(midi_in, midi_out, seed=0, presets="gm", log=print):
    rng = random.Random(seed * 7919 + 3)
    notes, tpb, tempo, ts = _load_notes(midi_in)
    if not notes:
        raise ValueError(f"no notes in {midi_in}")
    end = max(n[1] for n in notes)
    bar_ticks = tpb * 4 * ts[0] // ts[1]
    n_bars = max(1, end // bar_ticks + 1)
    key_pc, minor = _detect_key([n[2] for n in notes])
    scale = MINOR if minor else MAJOR
    triads = MINOR_TRIADS if minor else MAJOR_TRIADS
    log(f"key: {'minor' if minor else 'major'} on {['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'][key_pc]} · {len(notes)} notes · {n_bars} bars")

    chord_of_bar = []
    last_deg = 0
    for b in range(n_bars):
        lo, hi = b * bar_ticks, (b + 1) * bar_ticks
        w = np.zeros(12)
        for (s, e, p, _) in notes:
            if lo <= s < hi:
                w[p % 12] += min(e, hi) - s
        best, best_s = last_deg, -1e9
        for d in range(7):
            ch = {(key_pc + t) % 12 for t in triads[d]}
            s = sum(w[pc] for pc in ch) / max(1.0, w.sum())
            s -= 0.08 if d == last_deg else 0.0
            if s > best_s:
                best, best_s = d, s
        if b % 8 == 7:
            best = 4
        chord_of_bar.append(best)
        last_deg = best

    tracks, last_t, prog_done = {}, {}, set()

    def track(name, ch, prog):
        tr = MidiTrack()
        pan, rev = SEATING[name]
        tr.append(MetaMessage("track_name", name=name, time=0))
        tr.append(Message("control_change", channel=ch, control=10, value=pan, time=0))
        tr.append(Message("control_change", channel=ch, control=91, value=rev, time=0))
        tracks[name] = (tr, ch, prog)
        return tr

    pmap = PRESETS[presets]
    for name, ch in (("Violin", 0), ("Flute", 1), ("Viola", 2),
                     ("Horn", 3), ("Cello", 4), ("Contrabass", 5), ("Timpani", 6)):
        track(name, ch, pmap[name])

    def play(name, pitch, t0, t1, vel, legato=0):
        tr, ch, prog = tracks[name]
        pitch = _fit(pitch, *RANGES[GM_NAME[name]])
        if ch not in prog_done:
            tr.append(Message("program_change", channel=ch, program=prog, time=0))
            last_t[ch] = 0
            prog_done.add(ch)
        jit = rng.randint(-3, 3)
        t0 = max(int(t0) + jit, last_t[ch])
        t1 = max(int(t1) + rng.randint(-2, 4) + legato, t0 + tpb // 8)
        vel = int(np.clip(vel + rng.randint(-6, 6), 18, 118))
        tr.append(Message("note_on", channel=ch, note=int(pitch), velocity=vel, time=t0 - last_t[ch]))
        last_t[ch] = t0
        tr.append(Message("note_off", channel=ch, note=int(pitch), velocity=0, time=t1 - t0))
        last_t[ch] = t1

    phrase = 0
    n_mel = len(notes)
    for i, (s, e, p, _) in enumerate(notes):
        if i > 0 and s - notes[i - 1][1] > tpb:
            phrase += 1
        pos_in_bar = (s % bar_ticks) / bar_ticks
        arc = 1.0 - 0.25 * pos_in_bar
        base = 66 + 30 * arc
        play("Violin", p, s, e, base, legato=tpb // 16)
        dur = e - s
        if dur >= bar_ticks // 2 or (i == 0):  # highlight long/first notes only
            play("Flute", p + 12, s, min(e, s + dur), 56, legato=0)

    for b in range(n_bars):
        lo = b * bar_ticks
        deg = chord_of_bar[b]
        root, third, fifth = (key_pc + t for t in triads[deg])
        play("Viola", third + 48, lo, lo + bar_ticks, 40)
        play("Horn", fifth + 43, lo, lo + bar_ticks, 38)
        play("Cello", root + 36, lo, lo + bar_ticks // 2, 58)
        play("Cello", (fifth if b % 2 else root) + 36, lo + bar_ticks // 2, lo + bar_ticks, 50)
        if b % 2 == 0:
            play("Contrabass", root + 24, lo, lo + 2 * bar_ticks, 54)
        if b % 4 == 0:
            play("Timpani", (key_pc + (7 if b % 16 == 8 else 0)) % 12 + 36,
                 lo, lo + bar_ticks // 4, 60 if b % 16 == 0 else 44)

    out = MidiFile(ticks_per_beat=tpb)
    for name in ("Violin", "Flute", "Viola", "Horn", "Cello", "Contrabass", "Timpani"):
        out.tracks.append(tracks[name][0])
    out.tracks[0].insert(0, MetaMessage("set_tempo", tempo=tempo, time=0))
    out.save(midi_out)
    log(f"orchestra MIDI: {midi_out}  (7 parts, humanized, seated + hall reverb)")
    return midi_out
