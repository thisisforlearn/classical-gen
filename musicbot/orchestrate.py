# ============================================================
#  Maestro orchestrator — piano reduction -> General MIDI orchestra
#  Auto-harmonizes the monophonic line and assigns orchestra parts.
#  Author: Vaibhav (sole author; final authority on this project)
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
from __future__ import annotations
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


def orchestrate(midi_in, midi_out, log=print):
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
        pcs = {n[2] % 12 for n in notes if lo <= n[0] < hi}
        best, best_s = last_deg, -1
        for d in range(7):
            ch = {(key_pc + t) % 12 for t in triads[d]}
            s = len(pcs & ch) - (0.6 if d == last_deg else 0.0)
            if s > best_s:
                best, best_s = d, s
        if (b % 8) == 7:
            best = 4 if not minor else 4
        chord_of_bar.append(best)
        last_deg = best

    prog_done, last_t = set(), {}

    def note_on(tr, ch, prog, pitch, t0, t1, vel):
        pitch = _fit(pitch, *RANGES[prog])
        if ch not in prog_done:
            tr.append(Message("program_change", channel=ch, program=prog, time=0))
            last_t[ch] = 0
            prog_done.add(ch)
        t0 = max(int(t0), last_t[ch])
        t1 = max(int(t1), t0 + tpb // 8)
        tr.append(Message("note_on", channel=ch, note=int(pitch), velocity=int(vel), time=t0 - last_t[ch]))
        last_t[ch] = t0
        tr.append(Message("note_off", channel=ch, note=int(pitch), velocity=0, time=t1 - t0))
        last_t[ch] = t1

    tracks = {}

    def track(name, ch, prog):
        tr = MidiTrack()
        tr.append(MetaMessage("track_name", name=name, time=0))
        tracks[name] = (tr, ch, prog)
        return tr

    track("Violin", 0, VIOLIN)
    track("Flute", 1, FLUTE)
    track("Viola", 2, VIOLA)
    track("Horn", 3, HORN)
    track("Cello", 4, CELLO)
    track("Contrabass", 5, BASS)
    track("Timpani", 6, TIMPANI)

    phrase = 0
    for i, (s, e, p, _) in enumerate(notes):
        note_on(tracks["Violin"][0], 0, VIOLIN, p, s, e, max(40, min(110, int(70 + (p % 12)))))
        if i > 0 and s - notes[i - 1][1] > tpb:  # rest = phrase boundary
            phrase += 1
        if phrase % 2 == 0:
            note_on(tracks["Flute"][0], 1, FLUTE, p + 12, s, e, 62)

    for b in range(n_bars):
        lo = b * bar_ticks
        deg = chord_of_bar[b]
        root = key_pc + triads[deg][0]
        third = key_pc + triads[deg][1]
        fifth = key_pc + triads[deg][2]
        note_on(tracks["Viola"][0], 2, VIOLA, third + 48, lo, lo + bar_ticks, 52)
        note_on(tracks["Horn"][0], 3, HORN, fifth + 43, lo, lo + bar_ticks, 48)
        note_on(tracks["Cello"][0], 4, CELLO, root + 36, lo, lo + bar_ticks // 2, 68)
        note_on(tracks["Cello"][0], 4, CELLO, root + 36, lo + bar_ticks // 2, lo + bar_ticks, 60)
        note_on(tracks["Contrabass"][0], 5, BASS, root + 24, lo, lo + bar_ticks, 62)
        if b % 4 == 0:
            note_on(tracks["Timpani"][0], 6, TIMPANI, (key_pc + (7 if b % 16 == 8 else 0)) % 12 + 36,
                    lo, lo + bar_ticks // 4, 64 if b % 16 == 0 else 48)

    out = MidiFile(ticks_per_beat=tpb)
    for name in ("Violin", "Flute", "Viola", "Horn", "Cello", "Contrabass", "Timpani"):
        tr, _, _ = tracks[name]
        out.tracks.append(tr)
    out.tracks[0].insert(0, MetaMessage("set_tempo", tempo=tempo, time=0))
    out.save(midi_out)
    log(f"orchestra MIDI: {midi_out}  (7 parts)")
    return midi_out
