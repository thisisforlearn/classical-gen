# ============================================================
#  Maestro tokenizer — REMI-style symbolic music <-> token ids
#  Author: thisisforlearn (sole author; final authority on this project)
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
from __future__ import annotations
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage

TICKS_PER_BEAT = 480
POS_PER_BEAT = 16
STEP = TICKS_PER_BEAT // POS_PER_BEAT
POS_PER_BAR = POS_PER_BEAT * 4

PAD, BOS, EOS, SEP = 0, 1, 2, 3
BAR = 4
POS0, N_POS = 5, POS_PER_BAR
TEMPO0, N_TEMPO = POS0 + N_POS, 12
TS0, N_TS = TEMPO0 + N_TEMPO, 4
PITCH0, N_PITCH = TS0 + N_TS, 88
DUR0, N_DUR = PITCH0 + N_PITCH, 32
VEL0, N_VEL = DUR0 + N_DUR, 16
FORM0, N_FORM = VEL0 + N_VEL, 8
VOCAB_SIZE = 256

FORMS = ["intro", "A", "A2", "B", "C", "dev", "recap", "coda"]
DUR_STEPS = list(range(1, 17)) + [18, 20, 24, 28, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384]
_TEMPOS = np.geomspace(40.0, 240.0, N_TEMPO)
TS_LIST = [(4, 4), (3, 4), (6, 8), (2, 4)]
PITCH_LO, PITCH_HI = 21, 108


def tempo_bin(bpm: float) -> int:
    return int(np.argmin(np.abs(_TEMPOS - float(bpm))))


def tempo_bpm(bin_id: int) -> float:
    return float(_TEMPOS[bin_id])


def dur_index(steps: int) -> int:
    return int(np.clip(np.searchsorted(DUR_STEPS, max(1, min(int(steps), DUR_STEPS[-1]))), 0, N_DUR - 1))


def _ts_index(num: int, den: int) -> int:
    for i, (n, d) in enumerate(TS_LIST):
        if (num, den) == (n, d):
            return i
    return 0


def encode_midi(path: str) -> np.ndarray:
    mid = MidiFile(path)
    notes, tempos, sigs = [], [], []
    for tr in mid.tracks:
        t = 0
        open_notes = {}
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                open_notes.setdefault((msg.channel, msg.note), []).append((t, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if open_notes.get(key):
                    on_t, vel = open_notes[key].pop(0)
                    if t > on_t:
                        notes.append((on_t, t, msg.note, vel))
            elif msg.type == "set_tempo":
                tempos.append((t, 60_000_000.0 / msg.tempo))
            elif msg.type == "time_signature":
                sigs.append((t, msg.numerator, msg.denominator))
    if not notes:
        return np.array([BOS, EOS], dtype=np.uint16)

    dur_of = {}
    for on_t, off_t, p, v in notes:
        dur_of.setdefault(on_t, []).append((off_t, p, v))
    tempos = [(t, tempo_bin(b)) for t, b in tempos] or [(0, tempo_bin(120.0))]
    sigs = [(t, _ts_index(n, d)) for t, n, d in sigs] or [(0, 0)]

    events = []
    for t, b in tempos:
        events.append((max(0, int(round(t / STEP))), 0, -1, (TEMPO0 + b,)))
    for t, s in sigs:
        events.append((max(0, int(round(t / STEP))), 0, -2, (TS0 + s,)))
    for on_t, lst in dur_of.items():
        q = max(0, int(round(on_t / STEP)))
        for off_t, p, v in sorted(lst, key=lambda x: -x[1]):
            if PITCH_LO <= p <= PITCH_HI:
                steps = max(1, int(round((off_t - on_t) / STEP)))
                events.append((q, 1, p, (PITCH0 + p - PITCH_LO, VEL0 + min(15, v >> 3), DUR0 + dur_index(steps))))
    events.sort(key=lambda e: (e[0], e[1], -e[2]))

    out = [BOS]
    cur_bar, cur_pos, cur_ts, cur_tempo = -1, -1, -1, -1
    for q, prio, _, toks in events:
        bar, pos = q // POS_PER_BAR, q % POS_PER_BAR
        if bar != cur_bar:
            out.append(BAR)
            cur_bar, cur_pos = bar, -1
        if pos != cur_pos:
            out.append(POS0 + pos)
            cur_pos = pos
        if prio == 1:
            out.extend(toks)
        else:
            tok = toks[0]
            if TEMPO0 <= tok < TS0:
                if tok != cur_tempo:
                    out.append(tok)
                    cur_tempo = tok
            elif TS0 <= tok < PITCH0:
                if tok != cur_ts:
                    out.append(tok)
                    cur_ts = tok
    out.append(EOS)
    return np.array(out, dtype=np.uint16)


def decode_tokens(tokens) -> MidiFile:
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    tr = MidiTrack()
    mid.tracks.append(tr)
    events = []
    cur_bar, cur_pos = -1, -1
    ts = (4, 4)
    pending = None
    pending_vel = 64

    def abs_step():
        return max(cur_bar, 0) * POS_PER_BAR + max(cur_pos, 0)

    for tok in [int(t) for t in tokens]:
        if tok == EOS:
            break
        if tok == BAR:
            cur_bar += 1
            cur_pos = -1
        elif POS0 <= tok < TEMPO0:
            cur_pos = tok - POS0
        elif TEMPO0 <= tok < TS0:
            events.append((abs_step(), 1, "tempo", tempo_bpm(tok - TEMPO0)))
        elif TS0 <= tok < PITCH0:
            ts = TS_LIST[tok - TS0]
            events.append((abs_step(), 2, "ts", ts))
        elif PITCH0 <= tok < DUR0:
            pending = tok - PITCH0 + PITCH_LO
        elif VEL0 <= tok < FORM0:
            pending_vel = min(127, (tok - VEL0) * 8 + 4)
        elif DUR0 <= tok < VEL0 and pending is not None:
            steps = DUR_STEPS[tok - DUR0]
            start = abs_step()
            events.append((start, 3, "on", (pending, pending_vel)))
            events.append((start + steps, 0, "off", pending))
            pending = None
    events.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for tick, _, kind, val in events:
        t = tick * STEP
        if kind == "on":
            tr.append(Message("note_on", note=val[0], velocity=val[1], time=t - last))
        elif kind == "off":
            tr.append(Message("note_off", note=val, velocity=0, time=t - last))
        elif kind == "tempo":
            tr.append(MetaMessage("set_tempo", tempo=int(60_000_000 / val), time=t - last))
        elif kind == "ts":
            tr.append(MetaMessage("time_signature", numerator=val[0], denominator=val[1], time=t - last))
        last = t
    return mid


def duration_sec_weighted(tokens) -> float:
    toks = [int(t) for t in tokens]
    bars = toks.count(BAR)
    tempos = [t - TEMPO0 for t in toks if TEMPO0 <= t < TS0]
    bpm = tempo_bpm(tempos[-1]) if tempos else 120.0
    return bars * 4.0 * 60.0 / bpm
