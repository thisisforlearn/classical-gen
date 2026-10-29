# ============================================================
#  Maestro sheet — MIDI -> MusicXML (+ PDF when MuseScore exists)
#  Author: Vaibhav Agarwal (@thisisforlearn) - sole author; final authority on this project
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
from __future__ import annotations
import shutil, subprocess
from pathlib import Path


def midi_to_musicxml(midi_path, out_path, log=print):
    from music21 import converter

    log("parsing midi with music21 (quantized)...")
    score = converter.parse(midi_path, forceSource=True, quantizePost=True, quarterLengthDivisors=(4, 3))
    score.write("musicxml", fp=str(out_path))
    log(f"wrote {out_path}")
    return out_path


def _find_musescore():
    for c in ("musescore4", "musescore3", "musescore", "mscore"):
        w = shutil.which(c)
        if w:
            return w
    for p in (
        "/mnt/c/Program Files/MuseScore 4/bin/MuseScore4.exe",
        "/mnt/c/Program Files/MuseScore 3/bin/MuseScore3.exe",
    ):
        if Path(p).exists():
            return p
    return None


def musicxml_to_pdf(xml_path, out_dir, log=print):
    ms = _find_musescore()
    if not ms:
        log("MuseScore not found — skipping PDF (open the .musicxml in MuseScore to print)")
        return None
    pdf = Path(out_dir) / (Path(xml_path).stem + ".pdf")
    subprocess.run([ms, "-o", str(pdf), str(xml_path)], check=True, capture_output=True, timeout=300)
    log(f"wrote {pdf}")
    return pdf
