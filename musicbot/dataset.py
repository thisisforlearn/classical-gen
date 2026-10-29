# ============================================================
#  Maestro dataset — MAESTRO v3 download, parallel tokenize, packed memmap
#  Author: thisisforlearn (sole author; final authority on this project)
#  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
# ============================================================
from __future__ import annotations
import json, os, zipfile, urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    from musicbot.tokenizer import encode_midi, EOS
except ImportError:
    from tokenizer import encode_midi, EOS

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"


def download_maestro(data_dir: str, log=print) -> Path:
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    z = d / "maestro-v3.0.0-midi.zip"
    root = d / "maestro-v3.0.0"
    if root.exists() and any(root.rglob("*.midi")):
        return root
    if not z.exists():
        log(f"downloading MAESTRO v3 MIDI (~58 MB) -> {z}")

        def hook(n, bs, total):
            if n % 200 == 0:
                log(f"  {min(n * bs, total) / 1e6:.1f} / {total / 1e6:.1f} MB")

        urllib.request.urlretrieve(MAESTRO_URL, z, reporthook=hook)
    if not root.exists():
        log("extracting...")
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
    return root


def _encode_one(path):
    try:
        return encode_midi(path)
    except Exception:
        return None


MUTOPIA_FTP = "https://www.mutopiaproject.org/ftp/"


def _http_get(url, timeout=30, retries=2):
    import time as _t

    for i in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if i == retries:
                raise
            _t.sleep(1.5)


def _crawl_mutopia(limit_composers=None):
    import re
    from concurrent.futures import ThreadPoolExecutor

    html = _http_get(MUTOPIA_FTP).decode("utf-8", "ignore")
    composers = sorted(set(re.findall(r'href="([A-Za-z0-9~\-\.]+)/"', html)))
    if limit_composers:
        composers = composers[:limit_composers]
    urls = []

    def scan_composer(c):
        out = []
        try:
            ch = _http_get(MUTOPIA_FTP + c + "/").decode("utf-8", "ignore")
            out += [MUTOPIA_FTP + c + "/" + m for m in re.findall(r'href="([^"]+\.midi?)"', ch, re.I)]
            pieces = sorted(set(re.findall(r'href="([^"/?]+)/"', ch)))
            pieces = [p for p in pieces if p not in ("ftp",) and not p.startswith("?")]

            def scan_piece(pc):
                try:
                    ph = _http_get(MUTOPIA_FTP + c + "/" + pc + "/").decode("utf-8", "ignore")
                    return [MUTOPIA_FTP + c + "/" + pc + "/" + m
                            for m in re.findall(r'href="([^"]+\.midi?)"', ph, re.I)]
                except Exception:
                    return []

            with ThreadPoolExecutor(12) as ex:
                for r in ex.map(scan_piece, pieces):
                    out += r
        except Exception:
            pass
        return out

    with ThreadPoolExecutor(24) as ex:
        for r in ex.map(scan_composer, composers):
            urls += r
    return sorted(set(urls))


def download_mutopia(data_dir: str, log=print, limit_composers=None) -> Path:
    d = Path(data_dir) / "mutopia"
    d.mkdir(parents=True, exist_ok=True)
    marker = d / ".done"
    if marker.exists():
        return d
    log("crawling Mutopia FTP index (~2100 pieces)...")
    urls = _crawl_mutopia(limit_composers)
    log(f"found {len(urls)} midi files — downloading in parallel...")
    from concurrent.futures import ThreadPoolExecutor
    import urllib.request as _ur

    def fetch(u):
        dest = d / u.split("/ftp/")[1].split("/")[-2]
        dest.mkdir(exist_ok=True)
        f = dest / u.split("/")[-1]
        if f.exists() and f.stat().st_size > 0:
            return 1
        try:
            req = _ur.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with _ur.urlopen(req, timeout=60) as r:
                if int(r.headers.get("Content-Length", 0)) > 2_000_000:
                    return 0
                f.write_bytes(r.read())
            return 1
        except Exception:
            return 0

    ok = 0
    with ThreadPoolExecutor(24) as ex:
        for i, r in enumerate(ex.map(fetch, urls)):
            ok += r
            if (i + 1) % 250 == 0:
                log(f"  {i + 1}/{len(urls)} ({ok} ok)")
    marker.write_text(str(ok))
    log(f"mutopia: {ok} midi files saved")
    return d


def tokenize_corpus(midi_root, out_dir: str, workers: int = None, log=print, extra_roots=None):
    if isinstance(midi_root, (list, tuple)):
        midi_root, extra_roots = midi_root[0], list(midi_root[1:]) + list(extra_roots or [])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_npz = out / "packed.npy"
    meta_json = out / "meta.json"
    if data_npz.exists() and meta_json.exists():
        log("tokenized corpus already exists, skipping")
        return json.loads(meta_json.read_text())
    files = []
    for root in [midi_root] + list(extra_roots or []):
        files += sorted(str(p) for p in Path(root).rglob("*.midi"))
        files += sorted(str(p) for p in Path(root).rglob("*.mid"))
        files += sorted(str(p) for p in Path(root).rglob("*.MID"))
    files = list(dict.fromkeys(files))
    log(f"tokenizing {len(files)} MIDI files on {workers or os.cpu_count()} workers...")
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    seqs = []
    with Pool(workers) as pool:
        for i, s in enumerate(pool.imap_unordered(_encode_one, files, chunksize=8)):
            if s is not None and len(s) > 64:
                seqs.append(s)
            if (i + 1) % 100 == 0:
                log(f"  {i + 1}/{len(files)}")
    total = sum(len(s) for s in seqs) + len(seqs)
    log(f"{len(seqs)} pieces, {total / 1e6:.2f}M tokens -> packing")
    arr = np.memmap(data_npz, dtype=np.uint16, mode="w+", shape=(total,))
    off = 0
    for s in seqs:
        arr[off : off + len(s)] = s
        off += len(s)
        arr[off] = EOS
        off += 1
    arr.flush()
    meta = {"n_tokens": total, "n_pieces": len(seqs)}
    meta_json.write_text(json.dumps(meta))
    return meta


class PackedData:
    def __init__(self, out_dir: str, ctx: int):
        self.data = np.memmap(Path(out_dir) / "packed.npy", dtype=np.uint16, mode="r")
        meta = json.loads((Path(out_dir) / "meta.json").read_text())
        self.n_tokens, self.n_pieces = meta["n_tokens"], meta["n_pieces"]
        self.ctx = ctx

    def sample_batch(self, bs: int, rng: np.random.Generator):
        hi = self.n_tokens - self.ctx - 1
        starts = rng.integers(0, hi, size=bs)
        idx = starts[:, None] + np.arange(self.ctx + 1)[None, :]
        buf = np.asarray(self.data[idx], dtype=np.int64)
        return buf[:, :-1], buf[:, 1:]
