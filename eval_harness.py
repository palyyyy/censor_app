"""
Evaluation harness for the speech-censoring application.

Measures, across one or more Whisper model sizes:
  * detection accuracy   - of every target-word occurrence in the reference
                           sentences, how many the app catches (recall / hit
                           rate), misses, or wrongly flags (false positive)
  * model-size comparison - the same metrics for tiny.en / base.en / small.en
  * processing speed      - real-time factor (processing time / audio duration)

Live mode is simulated by slicing each clip into short chunks with a 0.25 s
overlap, feeding them through the engine's transcribe_chunk, and de-duplicating
exactly as the real LiveProcessor does:
  * --simulate-live          live path at a single chunk size (--chunk-seconds)
  * --chunk-sweep 1,2,4      live path at several chunk sizes in one run, which
                             produces the chunk-size / accuracy trade-off curve

It is run against a Mozilla Common Voice subset, whose clips each carry an exact
reference sentence, so the ground truth requires no manual labelling. Detection
uses the application's OWN WordMatcher, so the numbers reflect the real pipeline.

Run from the project root (so that `import app...` works):

    python eval_harness.py \
        --clips-dir cv-corpus/en/clips \
        --tsv       cv-corpus/en/validated.tsv \
        --models    tiny.en,base.en,small.en \
        --auto-targets 10 \
        --max-clips 350 \
        --chunk-sweep 1,2,4 \
        --out results

Self-tests (no app, no audio, no models needed):

    python eval_harness.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

_PUNCT = ".,!?;:'\"()[]{}"
LIVE_OVERLAP_S = 0.25          # matches LiveProcessor


def normalize(token: str) -> str:
    return token.strip().lower().strip(_PUNCT)


def count_targets(text: str, targets: set[str]) -> Counter:
    counts: Counter = Counter()
    for tok in text.split():
        n = normalize(tok)
        if n in targets:
            counts[n] += 1
    return counts


# ----------------------------------------------------------------------------
# Metric accumulation
# ----------------------------------------------------------------------------
class Tally:
    def __init__(self) -> None:
        self.tp = 0
        self.miss = 0
        self.fp = 0
        self.audio_s = 0.0
        self.proc_s = 0.0
        self.clips = 0

    def add_clip(self, ref: Counter, hyp: Counter) -> None:
        for word in set(ref) | set(hyp):
            r, h = ref.get(word, 0), hyp.get(word, 0)
            self.tp += min(r, h)
            self.miss += max(0, r - h)
            self.fp += max(0, h - r)

    @property
    def recall(self) -> float:
        d = self.tp + self.miss
        return self.tp / d if d else 0.0

    @property
    def miss_rate(self) -> float:
        d = self.tp + self.miss
        return self.miss / d if d else 0.0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def rtf(self) -> float:
        return self.proc_s / self.audio_s if self.audio_s else 0.0


# ----------------------------------------------------------------------------
# Live-mode simulation: mirrors LiveProcessor._stt_loop windowing + dedup.
# ----------------------------------------------------------------------------
def live_detect(audio, sr, transcribe_chunk, matcher_match,
                chunk_seconds=1.0, overlap_s=LIVE_OVERLAP_S):
    chunk_samples = int(chunk_seconds * sr)
    overlap_samples = int(overlap_s * sr)
    total = len(audio)
    last_end = 0
    detected: Counter = Counter()
    proc = 0.0
    windows = 0
    while last_end < total:
        window_end = min(last_end + chunk_samples, total)
        window_start = max(last_end - overlap_samples, 0)
        window = audio[window_start:window_end]
        t0 = time.perf_counter()
        words = transcribe_chunk(window, sr, window_start / sr)
        proc += time.perf_counter() - t0
        windows += 1
        for w in words:
            if w.end * sr < last_end - overlap_samples:
                continue                      # already handled in a prior window
            if matcher_match(w.text) is not None:
                detected[normalize(w.text)] += 1
        last_end = window_end
    return detected, proc, windows


# ----------------------------------------------------------------------------
# Common Voice loading / audio
# ----------------------------------------------------------------------------
def load_rows(tsv_path: Path, clips_dir: Path, max_clips: int):
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sentence = r.get("sentence", "").strip()
            name = r.get("path", "").strip()
            if not sentence or not name:
                continue
            clip = clips_dir / name
            if clip.exists():
                rows.append((clip, sentence))
            if len(rows) >= max_clips:
                break
    return rows


def auto_pick_targets(rows, n: int):
    freq: Counter = Counter()
    for _, sentence in rows:
        for tok in sentence.split():
            w = normalize(tok)
            if len(w) >= 4 and w.isalpha():
                freq[w] += 1
    return [w for w, _ in freq.most_common(n)]


def audio_duration_seconds(path: Path) -> float:
    from pydub import AudioSegment
    return len(AudioSegment.from_file(str(path))) / 1000.0


def load_audio_16k_mono(path: Path):
    """Mono float32 in [-1, 1] at 16 kHz, plus duration in seconds."""
    import numpy as np
    from pydub import AudioSegment
    seg = AudioSegment.from_file(str(path)).set_channels(1).set_frame_rate(16000)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    max_val = float(1 << (8 * seg.sample_width - 1))
    return samples / max_val, len(seg) / 1000.0


# ----------------------------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------------------------
def evaluate(args) -> None:
    from app.stt.registry import get_engine
    from app.stt import faster_whisper_engine  # noqa: F401  (registers engine)
    from app.censor.censor_rules import CensorList, CensorRule, CensorMode
    from app.censor.word_matcher import WordMatcher

    # Decide which live chunk sizes to run, if any.
    if args.chunk_sweep:
        chunk_sizes = [float(x) for x in args.chunk_sweep.split(",") if x.strip()]
    elif args.simulate_live:
        chunk_sizes = [args.chunk_seconds]
    else:
        chunk_sizes = []

    clips_dir = Path(args.clips_dir)
    rows = load_rows(Path(args.tsv), clips_dir, args.max_clips)
    if not rows:
        sys.exit("No clips found - check --clips-dir and --tsv paths.")

    if args.targets:
        targets = [normalize(w) for w in Path(args.targets).read_text().split() if w.strip()]
    else:
        targets = auto_pick_targets(rows, args.auto_targets)
    target_set = set(targets)
    print(f"Clips: {len(rows)}   Targets ({len(targets)}): {', '.join(targets)}")
    print(f"Live chunk sizes: {chunk_sizes if chunk_sizes else 'none (file mode only)'}\n")

    censor_list = CensorList()
    for w in targets:
        censor_list.add(CensorRule(word=w, mode=CensorMode.BEEP))
    matcher = WordMatcher(censor_list)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results = []   # (model, mode, chunk_or_None, tally)

    for model in models:
        print(f"=== model: {model} ===")
        engine = get_engine("faster-whisper", model=model, language="en")
        file_t = Tally()
        live_ts = {cs: Tally() for cs in chunk_sizes}

        for i, (clip, sentence) in enumerate(rows, 1):
            ref = count_targets(sentence, target_set)

            # ---- file mode (transcribe_file) ----
            try:
                dur = audio_duration_seconds(clip)
                t0 = time.perf_counter()
                transcript = engine.transcribe_file(clip)
                proc = time.perf_counter() - t0
            except Exception as e:
                print(f"  [skip] {clip.name}: {e}")
                continue
            hyp: Counter = Counter()
            for w in transcript.words:
                if matcher.match(w.text) is not None:
                    hyp[normalize(w.text)] += 1
            file_t.add_clip(ref, hyp)
            file_t.audio_s += dur
            file_t.proc_s += proc
            file_t.clips += 1

            # ---- live simulation at each chunk size (load audio once, reuse) ----
            if chunk_sizes:
                try:
                    audio16k, dur16 = load_audio_16k_mono(clip)
                except Exception as e:
                    print(f"  [live skip] {clip.name}: {e}")
                    audio16k = None
                if audio16k is not None:
                    for cs in chunk_sizes:
                        det, lproc, _ = live_detect(
                            audio16k, 16000, engine.transcribe_chunk,
                            matcher.match, chunk_seconds=cs)
                        lt = live_ts[cs]
                        lt.add_clip(ref, det)
                        lt.audio_s += dur16
                        lt.proc_s += lproc
                        lt.clips += 1

            if i % 25 == 0:
                print(f"  {i}/{len(rows)} clips...")

        try:
            engine.close()
        except Exception:
            pass

        results.append((model, "file", None, file_t))
        print(f"  [file]        recall={file_t.recall:.3f} miss={file_t.miss_rate:.3f} "
              f"prec={file_t.precision:.3f} RTF={file_t.rtf:.3f}")
        for cs in chunk_sizes:
            lt = live_ts[cs]
            results.append((model, "live", cs, lt))
            print(f"  [live {cs:>4.1f}s] recall={lt.recall:.3f} miss={lt.miss_rate:.3f} "
                  f"prec={lt.precision:.3f} RTF={lt.rtf:.3f}")
        print()

    write_outputs(Path(args.out), results, targets)
    print(f"Wrote CSVs, summary, and LaTeX tables to {args.out}/")


# ----------------------------------------------------------------------------
# Output: CSV + LaTeX tables (matching the thesis template's table style)
# ----------------------------------------------------------------------------
def _chunk_label(chunk) -> str:
    return "-" if chunk is None else f"{chunk:.1f}"


def write_outputs(out_dir: Path, results, targets) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "accuracy_by_model.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mode", "chunk_s", "occurrences", "detected", "missed",
                    "false_pos", "recall_pct", "miss_rate_pct", "precision_pct"])
        for model, mode, chunk, t in results:
            w.writerow([model, mode, _chunk_label(chunk), t.tp + t.miss, t.tp, t.miss, t.fp,
                        f"{t.recall*100:.1f}", f"{t.miss_rate*100:.1f}", f"{t.precision*100:.1f}"])

    with open(out_dir / "speed_by_model.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mode", "chunk_s", "clips", "audio_s", "proc_s", "mean_rtf"])
        for model, mode, chunk, t in results:
            w.writerow([model, mode, _chunk_label(chunk), t.clips,
                        f"{t.audio_s:.1f}", f"{t.proc_s:.1f}", f"{t.rtf:.3f}"])

    tex = []
    tex.append("\\begin{table}[h]")
    tex.append("\\caption{Detection accuracy by model size and mode}")
    tex.append("\\label{table:accuracy}")
    tex.append("\\centering")
    tex.append("\\begin{tabular}{ |l|l|r|r|r|r|r| }")
    tex.append(" \\hline")
    tex.append(" Model & Mode & Chunk (s) & Occurrences & Recall (\\%) & Miss (\\%) & Precision (\\%) \\\\")
    tex.append(" \\hline")
    for model, mode, chunk, t in results:
        tex.append(f" {model} & {mode} & {_chunk_label(chunk)} & {t.tp + t.miss} & "
                   f"{t.recall*100:.1f} & {t.miss_rate*100:.1f} & {t.precision*100:.1f} \\\\")
    tex.append(" \\hline")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")
    tex.append("")
    tex.append("\\begin{table}[h]")
    tex.append("\\caption{Processing speed by model size and mode (real-time factor)}")
    tex.append("\\label{table:speed}")
    tex.append("\\centering")
    tex.append("\\begin{tabular}{ |l|l|r|r|r|r| }")
    tex.append(" \\hline")
    tex.append(" Model & Mode & Chunk (s) & Clips & Audio (s) & Mean RTF \\\\")
    tex.append(" \\hline")
    for model, mode, chunk, t in results:
        tex.append(f" {model} & {mode} & {_chunk_label(chunk)} & {t.clips} & "
                   f"{t.audio_s:.1f} & {t.rtf:.3f} \\\\")
    tex.append(" \\hline")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")
    (out_dir / "tables.tex").write_text("\n".join(tex) + "\n")

    summary = [f"Target words: {', '.join(targets)}", "",
               f"{'model':<12}{'mode':<6}{'chunk':>6}{'occur':>7}{'recall%':>9}"
               f"{'miss%':>8}{'prec%':>8}{'RTF':>8}"]
    for model, mode, chunk, t in results:
        summary.append(f"{model:<12}{mode:<6}{_chunk_label(chunk):>6}{t.tp + t.miss:>7}"
                       f"{t.recall*100:>9.1f}{t.miss_rate*100:>8.1f}"
                       f"{t.precision*100:>8.1f}{t.rtf:>8.3f}")
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


# ----------------------------------------------------------------------------
# Self-tests
# ----------------------------------------------------------------------------
def selftest() -> None:
    from collections import namedtuple
    W = namedtuple("W", "text start end")

    # --- metric math ---
    t = Tally()
    t.add_clip(Counter({"banana": 2, "apple": 1}), Counter({"banana": 1, "apple": 1}))
    assert (t.tp, t.miss, t.fp) == (2, 1, 0)
    assert abs(t.recall - 2/3) < 1e-9 and abs(t.miss_rate - 1/3) < 1e-9
    t2 = Tally()
    t2.add_clip(Counter({"apple": 0}), Counter({"apple": 2}))
    assert (t2.tp, t2.miss, t2.fp) == (0, 0, 2)
    t2.audio_s, t2.proc_s = 10.0, 4.0
    assert abs(t2.rtf - 0.4) < 1e-9

    # --- live windowing + dedup ---
    sr = 1000
    audio = [0.0] * 2000                       # 2 windows of 1000 samples
    calls = {"n": 0}

    def fake_chunk(window, _sr, offset):
        calls["n"] += 1
        if calls["n"] == 1:
            return [W("apple", 0.1, 0.5)]
        return [W("apple", 0.6, 0.7), W("apple", 0.85, 0.9)]  # first deduped

    det, _, windows = live_detect(audio, sr, fake_chunk, lambda txt: True, chunk_seconds=1.0)
    assert windows == 2, windows
    assert det == Counter({"apple": 2}), det

    # --- larger chunk -> fewer windows ---
    calls["n"] = 0
    _, _, windows4 = live_detect([0.0] * 4000, sr, lambda *a: [], lambda txt: True, chunk_seconds=4.0)
    assert windows4 == 1, windows4

    # --- output with sweep rows renders ---
    res = [("tiny.en", "file", None, t)]
    for cs in (1.0, 2.0, 4.0):
        res.append(("tiny.en", "live", cs, t))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        write_outputs(Path(d), res, ["banana", "apple"])
        body = (Path(d) / "tables.tex").read_text()
        assert "Chunk (s)" in body and "live & 4.0" in body

    print("selftest OK (metrics + live windowing/dedup + sweep output)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--clips-dir")
    p.add_argument("--tsv")
    p.add_argument("--models", default="tiny.en,base.en,small.en")
    p.add_argument("--targets")
    p.add_argument("--auto-targets", type=int, default=8)
    p.add_argument("--max-clips", type=int, default=200)
    p.add_argument("--simulate-live", action="store_true",
                   help="live path at a single chunk size (--chunk-seconds)")
    p.add_argument("--chunk-seconds", type=float, default=1.0)
    p.add_argument("--chunk-sweep",
                   help="comma-separated chunk sizes for the live path, e.g. 1,2,4")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.clips_dir and args.tsv):
        p.error("--clips-dir and --tsv are required (or use --selftest)")
    evaluate(args)


if __name__ == "__main__":
    main()