"""Replay a recording with the live recognition and sentence assembly settings.

python benchmarks/compare_recognition_profiles.py --audio recording.wav \
    --reference reference.txt --output /tmp/asr-comparison.json

Models are downloaded once if absent. Audio/reference/results remain local.
The replay measures recognition accuracy and compute time, not live latency or
Windows device reliability. It does not simulate concurrent translation load.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import time
import types
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_audio_helpers():
    """Load the production pure functions without a Windows audio dependency."""
    wanted = {
        "SentenceBuffer", "QUALITY_OVERLAP_SECONDS", "_OVERLAP_WORD",
        "add_audio_context", "strip_transcript_overlap", "collapse_asr_restarts",
        "should_finalize_audio", "recognition_prompt", "_join_recognized_segments",
        "_split_long_english_batch",
    }
    source = ROOT / "streaming_pipeline.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in wanted for t in node.targets
        ):
            selected.append(node)
    namespace = {"np": np, "re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


def normalized_words(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace("favour", "favor").replace("okay", "ok"))


def score_words(reference, hypothesis):
    expected, actual = normalized_words(reference), normalized_words(hypothesis)
    if not expected:
        raise ValueError("Reference contains no English words")
    previous = list(range(len(actual) + 1))
    for i, word in enumerate(expected, 1):
        row = [i]
        for j, candidate in enumerate(actual, 1):
            row.append(min(row[-1] + 1, previous[j] + 1,
                           previous[j - 1] + (word != candidate)))
        previous = row
    return {"word_errors": previous[-1], "reference_words": len(expected),
            "wer": round(previous[-1] / len(expected), 4)}


def score_region(text, start_marker=None, end_marker=None):
    start = text.casefold().find(start_marker.casefold()) if start_marker else 0
    if start < 0:
        raise ValueError("Start marker missing from recognition; inspect results")
    end = text.casefold().find(end_marker.casefold(), start) if end_marker else len(text)
    if end < 0:
        raise ValueError("End marker missing from recognition; inspect results")
    return text[start:end + len(end_marker)] if end_marker else text[start:]


def read_audio(path):
    if path.suffix.lower() != ".wav":
        from faster_whisper.audio import decode_audio
        return decode_audio(str(path), sampling_rate=16000)
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16000):
            raise ValueError("For WAV use mono 16-bit PCM at 16000 Hz")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def replay(audio, model, seconds, helpers, *, quality_priority=False):
    chunks, parts = [], []
    count = quiet = 0
    tail = np.empty(0, dtype=np.float32)
    for at in range(0, len(audio), 1600):
        piece = audio[at:at + 1600]
        parts.append(piece)
        count += piece.size
        quiet = quiet + piece.size if np.sqrt(np.mean(piece * piece)) < 0.0015 else 0
        if helpers.should_finalize_audio(count, quiet, 16000, seconds, True):
            block, tail = helpers.add_audio_context(np.concatenate(parts), tail, 16000, True)
            cut = quiet < int(16000 * 0.28)
            chunks.append((block, cut, (at + len(piece)) / 16000))
            parts, count, quiet = [], 0, 0
    if parts:
        block, _ = helpers.add_audio_context(np.concatenate(parts), tail, 16000, True)
        chunks.append((block, False, len(audio) / 16000))
    originals, inputs = [], []
    last = recent = ""
    assembler = helpers.SentenceBuffer(seconds)
    compute_seconds = 0.0
    for block, cut, at in chunks:
        if np.sqrt(np.mean(block * block)) < 0.0005:
            last = recent = ""
            item = assembler.flush()
            if item:
                inputs.extend(helpers._split_long_english_batch(item[0]))
            continue
        before = time.monotonic()
        segments, _ = model.transcribe(
            block, language="en", beam_size=5, best_of=1, temperature=0,
            initial_prompt=helpers.recognition_prompt(recent, "en", True, quality_priority),
            without_timestamps=True,
            condition_on_previous_text=False, vad_filter=False,
        )
        raw = helpers._join_recognized_segments(segments, "en", True)
        compute_seconds += time.monotonic() - before
        if not raw:
            continue
        cleaned = helpers.collapse_asr_restarts(raw)
        original = helpers.strip_transcript_overlap(last, cleaned, "en")
        last = cleaned
        if original:
            recent = original[-180:]
            originals.append(original)
            for text, _ in assembler.add(original, "en", at, cut_mid_speech=cut):
                inputs.extend(helpers._split_long_english_batch(text))
    item = assembler.flush()
    if item:
        inputs.extend(helpers._split_long_english_batch(item[0]))
    return {"recognized": originals, "translation_inputs": inputs,
            "asr_compute_seconds": round(compute_seconds, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--start-marker")
    parser.add_argument("--end-marker")
    parser.add_argument("--compare-prompt", action="store_true",
                        help="Compare previous-text prompting on/off using small/5s in both runs")
    args = parser.parse_args()
    from faster_whisper import WhisperModel
    audio = read_audio(args.audio)
    reference = args.reference.read_text(encoding="utf-8-sig")
    helpers = load_audio_helpers()
    results = []
    profiles = (("small", 5, False), ("small", 5, True)) if args.compare_prompt else (
        ("base", 3, False), ("small", 5, True),
    )
    for name, seconds, quality_priority in profiles:
        model_path = str(args.model_root / name) if args.model_root else name
        model = WhisperModel(model_path, device="cpu", compute_type="int8", cpu_threads=6)
        result = replay(audio, model, seconds, helpers, quality_priority=quality_priority)
        hypothesis = score_region(" ".join(result["recognized"]), args.start_marker, args.end_marker)
        result.update(score_words(reference, hypothesis))
        result.update(model=name, capture_seconds=seconds, previous_text_prompt=not quality_priority,
                      audio_seconds=round(len(audio) / 16000, 2))
        results.append(result)
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in results:
        print(result["model"], f"previous_text_prompt={result['previous_text_prompt']}",
              result["wer"], result["asr_compute_seconds"])


if __name__ == "__main__":
    main()
