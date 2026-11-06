#!/usr/bin/env python3
"""Standalone CPU model comparison; never changes LiveTranslator or its models.

Install: pip install ctranslate2 sentencepiece huggingface_hub sacrebleu psutil
Run: python benchmarks/compare_translation_models.py --engine quickmt
For Hy-MT2 also provide a recent llama-server binary:
     python benchmarks/compare_translation_models.py --engine hymt --llama-server PATH
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from urllib import error as urlerror
from urllib import request as urlrequest

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "en_ko_cases.json"
OUTPUT_DIR = ROOT / "results"
HY_REPO = "tencent/Hy-MT2-1.8B-GGUF"
HY_FILENAME = "Hy-MT2-1.8B-Q4_K_M.gguf"
QUICK_REPO = "quickmt/quickmt-en-ko"


def quickmt_model(threads):
    """Match quickmt's actual SentencePiece tokenization (including EOS)."""
    import ctranslate2
    import sentencepiece as spm
    from huggingface_hub import snapshot_download

    files = ["model.bin", "config.json", "source_vocabulary.json",
             "target_vocabulary.json", "src.spm.model", "tgt.spm.model"]
    model_dir = snapshot_download(QUICK_REPO, allow_patterns=files)
    enc = spm.SentencePieceProcessor(model_file=str(Path(model_dir) / "src.spm.model"))
    dec = spm.SentencePieceProcessor(model_file=str(Path(model_dir) / "tgt.spm.model"))
    translator = ctranslate2.Translator(
        model_dir, device="cpu", compute_type="int8",
        inter_threads=1, intra_threads=threads
    )

    def translate(text):
        tokens = enc.encode(text, out_type=str) + ["</s>"]
        result = translator.translate_batch(
            [tokens], beam_size=5, max_decoding_length=192,
            disable_unk=True
        )
        return dec.decode(result[0].hypotheses[0]).strip()

    return translate, {"model": QUICK_REPO, "quantization": "int8",
                       "backend": "CTranslate2", "source": model_dir}


def json_post(url, payload, timeout=130):
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        url, data=encoded, headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urlrequest.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read(2_000_000).decode("utf-8"))


@contextlib.contextmanager
def hymt_model(server_path, threads, port):
    """Run the actual CPU llama.cpp model locally via an HTTP-only loopback."""
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(HY_REPO, filename=HY_FILENAME)
    executable = Path(server_path).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(
            f"llama-server binary not found: {executable}. "
            "Install/build llama.cpp first; do not use an unknown downloaded executable."
        )
    args = [
        str(executable), "--model", model_path, "--host", "127.0.0.1",
        "--port", str(port), "--threads", str(threads), "--ctx-size", "2048",
        "--n-gpu-layers", "0", "--parallel", "1"
    ]
    log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    server = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    try:
        health_url = f"http://127.0.0.1:{port}/health"
        ready = False
        for _ in range(120):
            if server.poll() is not None:
                break
            try:
                with urlrequest.urlopen(health_url, timeout=1) as r:
                    data = json.loads(r.read(16384))
                    if r.status == 200 and data.get("status") in ("ok", "ready"):
                        ready = True
                        break
            except (OSError, ValueError, urlerror.HTTPError):
                pass
            time.sleep(1)
        if not ready:
            log.seek(0)
            raise RuntimeError(
                "Hy-MT2 llama-server failed to start. Log tail:\n" +
                log.read()[-3500:]
            )

        def translate(text):
            request = {
                "model": HY_FILENAME,
                "messages": [{
                    "role": "user",
                    "content":
                        "Translate the following English text into natural Korean. "
                        "Preserve all numbers, amounts and proper names faithfully. "
                        "Output only the translation, without explanations.\n\n" + text
                }],
                "temperature": 0,
                "max_tokens": 180,
                "stream": False
            }
            obj = json_post(f"http://127.0.0.1:{port}/v1/chat/completions", request)
            return obj["choices"][0]["message"]["content"].strip()

        yield translate, {
            "model": HY_REPO, "quantization": "Q4_K_M",
            "backend": "llama.cpp llama-server CPU",
            "source": model_path
        }
    finally:
        server.terminate()
        try:
            server.wait(timeout=8)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=4)
        log.close()


def memory_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1024**2, 1)
    except ImportError:
        return None


def run_cases(translate, name, model_info, data, limit):
    from sacrebleu.metrics import CHRF
    clean = data["clean"][:limit] if limit else data["clean"]
    noisy = data["noisy"][:max(3, min(6, limit))] if limit else data["noisy"]
    results = []
    for category, cases in [("clean", clean), ("noisy", noisy)]:
        for case in cases:
            begin = time.perf_counter()
            result = translate(case["en"])
            elapsed = round(time.perf_counter() - begin, 3)
            item = {
                "group": category,
                "id": case["id"],
                "input": case["en"],
                "output": result,
                "seconds": elapsed,
                "reference": case.get("ko")
            }
            results.append(item)
            print(f"{name} [{category}/{case['id']}] {elapsed:.2f}s", flush=True)
            print(f"  EN: {case['en']}", flush=True)
            print(f"  KO: {result}", flush=True)

    clean_rows = [item for item in results if item["group"] == "clean"]
    scored = [r for r in clean_rows if r["reference"]]
    chrf = CHRF(word_order=0).corpus_score(
        [r["output"] for r in scored],
        [[r["reference"] for r in scored]]
    ).score
    all_seconds = [r["seconds"] for r in results]
    metrics = {
        "chrF": round(chrf, 2),
        "clean_case_count": len(clean_rows),
        "noisy_case_count": len(results) - len(clean_rows),
        "mean_seconds": round(statistics.mean(all_seconds), 3),
        "p50_seconds": round(statistics.median(all_seconds), 3),
        "max_seconds": max(all_seconds),
        "memory_mb_process_after": memory_mb(),
        "hardware": {
            "cpu_count": os.cpu_count(),
            "platform": sys.platform
        },
        "note": "chrF on 16 original sentences is a small-sample proxy, not professional adequacy. Runner CPU is not the user's Windows laptop."
    }
    return {"engine": name, "model_info": model_info,
            "metrics": metrics, "cases": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=("quickmt", "hymt"), required=True)
    p.add_argument("--llama-server", default="")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--port", type=int, default=18081)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = p.parse_args()
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{args.engine}.json"
    try:
        if args.engine == "quickmt":
            translate, info = quickmt_model(args.threads)
            result = run_cases(translate, args.engine, info, data, args.limit)
        else:
            if not args.llama_server:
                p.error("--llama-server is required with --engine hymt")
            with hymt_model(args.llama_server, args.threads, args.port) as (translate, info):
                result = run_cases(translate, args.engine, info, data, args.limit)
        result["success"] = True
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("BENCHMARK_RESULT " + json.dumps(result["metrics"], ensure_ascii=False), flush=True)
    except Exception as exc:
        result = {
            "engine": args.engine, "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-6000:]
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("BENCHMARK_FAILED " + result["error"], flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
