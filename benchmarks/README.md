# Translation model benchmark (independent test)

This directory is an optional experiment. It does not modify LiveTranslator, its model directory, or the in-app update feed.

## Candidates

- QuickMT en→ko: https://huggingface.co/quickmt/quickmt-en-ko — CTranslate2/SentencePiece; about 361 MB of model weights; CC BY 4.0.
- Hy-MT2 1.8B Q4_K_M: https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF — llama.cpp CPU; about 1.13 GB; Apache 2.0.

Hy-MT2's official card mentions specialized GGUF kernel support. A failed llama.cpp load is a compatibility issue, not a measured translation quality result.

## Test using GitHub Actions

Open the GitHub Actions workflow 'Compare offline English-Korean translation models'. The workflow was configured for a single initial run when introduced, and can be triggered again manually.

The job downloads both models onto a temporary runner (not your laptop), attempts CPU translation of 16 original clean English sentences and 6 intentionally inaccurate ASR sentences, and uploads the translation-model-comparison artifact. Read the run Summary or report.md to compare outputs, chrF reference similarity, per-sentence latency and memory samples. Only result files are uploaded — never model weights.

## Optional local reproducibility

Use Python 3.12 and install ctranslate2, sentencepiece, huggingface_hub, sacrebleu, psutil. Then run:

    python benchmarks/compare_translation_models.py --engine quickmt --threads 4

For Hy-MT2, obtain a compatible official llama.cpp llama-server binary and run:

    python benchmarks/compare_translation_models.py --engine hymt --llama-server C:\\path\\to\\llama-server.exe --threads 4

Models are cached by Hugging Face; repeated runs do not deliberately redownload the entire model.

## Evaluation criteria

- Quality: preserve 30 versus 60 days, quantities, order/invoice terminology, natural Korean idioms.
- Speed: compare translation-only latency; LiveTranslator must also run Whisper.
- Memory: distinguish llama-server memory from the small benchmark Python process.
- The noisy-ASR examples have no reference score because mistranscribed English is semantically ambiguous.
- chrF from 16 original samples is not definitive. Review the resulting translations manually and repeat on the actual Windows CPU before changing the default engine.

License attribution and redistribution duties must be respected if either model is later packaged for others.


## Recognition quality profiles (v0.6.42)

`compare_recognition_profiles.py` replays one local recording through the actual
production pause/overlap/dedup/sentence helpers with base/3 seconds and small/5
seconds. It does not open microphones or transmit the recording; missing speech
models may be downloaded. Supply a plain English reference and an output path:

    python benchmarks/compare_recognition_profiles.py --audio recording.wav --reference reference.txt --output comparison.json

WAV input must be mono 16-bit PCM at 16000 Hz. Other formats use PyAV through
faster-whisper. Optional `--model-root` uses existing `base` and `small` folders.
Optional `--start-marker` and `--end-marker` exclude a recording's intro/outro
from scoring only, after recognition; they never supply missing words to ASR.
The reference must cover the same scoring region. JSON includes actual raw
recognition and assembled translation inputs. Keep recordings/references/results
out of git. The replay does not model live queue overflow or concurrent Hy-MT2
CPU contention; inspect those counters in the real Windows app as well.
