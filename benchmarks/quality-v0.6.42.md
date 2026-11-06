# v0.6.42 recognition quality comparison

Source: British Council, *A phone call from a customer*:
https://learnenglish.britishcouncil.org/free-resources/listening/b1/phone-call-customer

Executed 2026-11-06. Input: official MP3, converted to mono 16-bit PCM WAV at
16000 Hz. Both configurations replayed the same 146.25-second recording from
its beginning. Scoring covers recognized text from the first `Hello` to the
first `Goodbye`; intro/outro remain present during ASR but are excluded from WER.
The reference is the site's conversation transcript with speaker labels removed.
No reference words, names or vocabulary hints were provided to the recognizer.
Audio/reference/model files are not committed.

Command (paths are local inputs):

```sh
python benchmarks/compare_recognition_profiles.py --audio recording.wav --reference reference.txt --model-root model-folder --start-marker Hello --end-marker Goodbye --output comparison.json
```

| Recognizer | Minimum capture | Word errors / reference words | WER | ASR compute time |
|---|---:|---:|---:|---:|
| base | 3 s | 39/279 | 13.98% | 21.32 s |
| small | 5 s | 27/279 | 9.68% | 36.63 s |

Both use CPU INT8, six threads, beam five without artificial backlog, the
production 0.9-second overlap, pause rule, deduplication and sentence assembler.
The replay flushes its final partial audio/text for inspection. Live cancellation
and capture-driver behavior are not simulated. Normalization lowercases English,
ignores punctuation and equates favour/favor and okay/OK. WER counts word edits;
it does not grade translation meaning or Korean fluency.

The quality profile reduced word edits from 39 to 27 in this run (about 31%).
It preserved the 60-day request and recognized `I'm not sure if I can do that,
Andrea`, where the base replay produced `... do the Andrea` and lost the 60-day
amount. It also preserved the end of `... this new order is delivered` more
completely. Recognition still contains name errors and missing words, including
an incomplete `manage our own` phrase. The new boundary handling leaves that
incomplete speech separate instead of merging a new speaker's promise into it.

This is one recording on a Linux CPU environment, not a general accuracy claim
or a Windows headset test. Another exploratory float-audio replay also favored
small, but exact errors differed with PCM conversion/chunk boundaries. Small
required about 1.7 times the ASR compute time here. Real caption latency includes
5–7 seconds of capture at the default interval, translation queueing and model
inference. Concurrent Hy-MT2 CPU contention can cause live queue drops even if
this offline replay keeps up. Use realtime mode if skip counters rise.

## Translation prompt checks

Real CPU Hy-MT2 1.8B Q4_K_M inference compared the existing prompt against the
upstream default template, a style template, Korean instructions, longer role
instructions and source-gated pronoun instructions. Inputs included user-supplied
business-call excerpts, intentionally wrong ASR and unrelated service/network
sentences. These trials were inspected manually; no automated fluency score is
claimed. Some alternatives corrected a third-person statement but regressed
business idioms, requests or tone elsewhere. None was promoted. The existing
current-utterance-only prompt and existing narrow guards are unchanged.

Runtime: official llama.cpp b12122, CPU four threads, context 2048, no GPU,
temperature zero. The GGUF bytes matched the publisher SHA-256:
`dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699`.

Recognition environment: Python 3.12.14, faster-whisper 1.2.1, CTranslate2 4.8.5.
