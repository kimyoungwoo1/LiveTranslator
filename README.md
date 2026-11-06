# LiveTranslator

Windows PC output-device / microphone audio transcription and offline Korean translation prototype.

## Project status
- **Development / experimental**. Not yet validated on all Bluetooth headsets or Windows systems.
- Windows WASAPI loopback captures audio *played by the PC* through the selected output device. It cannot capture audio originating from a separate phone that is paired directly to the headset.
- Speech recognition: faster-whisper (CPU INT8). Translation: Argos Translate local language packs.
- First model download needs network; subsequent inference runs locally.

## Run on Windows
Python 3.12 or compatible Python 3.13 required.

1. `py -3.13 -m venv .venv` (or `py -3.12 -m venv .venv`)
2. `.venv\Scripts\python.exe -m pip install -r requirements.txt`
3. `.venv\Scripts\python.exe main.py`

Run commands from the repository folder. The application source is included as `main.py`.

## Distribution
GitHub Actions builds an experimental Windows executable artifact on source updates. Distributing to others should wait for functional tests and a verified release. The EXE may still require a one-time model download.

## Privacy and safety
No microphone recordings, transcripts, settings, access tokens, installed packages or downloaded AI models should be committed. Always verify that recording workplace audio is permitted under company policy, and obtain consent when needed.

## Roadmap
- Fix stop/cancellation behavior via isolating inference
- Test CPU and Bluetooth output device selection
- Verify frozen EXE behavior
- Signed / integrity-checked in-app update flow using public GitHub Releases

## If the Whisper base model download stalls

For machines whose Hugging Face Xet transfer stalls, run `download_base_model.bat` (right click the file on GitHub and download raw). It downloads the four required files using curl directly to `%LOCALAPPDATA%\LiveTranslator\models\base`, verifies the 145217532-byte model.bin, and keeps the model across app updates. Restart a build containing the local-model support after the download succeeds. The app now disables Xet transfers as a fallback.

This does not establish that workplace policy allows these downloads; follow organizational security rules.

## Live subtitle latency (v0.5 prototype)

Audio capture runs on its own thread while Whisper and Argos work. A bounded queue preserves fresh audio without growing delay indefinitely. **When the CPU cannot keep up, old audio chunks are discarded** (the GUI reports a skipped-chunk count); this is not a complete transcription mode.

The status displays measured capture-to-result latency, speech recognition time and Korean translation time. The English original is displayed as soon as transcription finishes, ahead of the translated subtitle. Choose the actual spoken language (en/ja/ko), not necessarily the desired translation language. Using `en` for Korean speech produces nonsense.

Recommended first test on a 16GB Core Ultra 5 225H: `base`, language `en` for English audio, 3-second chunks. If processing consistently exceeds 3 seconds, try `tiny` (which is a separate downloadable speech model); 2-second chunks trade some accuracy for responsiveness.

This is an experimental CPU streaming pipeline, not guaranteed simultaneous interpretation. Stop may still wait for the active inference call to finish.

## In-app updates (v0.6.0+)

The program has a stable PyInstaller launcher and a separate patchable application module. **Download the v0.6.0 updater-enabled EXE once** after the Windows workflow succeeds. Then use `업데이트 확인` in the app to fetch the current published patch from [updates/stable.json](updates/stable.json). Updates are opt-in, HTTPS-only, limited to `app_core.py` and `streaming_pipeline.py`, and verified against the Git blob object digests recorded in the manifest before activation. The previous patch is backed up.

This proof of concept publishes small code-only patches through the same manifest. The manifest currently points to v0.6.4.

- No new EXE build for changes confined to those two Python files. Downloaded AI models, local settings and transcripts remain untouched.
- New Python/native dependencies, PyInstaller configuration, runtime hooks, and fixes to `launcher.py` or `updater.py` still require a **new EXE**.
- Updates are NOT silently installed. Only publish reviewed, tested code to the stable manifest; it is not automatically tied to the `main` branch.
- Because the public repository provides executable Python updates, publishing access must be protected (e.g. GitHub MFA / protected branches). A digest in the same repository prevents accidental corruption, not compromise of the repository.
- For the current experimental build, open GitHub Actions > **Windows EXE lite experiment** > **Run workflow** exactly once. Only use the resulting artifact if its import checks pass; test full translation on Windows.


## Select translation provider (v0.6.3 code patch)

- **Argos (오프라인)**: default, works without an API key and without transmitting transcripts to an external service.
- **Google Cloud (온라인)**: uses the official **Cloud Translation Basic v2 NMT API**. Click **Google API 키 설정** in the app to save your own key using Windows per-user DPAPI encryption, then select **Google Cloud (온라인)** as the translation engine before starting. Leave the input blank in the key dialog to delete a stored key.
- Google requires an enabled **Cloud Translation API**, linked billing account, and an appropriately **API-restricted key**. Charges may apply; configure quotas and budgets in your own Google Cloud project.
- Only **recognized text** is sent to Google when the Google mode is actively selected; microphone/audio recordings are processed by the local Whisper model and not uploaded by this feature.
- The key is **not** written to the public GitHub repository, the settings.json file, or the transcript TXT file. It lives under the local application-data directory encrypted for the Windows user.
- Each person receiving the EXE must configure their own key to enable Google mode. No shared or hard-coded developer key is provided.
- The API timeout is 8 seconds per request, and network/API errors are reported in the status bar. Argos does not automatically switch to Google or vice versa.
- **Compatible with the v0.6.0+ patch-capable EXE**, assuming its code-updater bootstrap is installed and working. In the app choose **업데이트 확인** to get the latest patch. No new Python dependencies are needed. Earlier standalone EXEs cannot install this patch.
- This is a code-release, **not yet validated end-to-end with a real billed Google Cloud account on Windows**. Avoid sending confidential workplace conversations unless organizational policy explicitly permits this.

Google Cloud's official API documentation: https://docs.cloud.google.com/translate/docs/reference/rest/v2/translate

## Argos high-quality mode (v0.6.4)

There is ONE Argos mode (no speed/balanced/high-quality selector). It preserves the separate option to use Google Cloud Translation.

- Argos beam search is increased from 1 to 4 and CTranslate2 compute type is set to float32 for translation quality priority.
- Whisper uses beam size 3 rather than 1 when Argos is selected (speech recognition can also slow down).
- Short, uninterrupted speech fragments are joined and translated at a sentence-ending punctuation mark, after 3 fragments, after a maximum 240 characters, or after 8+ seconds of accumulation. Silence flushes unfinished sentences. This improves translation context, but is not perfect sentence recognition.
- Google mode keeps its previous chunk-based low-latency processing.
- Quality comes with higher CPU use and potentially longer subtitle delay, especially with 16 GB RAM and CPU-only inference. If long delays occur, consider choosing the base speech model and a 3-second capture interval; not every Whisper small model improves real-time responsiveness.
- There is still a bounded queue and skipped-chunk counter: severe overload can cause missing words even in high-quality mode. No claim of professional-grade simultaneous interpretation is made.

Use **업데이트 확인** on an updater-enabled v0.6+ EXE to install the current source patch. This does **not** require downloading a new executable or a new speech model.

## WASAPI invalid device fix (v0.6.5)

If starting translation shows `오디오 캡처 오류: [Errno -9996] Invalid device`:

- Click **오디오 장치 > 새로고침**, then choose the speaker/headphone **Loopback** device through which Windows is *actually playing* audio. Do not select virtual 7.1 Surround output unless it really is your active playback device.
- The **기본 출력** WASAPI loopback is now listed first on fresh starts. Previously selected devices are still preserved by exact name if available.
- Device indices are resolved at capture time, using their current names and live PortAudio indices; a disconnected device is reported instead of silently capturing from an unrelated microphone or speaker.
- Loopback capture tries its advertised native channel count (including 6/8 channels), 16-bit PCM first and float32 next. Stereo/mono fallbacks follow if the device rejects native channels. Multichannel audio is downmixed from active channels to avoid excessive attenuation.
- In-app update: choose **업데이트 확인** > **v0.6.5** > restart. No EXE download is required for updater-enabled clients. If an audio driver exposes a loopback but refuses capture altogether, changing the Windows active playback device or driver may still be necessary.
- This is a defensive compatibility fix validated at code level only; it has not been tested on the user's specific 7.1 virtual device.

## Diagnose empty subtitles (v0.6.6)

If the GUI says an audio device has connected but both transcription panes remain empty, look at the **오디오 입력** meter below the status line while the video is playing:

- **오디오 입력 정상** and a moving meter: audio packets are reaching Whisper. If no original text appears after about 15 seconds, check the speech language (English `en` or Japanese `ja`), the CPU recognition delay, and whether the video is actually speaking.
- **거의 무음**, a nearly stationary meter, or levels near -100 dBFS: the selected WASAPI loopback is not carrying the video's audible signal. Check Windows Volume Mixer for the browser's playback device and choose that device's loopback; the default `7.1 Surround Sound` virtual device may differ from the physical speakers/headphones.
- **오디오 데이터 수신 없음**: the selected driver opened the stream, but no packets are arriving. Refresh devices, reconnect headphones, switch Windows output or its driver.
- The 5.1/7.1 audio downmix now falls back to the loudest channel when summing active channels cancels sound. The near-silence threshold was lowered to retain quiet loopback captures.
- The level meter does not record or upload audio. This patch does not guarantee that a particular virtual output driver can provide a readable audio signal.

Apply from **업데이트 확인** > **v0.6.6** > restart on an updater-enabled LiveTranslator EXE.

## If audio connects but transcription never starts (v0.6.8)

The **소리 찾기** button by the device picker directly probes the audio streams. It does **not** use or download a Whisper model, Argos language pack, or Google API.

1. Stop translation, keep your YouTube/video playing audibly, and click **소리 찾기**.
2. It measures the currently available **playback loopback** sources (not random microphones). Their measured levels are shown in dBFS. A connected 7.1 virtual output can still be silent if Chrome plays through another Windows output.
3. If another output has stronger signal, the app asks permission before switching. Then click **번역 시작**.
4. If all output sources are near silent, use **Windows Settings > System > Sound > Volume mixer** to verify which output Chrome actually uses, and confirm that audio is not muted. Reconnect Bluetooth headsets if necessary.
5. If a source shows clear audio but the original transcript is still blank, check the displayed Whisper status and try the **모델 진단** button after stopping; the problem is then likely speech recognition latency or content rather than the audio source.

No audio from the probe is saved to disk or transmitted online. The loopback probe does not guarantee the underlying device driver is correct. All functionality is delivered via the small in-app update; no EXE rebuild is required for v0.6+.

## Audio scan stuck at 2/5 (v0.6.9)

Some inactive AMD/HDMI or virtual outputs can open successfully but block indefinitely when the scan tries to read silence. The v0.6.9 code patch changes the sound-finder diagnostic only:

- Poll PortAudio input buffer availability before reading; do not call a blocking `read()` on an empty buffer.
- Give each device a **3.5-second watchdog limit** and continue to the next entry even if a driver never returns from a native call. A timeout appears beside that device in the scan results.
- Do not change the Windows playback device or the selected loopback automatically. Choose a detected source only when asked.
- If an old v0.6.8 scan is stuck, close and reopen LiveTranslator, use **업데이트 확인** to get v0.6.9, restart, and then scan while the video is playing.

Note: An unresponsive Windows native driver can leave a daemon diagnostic thread until the application exits. Repeated scans of broken drivers are discouraged; the fix prevents the UI's sequential scan from waiting indefinitely but cannot repair the driver itself.

## Argos numeric accuracy guard (v0.6.10)

This patch corrects a narrow class of clear English-to-Korean scale errors when the source has exactly one 'million', 'billion' or 'trillion' amount and the Korean translation contains exactly one simple '만', '억' or '조' number. Example: 'six billion euro' must be '60억 유로', not '6억 유로'. Ambiguous/compound/multiple amounts are left unchanged.

This guard does not verify Whisper's transcription, infer context that is missing from 3-second speech fragments, or make Argos fluent like professionally edited subtitles. You can manually choose the **small** Whisper model and a 5-second recognition interval for a quality-versus-CPU-delay test; the extra speech model download is not triggered by the v0.6.10 code patch. Google Cloud Translation remains an optional online alternative.

## Remember audio output between launches (v0.6.11)

LiveTranslator now remembers the last explicitly selected or confirmed working audio source in the local settings file (name + loopback/microphone type, not volatile PortAudio device ID). On the next start it restores that source, even if Windows has renumbered devices. This fixes the app repeatedly starting with a silent 7.1 Surround device instead of the previously confirmed Razer headset.

If no saved output is available, the first `번역 시작` automatically probes playback loopbacks for an audible signal and selects/saves the first one that is playing. This may take several seconds if a virtual driver times out; no microphone is auto-selected. After finding a source, translation starts automatically. Later starts use the saved source without an extra probe. You can still manually switch audio devices; changes are saved immediately.

Apply using `업데이트 확인` to v0.6.11 and restart; a full EXE rebuild or model download is unnecessary. Selecting an output with `소리 찾기` and confirming it also saves your preference. If the output device is disconnected at start, automatic detection runs again.

## Pause-aware speech recognition (v0.6.12)

The Argos high-quality option now captures a longer, more natural phrase before recognizing speech. Instead of forcibly splitting every 3 seconds, it waits for a short quiet pause after the configured interval, with a hard cap of 2 extra seconds (up to 10 seconds for longer configured intervals). The local Whisper decoding beam increases to 5, and a short previous transcript is supplied as optional vocabulary context. That context is cleared after silence or dropped audio.

**What this improves:** speech recognition around phrase boundaries, which may in turn help offline translation with vocabulary and sentence context. It cannot guarantee parity with professionally edited video subtitles, nor recover words that the microphone or audio output never captured.

**Tradeoff:** higher CPU work per phrase and potentially increased subtitle delay or skipped chunks on slower CPUs. The existing base model remains selected unless the user explicitly chooses small; the update itself does not download any additional model. The Google provider retains its previous fixed-time capture processing.

Apply by using **업데이트 확인** > v0.6.12 > restart on any updater-enabled v0.6+ client. For the prior Volkswagen news example, compare the original English transcript first, then the Korean translation; the first stage determines whether the translation gets the right English sentence.

## Dark dashboard UI and Exit button (v0.6.13)

- A compact dark dashboard groups audio device and speech-recognition settings, translation providers, primary actions, and live audio status.
- The **원문** and **한국어 번역** panels have scrollbars and a draggable divider; both transcript views remain simultaneously visible.
- High-frequency controls are **번역 시작**, **중지**, **자막 지우기**. Model installation, diagnostics, records and updates remain accessible in the footer.
- The new bottom-right **프로그램 종료** button uses the same exit handler as the title-bar X. If translation is running it requests confirmation, signals capture to stop, saves current settings and closes the app.
- The selected playback source, offline/Google translation mode and the existing updater remain intact. No new third-party UI dependency or EXE build is needed for an updater-enabled installation.
- This is a Tkinter styling/layout patch. The interface has not yet been visually validated on every Windows display-scale configuration.

## v0.6.14 Emergency startup hotfix: Tkinter scrollbar error

If installing the v0.6.13 dark UI patch produces `_tkinter.TclError: Layout Vertical.LT.TScrollbar not found`, the error is a broken custom ttk scrollbar style, **not a Whisper model or audio driver problem**.

**Recover without reinstalling the 800MB EXE or models:**

1. Close the crash dialog and LiveTranslator.
2. Press Win+R and open the Windows LocalAppData `LiveTranslator` folder.
3. Rename the `code_patch` folder to `code_patch.broken` (choose another new backup name if that already exists). Do not delete `models`, `transcripts` or `settings.json`.
4. Restart the same LiveTranslator.exe. Its bundled v0.6.0 bootstrap will be used because the broken patch is no longer activated.
5. Click **업데이트 확인** and install **v0.6.14**, then restart. The new patch uses the built-in ttk scrollbar, progressbar and panedwindow layouts.

Regression tests now include a Windows Tk window creation smoke test (when the runner has a desktop) and a static check forbidding invalid oriented scrollbar style names. The update does not change Whisper, Argos, Google, saved audio devices or user data.

## Test transcript completeness and translation independently (v0.6.15)

On an English business phone conversation, compare the LiveTranslator original transcription with the provider's official transcript *before* evaluating Korean translation. A sample evaluation had 264 reference words, with 38 deletions, 7 substitutions and 1 insertion after lightweight spelling normalization (about 17.4% WER). Entire clauses including the extension to **60 days** were missing; lost speech cannot be reconstructed by Argos.

Before v0.6.15, the same worker performed Whisper and potentially slow Argos translation synchronously. While Argos was translating, newly captured audio accumulated in a bounded queue, eventually skipping old blocks. v0.6.15 runs translation on a separate worker; the UI reports **음성 누락** (audio input chunks skipped) and **번역 누락** (overflowed queued translations) separately.

**Validation:** run the same recording with English/base/3-second/Argos, and observe skip counters while the dialog plays. If audio skips remain zero but the source has omissions, improving the speech model or using longer bounded context may be the appropriate next step. If audio skips rise, the CPU cannot keep pace; model and recognition beam-size tradeoffs still matter. The revised pipeline might increase CPU contention; it does not guarantee a lower WER or professional subtitle quality.

Use **업데이트 확인 > v0.6.15** on an updater-enabled client. No new EXE, models, or Python packages are needed.

## Hy-MT2 offline translation (optional, v0.6.16)

**Argos and Google Cloud remain available.** The third selectable engine is Hy-MT2 1.8B Q4_K_M. It translates recognized English/Japanese text to Korean entirely on the local PC, using a temporary 127.0.0.1 llama.cpp CPU server. Whisper still recognizes the audio locally.

**Setup in the app:**

1. Use an update-capable v0.6+ LiveTranslator.exe. Click **업데이트 확인**, install **v0.6.16**, and restart. Do not download the old 800MB EXE again.
2. Click **Hy-MT2 설치** and approve the **one-time** download. Model weights are roughly **1.1GB** plus the official llama.cpp Windows x64 CPU archive (additional disk space and RAM are required). It downloads only on explicit confirmation.
3. The app verifies the official Hugging Face LFS SHA-256 and the GitHub llama.cpp release asset SHA-256 before installing. Incomplete models are left as resumable .part files. If the official release lacks verifiable digest metadata, the app refuses the runtime download instead of executing it.
4. Select **Hy-MT2 · 오프라인** and click **번역 시작**. It automatically starts a private local CPU inference server and stops it when the session ends.
5. **Optional**: if Windows security policy blocks downloading the runtime, obtain the CPU release from https://github.com/ggml-org/llama.cpp/releases, extract it, and select its **llama-server.exe** using **llama.cpp 실행기 선택**. Never select an executable from an untrusted source.

The model files and runtime are stored under the Windows LocalAppData LiveTranslator **hy_mt2** folder. Existing Argos, Whisper models, transcripts, settings and Google API key are untouched. Download requires internet; subsequent Hy-MT2 translation does not. Some company networks may block GitHub/Hugging Face downloads.

**Practical limits:** Hy-MT2 requires more RAM and CPU time than Argos, and may lag while Whisper runs. The GitHub CPU benchmark (mean translation-only latency ~1.225 s on a 4-core Linux runner) does NOT predict real Windows performance. No automatic fallback to Google/Argos occurs during private translations. If it fails to start, inspect **hy_mt2/server.log** and use Argos in the meantime.

**Distribution:** Hy-MT2 model license Apache 2.0; check attribution and third-party notices before distributing bundled weights or binaries. Our app does not bundle either; each user explicitly downloads from the original sources.

## Hy-MT2 installer failure hotfix (v0.6.17)

If v0.6.16 reported only **Hy-MT2 설치 실패**, update the existing executable to **v0.6.17** using the app's **업데이트 확인** button. Restart, then retry **Hy-MT2 설치**. There is no need to reinstall Whisper or download another LiveTranslator EXE.

The installer now retrieves Hugging Face Xet GGUF file SHA-256 metadata from a HEAD request without following the redirect; older LFS tree metadata remains a fallback. Official llama.cpp Windows CPU ZIP verification also has a hash-pinned fallback when the GitHub latest-release API lacks a digest. No security check is disabled; model and runtime bytes must match the official published hash before being used.

If installation still fails, click **설치 오류 확인** in the translation-engine settings. The latest stage and actual error are also saved under `%LOCALAPPDATA%\\LiveTranslator\\hy_mt2\\install_error.log`. This file can be shared for debugging (review for private paths before sharing). Partial `.part` files remain so the download can resume safely. The existing Argos and Google providers are unaffected.

Because access to Hugging Face / GitHub depends on the user's network, this compatibility patch cannot guarantee a successful download on every Windows machine. In particular, corporate TLS proxies or endpoint antivirus may block the official artifacts; the error log should make that visible.

Official upstream references: https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF and https://github.com/ggml-org/llama.cpp/releases . The pinned fallback CPU ZIP digest is taken from an official GitHub release attestation (b12122).

## v0.6.18: Whisper speech completeness and sentence boundaries

This quality patch keeps the existing **Hy-MT2 / Argos / Google** provider choices and all model files. No EXE rebuild or new model download is needed.

### Changes

- **0.7-second audio overlap (offline quality modes)**: carry a short tail of the previous capture block into the next Whisper inference, reducing word truncation at 3–5 second boundaries. Only the next inference sees the overlap; the recording timeline itself is not duplicated.
- **Conservative English duplication filtering**: strip repeated words only when at least 2 consecutive recognized words at the adjacent boundary match. One-word repeated replies are retained. Japanese transcripts are not mechanically de-duplicated by English word boundaries.
- **Sentence continuity**: treat fragments ending with `...` as unfinished, so `... another new...` followed by `order` can reach Hy-MT2 as a coherent `... another new order` rather than separate translations.
- **CPU adaptive decoding**: use beam 5 when the capture queue is clear, beam 3 with 2–3 pending chunks, beam 2 with 4+ pending chunks, and retain queue-overflow visibility. This favors capturing more speech when Whisper CPU work is falling behind.
- **Google unchanged**: its original fast fixed-length mode is preserved; no audio overlap is added to that provider.

### Test with the existing British Council phone conversation

Compare the *recognized English*, particularly `extend it to 60 days`, `Can you send me a quick email confirmation`, and `new order`. Check that words around chunk boundaries are no longer clipped or duplicated, and that the **음성 누락** counter does not rise. Compare Korean translation separately. More accurate English input can improve Hy-MT2 output, but **v0.6.18 does not guarantee a WER improvement** on every recording. CPU load may increase somewhat due to overlapping inference.

Apply via **업데이트 확인 → v0.6.18 → restart**. Existing EXE and local Hy-MT2 model/runtime are reused.

## v0.6.20 — session-based TXT records and open records folder

Enable **원문·번역 TXT 분리 저장** to save each translation run separately. The first recognized/transcribed segment of a session creates a new directory under `%LOCALAPPDATA%\\LiveTranslator\\transcripts` with a timestamp, translation provider and source language in its name, for example:

    2026-11-06_12-31-25_Hy-MT2_en/
      원문.txt
      한국어_번역.txt

The original text is saved on recognition even if translation is slower. Korean is saved as it arrives. Both files have timestamps, a UTF-8 BOM for Windows Notepad compatibility, and a short session header. Starting translation again creates a new directory, and names are collision-safe even if starts are in the same second. Enabling recording mid-session begins saving from that point.

The **기록 폴더 열기** button launches the folder in Windows Explorer, rather than showing its path in a popup. Old daily files such as `2026-11-05.txt` remain unchanged, and deleting/clearing on-screen subtitles does not delete saved TXT records.

This is an in-app Python code update, with no EXE/model redownload.

## v0.6.21 — Hy-MT2 dialogue subtitle continuity

Retest on the official British Council 'A phone call from a customer' recording exposed two recoverable speech-to-text presentation issues in v0.6.20: overlapping audio occasionally generated long repeated phrases (`we can make a We can make an exception`), and Whisper ended ongoing expressions early (`Let me see.` + `what I can do.`, `It works.` + `both ways.`).

**v0.6.21 changes:**

- Drop adjacent repeated **3+ word prefixes** only when they appear back-to-back within a recognized English segment and have no intervening sentence punctuation. Short/uncertain false starts such as `So you So your` remain untouched.
- Defer translation of a few recognizable unfinished English tails, remove an incorrectly inserted full stop, and join their immediate lowercase continuation before sending them to Hy-MT2 or Argos.
- If the chunk already contains full sentences, translate those complete sentences **right away** and hold only the unfinished tail. Other normal sentences continue to translate immediately.
- Preserve the 0.7-second audio overlap, saved transcripts and existing model runtime. No model or EXE redownload.

This is a conservative sentence-boundary heuristic, not a perfect grammar/parser, and is not guaranteed to repair omissions such as 'extend it to 60 days' or misheard names. Repeated real phrases without punctuation can rarely be collapsed; assess the English transcript against the known script, and keep an eye on the audio skip counters. Actual Windows live ASR and per-subtitle latency are not yet verified for this patch.
