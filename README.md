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
