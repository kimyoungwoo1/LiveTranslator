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
