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

This proof of concept publishes v0.6.1 as a small code-only patch so the update button can be tested. It only changes an initial UI status message; it does not claim faster translation than v0.6.0. Once applied, restart to see v0.6.1.

- No new EXE build for changes confined to those two Python files. Downloaded AI models, local settings and transcripts remain untouched.
- New Python/native dependencies, PyInstaller configuration, runtime hooks, and fixes to `launcher.py` or `updater.py` still require a **new EXE**.
- Updates are NOT silently installed. Only publish reviewed, tested code to the stable manifest; it is not automatically tied to the `main` branch.
- Because the public repository provides executable Python updates, publishing access must be protected (e.g. GitHub MFA / protected branches). A digest in the same repository prevents accidental corruption, not compromise of the repository.
- For the current experimental build, open GitHub Actions > **Windows EXE lite experiment** > **Run workflow** exactly once. Only use the resulting artifact if its import checks pass; test full translation on Windows.

