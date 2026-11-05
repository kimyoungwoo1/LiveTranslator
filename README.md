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
