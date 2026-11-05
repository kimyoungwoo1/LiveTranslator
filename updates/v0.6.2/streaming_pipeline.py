"""Separated audio capture and CPU inference for Windows WASAPI loopback.

Capture runs independently of Whisper/Argos.  A small bounded queue favors fresh
subtitles over silently accumulating an unbounded delay.  Skipped chunks are
reported in the GUI and should not be mistaken for complete transcription.
"""
import queue
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly


def run_stream(selected, interval_seconds, language, model, translate, emit_status, emit_original, emit_result, stop_event):
    pending = queue.Queue(maxsize=4)
    capture_done = threading.Event()
    skipped = [0]

    def capture():
        pa = None
        stream = None
        try:
            pa = pyaudio.PyAudio()
            info = pa.get_device_info_by_index(selected[1])
            rate = int(info["defaultSampleRate"])
            channels = min(int(info["maxInputChannels"]), 2)
            if channels < 1 or rate <= 0:
                raise RuntimeError("선택한 오디오 장치의 입력 설정이 올바르지 않습니다.")
            frames = max(512, min(int(rate * 0.1), 8192))
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=selected[1],
                frames_per_buffer=frames,
            )
            emit_status("오디오 수집 시작. 음성 인식 및 번역 대기 중...")
            parts = []
            count = 0
            target_samples = rate * interval_seconds
            while not stop_event.is_set():
                buffer = stream.read(frames, exception_on_overflow=False)
                samples = np.frombuffer(buffer, dtype=np.float32)
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)
                parts.append(samples)
                count += samples.size
                if count < target_samples:
                    continue
                block = np.concatenate(parts)
                parts = []
                count = 0
                # Timestamp at END of captured speech, for honest live lag readings.
                item = (time.monotonic(), block, rate)
                try:
                    pending.put_nowait(item)
                except queue.Full:
                    # Live mode: discard oldest pending audio, not the newest audio.
                    try:
                        pending.get_nowait()
                        skipped[0] += 1
                    except queue.Empty:
                        pass
                    try:
                        pending.put_nowait(item)
                    except queue.Full:
                        skipped[0] += 1
        except Exception as exc:
            emit_status(f"오디오 캡처 오류: {exc}")
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if pa is not None:
                pa.terminate()
            capture_done.set()

    thread = threading.Thread(target=capture, name="LiveTranslatorCapture", daemon=True)
    thread.start()
    try:
        while not stop_event.is_set():
            try:
                captured_at, samples, rate = pending.get(timeout=0.2)
            except queue.Empty:
                if capture_done.is_set():
                    break
                continue
            # Prioritize a useful live display over replaying very old subtitles.
            if time.monotonic() - captured_at > max(6.0, 2.5 * interval_seconds):
                skipped[0] += 1
                emit_status(f"오디오 수집 중 · 지연된 구간 {skipped[0]}개 건너뜀")
                continue
            rms = float(np.sqrt(np.mean(samples * samples)))
            if rms < 0.002:
                continue
            before = time.monotonic()
            audio = resample_poly(samples, 16000, rate).astype(np.float32)
            emit_status(f"음성 인식 중 · 대기 {pending.qsize()}개")
            segments, metadata = model.transcribe(
                audio,
                language=None if language == "auto" else language,
                beam_size=1,
                best_of=1,
                temperature=0,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            # faster-whisper does the actual inference as segments are iterated.
            original = " ".join(s.text.strip() for s in segments).strip()
            if stop_event.is_set():
                break
            if not original:
                continue
            asr_seconds = time.monotonic() - before
            emit_original(original)
            source_lang = metadata.language if language == "auto" else language
            translation_started = time.monotonic()
            translated = original if source_lang == "ko" else translate(original, source_lang)
            translation_seconds = time.monotonic() - translation_started
            if stop_event.is_set():
                break
            emit_result(original, translated)
            duration = time.monotonic() - before
            lag = time.monotonic() - captured_at
            overload = " · 처리 지연(모델을 tiny로 변경 권장)" if duration > interval_seconds else ""
            emit_status(
                f"번역 중 · 지연 {lag:.1f}초 · 인식 {asr_seconds:.1f}초 · "
                f"번역 {translation_seconds:.1f}초 · 대기 {pending.qsize()}개 · "
                f"건너뜀 {skipped[0]}개{overload}"
            )
    finally:
        stop_event.set()
        thread.join(timeout=2)
