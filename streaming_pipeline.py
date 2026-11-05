"""Continuous audio capture with one high-quality offline translation profile.

Argos receives complete sentences (or bounded groups of fragments), not
arbitrary 2-3 second fragments. Google still uses fast chunk translation.
Large CPU backlogs are bounded; skipped sections are reported.
"""
import queue
import re
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly


class SentenceBuffer:
    """Collect contiguous Whisper fragments until a reasonable sentence boundary.

    The buffer is bounded even when punctuation recognition fails.
    Japanese is joined without adding an English word separator.
    """

    TERMINAL = re.compile(r'[.!?。！？]["”’\')\]]*$')

    def __init__(self, seconds):
        self.max_age = max(8.0, 2.4 * seconds)
        self.max_chars = 240
        self.max_fragments = 3
        self.reset()

    def reset(self):
        self.text = ""
        self.language = None
        self.first_at = None
        self.count = 0
        self.last_fragment = ""

    def flush(self):
        if not self.text.strip():
            self.reset()
            return None
        item = (self.text.strip(), self.language)
        self.reset()
        return item

    def add(self, fragment, language, now):
        fragment = " ".join(fragment.strip().split())
        if not fragment:
            return []
        results = []
        if self.language and self.language != language:
            previous = self.flush()
            if previous:
                results.append(previous)
        # Avoid exact adjacent duplicate chunks due to Whisper hallucinations.
        if self.text and fragment == self.last_fragment:
            return results
        if self.first_at is None:
            self.first_at = now
            self.language = language
        if self.text:
            joiner = "" if language in ("ja", "zh") else " "
            self.text += joiner
        self.text += fragment
        self.last_fragment = fragment
        self.count += 1
        if (self.TERMINAL.search(self.text) or
                len(self.text) >= self.max_chars or
                self.count >= self.max_fragments or
                now - self.first_at >= self.max_age):
            completed = self.flush()
            if completed:
                results.append(completed)
        return results


def run_stream(selected, interval_seconds, language, model, translate, emit_status,
               emit_original, emit_result, stop_event, quality_mode=False):
    pending = queue.Queue(maxsize=6 if quality_mode else 4)
    capture_done = threading.Event()
    skipped = [0]
    assembler = SentenceBuffer(interval_seconds) if quality_mode else None
    seen_skipped = 0

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
            emit_status("오디오 수집 시작. 음성 인식 중...")
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
                item = (time.monotonic(), block, rate)
                try:
                    pending.put_nowait(item)
                except queue.Full:
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

    def deliver(text, source_lang, captured_at, asr_seconds):
        if stop_event.is_set() or not text.strip():
            return
        emit_status("완성된 문장 번역 중..." if quality_mode else "번역 중...")
        translation_started = time.monotonic()
        translated = text if source_lang == "ko" else translate(text, source_lang)
        translation_seconds = time.monotonic() - translation_started
        if stop_event.is_set():
            return
        emit_result(text, translated)
        lag = time.monotonic() - captured_at
        overload = " · CPU 처리 지연" if lag > 2.5 * interval_seconds else ""
        emit_status(
            f"번역 중 · 지연 {lag:.1f}초 · 인식 {asr_seconds:.1f}초 · "
            f"번역 {translation_seconds:.1f}초 · 대기 {pending.qsize()}개 · "
            f"건너뜀 {skipped[0]}개{overload}"
        )

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
            # A skipped audio chunk is a discontinuity: don't join text across it.
            if skipped[0] != seen_skipped:
                seen_skipped = skipped[0]
                if assembler:
                    item = assembler.flush()
                    if item:
                        deliver(item[0], item[1], captured_at, 0)
            if time.monotonic() - captured_at > max(12.0, 4 * interval_seconds):
                skipped[0] += 1
                emit_status(f"지연된 음성 {skipped[0]}개 건너뜀 · CPU 성능 또는 모델을 확인하세요.")
                continue
            if not samples.size:
                continue
            rms = float(np.sqrt(np.mean(samples * samples)))
            if rms < 0.002:
                if assembler:
                    item = assembler.flush()
                    if item:
                        deliver(item[0], item[1], captured_at, 0)
                continue
            before = time.monotonic()
            audio = resample_poly(samples, 16000, rate).astype(np.float32)
            emit_status(f"음성 인식 중 · 대기 {pending.qsize()}개")
            segments, metadata = model.transcribe(
                audio,
                language=None if language == "auto" else language,
                beam_size=3 if quality_mode else 1,
                best_of=1,
                temperature=0,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            original = " ".join(s.text.strip() for s in segments).strip()
            if stop_event.is_set():
                break
            if not original:
                continue
            asr_seconds = time.monotonic() - before
            emit_original(original)
            source_lang = metadata.language if language == "auto" else language
            if assembler:
                ready = assembler.add(original, source_lang, time.monotonic())
                if not ready:
                    emit_status(
                        f"문장 완성 대기 · 음성 인식 {asr_seconds:.1f}초 · "
                        f"대기 {pending.qsize()}개 · 건너뜀 {skipped[0]}개"
                    )
                for assembled_text, assembled_language in ready:
                    deliver(assembled_text, assembled_language, captured_at, asr_seconds)
            else:
                deliver(original, source_lang, captured_at, asr_seconds)
    finally:
        stop_event.set()
        thread.join(timeout=2)
