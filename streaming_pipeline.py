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



def resolve_capture_device(pa, selected):
    """Resolve a live PortAudio device by stable name, not a cached device index.

    The device list may change when Bluetooth/USB/default playback devices move.
    Never silently switch to another output because that would translate the
    wrong audio source.
    """
    old_index = int(selected[1])
    is_loopback = bool(selected[2])
    expected_name = selected[3] if len(selected) > 3 else None
    if is_loopback:
        devices = [
            info for info in pa.get_loopback_device_info_generator()
            if int(info.get("maxInputChannels", 0)) > 0
        ]
    else:
        devices = [
            pa.get_device_info_by_index(i)
            for i in range(pa.get_device_count())
        ]
        devices = [
            info for info in devices
            if int(info.get("maxInputChannels", 0)) > 0
            and not info.get("isLoopbackDevice", False)
        ]
    if expected_name:
        same_name = [info for info in devices if info.get("name") == expected_name]
        if not same_name:
            raise RuntimeError(
                "선택한 오디오 장치가 변경되거나 연결 해제되었습니다. "
                "오디오 장치 '새로고침' 후 실제 소리가 출력되는 장치를 선택하세요."
            )
        # If several host APIs expose the same name, preserve the selected index.
        return next(
            (info for info in same_name if int(info["index"]) == old_index),
            same_name[0],
        )
    # Support older UI records that saved only the numeric device index.
    for info in devices:
        if int(info["index"]) == old_index:
            return info
    raise RuntimeError("선택한 오디오 장치를 찾을 수 없습니다. 목록을 새로고침하세요.")


def open_capture_stream(pa, info):
    """Try the device's native input channel count first, then compatible formats.

    WASAPI loopback devices frequently mirror 5.1/7.1 outputs. Their advertised
    maxInputChannels may be 6/8, and forcing stereo can fail on some drivers.
    """
    channels_available = int(info.get("maxInputChannels", 0))
    rate = int(info.get("defaultSampleRate", 0))
    if channels_available < 1 or rate <= 0:
        raise RuntimeError("선택한 장치에 유효한 입력 채널 또는 샘플레이트가 없습니다.")
    channel_options = [channels_available]
    for channel in (2, 1):
        if channel <= channels_available and channel not in channel_options:
            channel_options.append(channel)
    sample_formats = (pyaudio.paInt16, pyaudio.paFloat32)
    frames = max(512, min(int(rate * 0.1), 8192))
    attempts = []
    for channels in channel_options:
        for sample_format in sample_formats:
            try:
                stream = pa.open(
                    format=sample_format, channels=channels, rate=rate,
                    input=True, input_device_index=int(info["index"]),
                    frames_per_buffer=frames,
                )
                return stream, rate, channels, frames, sample_format
            except (OSError, ValueError) as exc:
                attempts.append(f"{channels}ch/{sample_format}: {exc}")
    detail = attempts[0] if attempts else "알 수 없는 오디오 장치 오류"
    raise RuntimeError(
        f"'{info.get('name', '선택한 장치')}'를 열 수 없습니다. "
        f"Windows의 실제 출력 장치를 확인하고 목록을 새로고침하세요. "
        f"({detail})"
    )




def audio_to_mono(raw, channels, sample_format):
    """Decode captured PCM and downmix the channels that actually carry sound."""
    if sample_format == pyaudio.paInt16:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        samples = np.frombuffer(raw, dtype=np.float32)
    if samples.size == 0:
        return samples
    if channels <= 1:
        return samples
    multi = samples.reshape(-1, channels)
    power = np.mean(multi * multi, axis=0)
    loudest = float(np.max(power))
    if loudest <= 0:
        return multi.mean(axis=1)
    active = power >= loudest * 0.05
    mono = multi[:, active].mean(axis=1)
    # Do not let phase-inverted 5.1/7.1 channels cancel audible speech.
    if float(np.mean(mono * mono)) < loudest * 0.04:
        return multi[:, int(np.argmax(power))]
    return mono


def probe_audio_source(selected, seconds=1.1):
    """Measure loopback/mic RMS independently of Whisper and Argos.

    No audio is stored or uploaded.  Caller should only test a microphone
    selected by the user, not automatically survey microphones.
    """
    pa = None
    stream = None
    try:
        pa = pyaudio.PyAudio()
        device = resolve_capture_device(pa, selected)
        stream, rate, channels, frames, fmt = open_capture_stream(pa, device)
        remaining = max(1, int(seconds * rate))
        sum_sq = 0.0
        sample_count = 0
        peak = 0.0
        while remaining > 0:
            nframes = min(frames, remaining)
            raw = stream.read(nframes, exception_on_overflow=False)
            mono = audio_to_mono(raw, channels, fmt)
            if not mono.size:
                raise RuntimeError("오디오 장치가 빈 버퍼를 반환했습니다.")
            sum_sq += float(np.sum(mono * mono))
            sample_count += int(mono.size)
            peak = max(peak, float(np.max(np.abs(mono))))
            remaining -= nframes
        rms = (sum_sq / sample_count) ** 0.5 if sample_count else 0.0
        return {"name": device["name"], "rms": rms, "peak": peak, "channels": channels}
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if pa is not None:
            pa.terminate()


def find_active_loopback(devices, progress=None, seconds=1.0, max_devices=12):
    """Survey only playback loopbacks while external audio is playing.

    Returns one result per candidate. A failure for one driver does not block
    the remaining devices; automatic selection is left to the user.
    """
    candidates = [dev for dev in devices if dev[2]]
    seen = set()
    unique = []
    for dev in candidates:
        name = dev[3] if len(dev) > 3 else dev[0]
        if name in seen:
            continue
        seen.add(name)
        unique.append(dev)
    unique = unique[:max_devices]
    results = []
    for index, device in enumerate(unique, 1):
        if progress is not None:
            progress(index, len(unique), device[0])
        try:
            result = probe_audio_source(device, seconds=seconds)
            results.append({"device": device, **result})
        except Exception as exc:
            results.append({"device": device, "name": device[0], "error": str(exc)})
    return results


def run_stream(selected, interval_seconds, language, model, translate, emit_status,
               emit_original, emit_result, stop_event, quality_mode=False, emit_audio_level=None):
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
            info = resolve_capture_device(pa, selected)
            stream, rate, channels, frames, sample_format = open_capture_stream(pa, info)
            emit_status(f"오디오 연결 완료: {info['name']} · {channels}채널 · 음성 수집 중...")
            parts = []
            count = 0
            target_samples = rate * interval_seconds
            meter_power = 0.0
            meter_count = 0
            meter_peak = 0.0
            last_meter_at = time.monotonic()
            while not stop_event.is_set():
                buffer = stream.read(frames, exception_on_overflow=False)
                samples = audio_to_mono(buffer, channels, sample_format)
                # Level meter reflects the same mono audio sent to Whisper.
                meter_power += float(np.sum(samples * samples))
                meter_count += samples.size
                meter_peak = max(meter_peak, float(np.max(np.abs(samples))))
                now = time.monotonic()
                if emit_audio_level is not None and now - last_meter_at >= 0.7:
                    rms = (meter_power / meter_count) ** 0.5 if meter_count else 0.0
                    emit_audio_level(rms, meter_peak)
                    meter_power = 0.0
                    meter_count = 0
                    meter_peak = 0.0
                    last_meter_at = now
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
            # Preserve quiet but audible Windows loopback feeds.
            if rms < 0.0005:
                emit_status("오디오 수집 중 · 이 구간은 무음으로 판정됨")
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
                emit_status(
                    "오디오 신호는 수집됐지만 음성 인식 결과가 없습니다. "
                    "원본 언어·영상 음성·입력 음량을 확인하세요."
                )
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
