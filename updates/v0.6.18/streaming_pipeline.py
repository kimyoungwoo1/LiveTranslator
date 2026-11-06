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

    TERMINAL = re.compile(r'[.!?。！？]["”’\')\]]*

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
        # Whisper's "new..." is a continuation cue, not a final period.
        unfinished = bool(re.search(r"(?:\.{2,}|…)[\"'”’]*$", fragment))
        if unfinished:
            fragment = re.sub(r"(?:\s*\.{2,}|\s*…)[\"'”’]*$", "", fragment).rstrip()
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
        if ((not unfinished and self.TERMINAL.search(self.text)) or
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
        deadline = time.monotonic() + max(2.0, 2.5 * seconds)
        while remaining > 0 and time.monotonic() < deadline:
            nframes = min(frames, remaining)
            # Some disconnected HDMI/virtual WASAPI loopbacks open successfully
            # but block indefinitely in read(). Only read frames known to be
            # available, or let the outer worker watchdog time out.
            try:
                available = stream.get_read_available()
            except (AttributeError, OSError):
                available = None
            if available is not None:
                if available <= 0:
                    time.sleep(0.04)
                    continue
                nframes = min(nframes, int(available))
            raw = stream.read(nframes, exception_on_overflow=False)
            mono = audio_to_mono(raw, channels, fmt)
            if not mono.size:
                raise RuntimeError("오디오 장치가 빈 버퍼를 반환했습니다.")
            sum_sq += float(np.sum(mono * mono))
            sample_count += int(mono.size)
            peak = max(peak, float(np.max(np.abs(mono))))
            remaining -= nframes
        rms = (sum_sq / sample_count) ** 0.5 if sample_count else 0.0
        result = {"name": device["name"], "rms": rms, "peak": peak, "channels": channels}
        if not sample_count:
            result["warning"] = "오디오 데이터 없음 (출력 장치 비활성 또는 드라이버 대기)"
        return result
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if pa is not None:
            pa.terminate()


def probe_with_timeout(selected, seconds=1.0, timeout=3.5):
    """Bound each driver probe even if a native PortAudio call hangs.

    An unresponsive device is abandoned in a daemon thread. This is a
    diagnostic, not a long-running background capture service.
    """
    slot = queue.Queue(maxsize=1)

    def task():
        try:
            slot.put_nowait(("ok", probe_audio_source(selected, seconds=seconds)))
        except Exception as exc:
            slot.put_nowait(("error", str(exc)))

    thread = threading.Thread(target=task, name="AudioProbeWatchdog", daemon=True)
    thread.start()
    try:
        status, result = slot.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(
            "장치 응답 시간 초과 (연결되지 않은 HDMI/가상 장치일 수 있음)"
        ) from None
    if status != "ok":
        raise RuntimeError(result)
    return result


def find_active_loopback(devices, progress=None, seconds=1.0, max_devices=12,
                         max_seconds_per_device=3.5, stop_on_signal=False):
    """Survey playback loopbacks without hanging on any single device."""
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
            result = probe_with_timeout(
                device, seconds=seconds, timeout=max_seconds_per_device
            )
            results.append({"device": device, **result})
            # First-run automatic detection only needs one audible source.
            # A manual '소리 찾기' still examines every source for comparison.
            if (stop_on_signal and not result.get("warning")
                    and float(result.get("rms", 0)) >= 0.001):
                break
        except Exception as exc:
            results.append({
                "device": device, "name": device[0], "error": str(exc)
            })
    return results




# Optional Hy-MT2 translation backend: standard-library only.
# It is intentionally kept in the patchable streaming_pipeline.py so users do
# not need another 800MB EXE download or a new installed Python dependency.
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import zipfile
from urllib import error as urlerror
from urllib import request as urlrequest

HYM_REPO = "tencent/Hy-MT2-1.8B-GGUF"
HYM_FILE = "Hy-MT2-1.8B-Q4_K_M.gguf"
HYM_HF_API = "https://huggingface.co/api/models/" + HYM_REPO + "/tree/main"
HYM_MODEL_URL = "https://huggingface.co/" + HYM_REPO + "/resolve/main/" + HYM_FILE
HYM_GH_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
# SHA-256 published in the official GitHub release attestation for b12122.
# A fixed hash allows verified install when the latest-release API omits a digest.
HYM_PINNED_WIN_CPU = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b12122/"
    "llama-b12122-bin-win-cpu-x64.zip",
    "d1a53a4be7588d3b16512bbde27173fa9d77e44c430760390e621cce7256eb27",
    "llama-b12122-bin-win-cpu-x64.zip",
)


def _hym_json(url, timeout=20):
    request = urlrequest.Request(url, headers={
        "User-Agent": "LiveTranslator-HyMT2/0.6",
        "Accept": "application/json",
    })
    with urlrequest.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read(3_000_000).decode("utf-8"))


def _hym_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hym_download(url, dest, expected_hash, max_bytes, progress, cancel=None):
    """Bounded, resumable HTTPS download with compulsory publisher SHA-256."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash or ""):
        raise RuntimeError("공식 배포 파일의 SHA-256 검증 정보를 확인할 수 없습니다.")
    if not url.startswith(("https://huggingface.co/", "https://github.com/")):
        raise RuntimeError("허용되지 않은 모델 또는 실행기 다운로드 주소입니다.")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if dest.is_file() and dest.stat().st_size < max_bytes:
        if _hym_sha256(dest) == expected_hash:
            progress(f"이미 다운로드된 파일을 확인했습니다: {dest.name}")
            return dest
    # Resume interrupted files, but verify the final full-file digest.
    existing = part.stat().st_size if part.exists() else 0
    if existing > max_bytes:
        part.unlink()
        existing = 0
    request_headers = {"User-Agent": "LiveTranslator-HyMT2/0.6"}
    if existing:
        request_headers["Range"] = f"bytes={existing}-"
    request = urlrequest.Request(url, headers=request_headers)
    last_report = 0.0
    with urlrequest.urlopen(request, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
        if existing and (response.status != 206 or
                         not content_range.startswith(f"bytes {existing}-")):
            # Server returned an entire file rather than a resumable range.
            existing = 0
        mode = "ab" if existing and response.status == 206 else "wb"
        total = existing
        if response.headers.get("Content-Length"):
            total_expected = existing + int(response.headers["Content-Length"])
            if total_expected > max_bytes:
                raise RuntimeError("예상보다 큰 다운로드 파일입니다. 설치를 중단했습니다.")
        with part.open(mode) as out:
            while True:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("사용자가 설치를 중단했습니다.")
                block = response.read(2 * 1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise RuntimeError("다운로드 크기 제한을 초과했습니다.")
                out.write(block)
                now = time.monotonic()
                if now - last_report > 1.5:
                    progress(f"{dest.name} 다운로드 중 · {total / 1048576:.0f} MB")
                    last_report = now
    progress("SHA-256 무결성을 검증하는 중...")
    if _hym_sha256(part) != expected_hash:
        part.unlink(missing_ok=True)
        raise RuntimeError("다운로드 파일의 SHA-256이 공식 배포본과 다릅니다.")
    os.replace(part, dest)
    return dest


def _hym_model_metadata():
    """Get the publisher's SHA-256 for Xet *and* legacy LFS files.

    Xet models may have no 'lfs' field in the Hub tree listing. The Hub
    advertises x-linked-etag/x-linked-size in the first (302) HEAD response.
    Prevent redirects so that CDN headers cannot hide that metadata.
    """
    class StopRedirect(urlrequest.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    headers = None
    try:
        opener = urlrequest.build_opener(StopRedirect())
        req = urlrequest.Request(HYM_MODEL_URL, method="HEAD", headers={
            "User-Agent": "LiveTranslator-HyMT2/0.6"
        })
        try:
            response = opener.open(req, timeout=20)
        except urlerror.HTTPError as exc:
            if exc.code not in (301, 302, 303, 307, 308):
                raise
            response = exc
        with response:
            headers = response.headers
        sha = (headers.get("X-Linked-Etag") or headers.get("Etag") or "")
        sha = sha.strip().strip('"').lower()
        size = int(headers.get("X-Linked-Size") or
                   headers.get("Content-Length") or "0")
        if re.fullmatch(r"[0-9a-f]{64}", sha) and 500_000_000 <= size <= 2_500_000_000:
            return sha, size
    except (OSError, ValueError, TypeError):
        pass

    # Fallback for older LFS repositories and proxies that strip HEAD headers.
    entries = _hym_json(HYM_HF_API, timeout=20)
    if not isinstance(entries, list):
        raise RuntimeError("공식 모델 파일 목록 응답이 올바르지 않습니다.")
    entry = next((x for x in entries if x.get("path") == HYM_FILE), None)
    if not entry:
        raise RuntimeError("공식 모델 저장소에서 Q4_K_M 파일을 찾지 못했습니다.")
    lfs = entry.get("lfs") or {}
    digest = lfs.get("oid") or entry.get("oid")
    if isinstance(digest, str) and digest.startswith("sha256:"):
        digest = digest[7:]
    size = int(entry.get("size") or lfs.get("size") or 0)
    if not re.fullmatch(r"[0-9a-f]{64}", digest or ""):
        raise RuntimeError(
            "Hugging Face가 모델 SHA-256 정보를 반환하지 않습니다. "
            "모델 페이지 연결 또는 회사 네트워크의 HEAD 요청 차단 여부를 확인하세요."
        )
    return digest, size


def _hym_runtime_asset():
    """Use official GitHub SHA-256, or a pinned, publicly attested CPU release."""
    try:
        release = _hym_json(HYM_GH_API, timeout=12)
        assets = release.get("assets") or []
        choices = [
            asset for asset in assets
            if re.fullmatch(
                r"llama-.*-bin-win-cpu-x64\.zip",
                asset.get("name", ""), re.I
            )
        ]
        if choices:
            asset = choices[0]
            digest = asset.get("digest") or ""
            url = asset.get("browser_download_url") or ""
            if (re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
                    and url.startswith(
                        "https://github.com/ggml-org/llama.cpp/releases/download/"
                    )):
                return url, digest[7:], asset["name"]
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        pass
    # Never accept an unverified or mismatching binary: use a pinned official
    # release instead, and still verify its bytes before executing anything.
    return HYM_PINNED_WIN_CPU


def find_hym_server(folder, custom_path=None):
    if custom_path:
        p = Path(custom_path)
        if p.is_file() and p.name.lower() == "llama-server.exe":
            return p
        raise RuntimeError("선택한 llama-server.exe 경로를 찾을 수 없습니다.")
    runtime = Path(folder) / "runtime"
    return next(runtime.rglob("llama-server.exe"), None) if runtime.is_dir() else None


def hym_ready(folder, custom_path=None):
    folder = Path(folder)
    model = folder / "models" / HYM_FILE
    runtime = find_hym_server(folder, custom_path)
    return (runtime is not None and model.is_file()
            and model.stat().st_size > 500_000_000)


def setup_hym(folder, progress, custom_path=None, cancel=None):
    """Explicit install only: official model and official CPU runtime, once."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model = folder / "models" / HYM_FILE
    if not model.is_file() or model.stat().st_size < 500_000_000:
        progress("공식 Hugging Face 모델 파일 확인 중...")
        digest, size = _hym_model_metadata()
        if not 500_000_000 <= size <= 2_500_000_000:
            raise RuntimeError("공식 모델 크기 정보가 올바르지 않습니다.")
        _hym_download(HYM_MODEL_URL, model, digest, 2_500_000_000, progress, cancel)
    if find_hym_server(folder, custom_path):
        progress("Hy-MT2 모델과 llama.cpp 실행기 준비 완료")
        return
    progress("공식 llama.cpp CPU 실행기 확인 중...")
    url, digest, name = _hym_runtime_asset()
    zip_file = folder / name
    _hym_download(url, zip_file, digest, 350_000_000, progress, cancel)
    runtime = folder / "runtime"
    staging = Path(tempfile.mkdtemp(prefix="runtime-stage-", dir=folder))
    try:
        with zipfile.ZipFile(zip_file) as archive:
            items = archive.infolist()
            if len(items) > 1500 or sum(x.file_size for x in items) > 900_000_000:
                raise RuntimeError("실행기 압축파일에 허용되지 않은 데이터가 포함돼 있습니다.")
            for item in items:
                path = Path(item.filename.replace("\\", "/"))
                if (path.is_absolute() or ".." in path.parts or
                        item.file_size > 250_000_000 or
                        (item.external_attr >> 16) & 0o170000 == 0o120000):
                    raise RuntimeError("안전하지 않은 ZIP 항목을 차단했습니다.")
                target = staging / path
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=2 * 1024 * 1024)
        if not next(staging.rglob("llama-server.exe"), None):
            raise RuntimeError("실행기 ZIP에서 llama-server.exe를 찾을 수 없습니다.")
        if runtime.exists():
            shutil.rmtree(runtime)
        os.replace(staging, runtime)
        zip_file.unlink(missing_ok=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    progress("Hy-MT2 모델과 공식 llama.cpp CPU 실행기 설치 완료")


class HyMT2Backend:
    """Launch a private local CPU-only llama-server and translate via loopback."""
    def __init__(self, folder, server_path=None):
        self.folder = Path(folder)
        self.server_path = server_path
        self.process = None
        self.log = None
        self.port = None

    def start(self, stop_event, progress):
        if not hym_ready(self.folder, self.server_path):
            raise RuntimeError("Hy-MT2 설치가 필요합니다. 번역 엔진의 'Hy-MT2 설치'를 누르세요.")
        exe = find_hym_server(self.folder, self.server_path)
        model = self.folder / "models" / HYM_FILE
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.folder.mkdir(parents=True, exist_ok=True)
        self.log = (self.folder / "server.log").open("w", encoding="utf-8")
        args = [
            str(exe), "--model", str(model),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--threads", "4", "--ctx-size", "2048",
            "--n-gpu-layers", "0", "--parallel", "1"
        ]
        progress("Hy-MT2 CPU 번역 모델 로딩 중 · 최초 시작에는 시간이 걸릴 수 있습니다...")
        try:
            self.process = subprocess.Popen(
                args, cwd=str(exe.parent), stdout=self.log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and not stop_event.is_set():
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "Hy-MT2 서버가 종료되었습니다. 설치 폴더의 server.log를 확인하세요."
                    )
                try:
                    with urlrequest.urlopen(
                        f"http://127.0.0.1:{self.port}/health", timeout=1
                    ) as response:
                        health = json.loads(response.read(4096).decode("utf-8"))
                        if health.get("status") in ("ok", "ready"):
                            progress("Hy-MT2 로컬 번역 모델 준비 완료")
                            return
                except (OSError, ValueError):
                    pass
                time.sleep(0.5)
            if stop_event.is_set():
                raise RuntimeError("Hy-MT2 시작이 중지되었습니다.")
            raise RuntimeError("Hy-MT2 모델 시작 시간이 90초를 초과했습니다.")
        except Exception:
            self.stop()
            raise

    def translate(self, text, source_lang):
        if not text.strip():
            return ""
        if source_lang not in ("en", "ja"):
            raise RuntimeError("Hy-MT2는 영어 또는 일본어 원문만 번역하도록 설정돼 있습니다.")
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("Hy-MT2 로컬 번역 서버가 종료되었습니다.")
        lang = "English" if source_lang == "en" else "Japanese"
        prompt = (
            f"Translate the following {lang} text into natural Korean. "
            "Preserve numbers, amounts and proper names faithfully. "
            "Output only the Korean translation without explanation.\n\n" +
            text[:5000]
        )
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 220, "stream": False,
        }
        request = urlrequest.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urlrequest.urlopen(request, timeout=35) as response:
                answer = json.loads(response.read(1_000_000))
            translated = answer["choices"][0]["message"]["content"].strip()
            if not translated:
                raise RuntimeError("번역 결과가 비어 있습니다.")
            return translated
        except (urlerror.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Hy-MT2 로컬 번역 응답 실패: {exc}") from None
        except (KeyError, TypeError, IndexError, ValueError):
            raise RuntimeError("Hy-MT2 응답 형식을 해석할 수 없습니다.") from None

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
        if self.log:
            self.log.close()
            self.log = None
        self.port = None




# The ASR context overlap is not streamed twice: only the NEW audio block
# counts toward the next capture interval. A small portion of the previous
# block is included solely to avoid truncating a word at a hard cut.
QUALITY_OVERLAP_SECONDS = 0.7
_OVERLAP_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)


def add_audio_context(block, previous_tail, rate, quality_mode=True):
    """Return audio for ASR plus the tail to carry into the next block.

    Both arrays must be mono samples with the same sample rate. No audio is
    duplicated in the capture timeline or added to the queue more than once.
    """
    if not quality_mode or block.size == 0:
        return block, np.empty(0, dtype=np.float32)
    tail_length = min(block.size, max(1, int(rate * QUALITY_OVERLAP_SECONDS)))
    audio = (np.concatenate((previous_tail, block))
             if previous_tail.size else block)
    return audio, block[-tail_length:].copy()


def strip_transcript_overlap(previous, current, language="en", max_words=16):
    """Remove only *verified* duplicate words caused by overlapping audio.

    Requires two or more matching words so legitimate repeated one-word
    replies (e.g. 'Yes. Yes.') are not discarded. Skips Japanese because it
    lacks consistent word boundaries; Japanese keeps the exact ASR output.
    """
    current = current.strip()
    if not previous or not current or language not in ("en", "auto"):
        return current
    prev_words = [(m.group().casefold(), m.start(), m.end())
                  for m in _OVERLAP_WORD.finditer(previous)]
    next_words = [(m.group().casefold(), m.start(), m.end())
                  for m in _OVERLAP_WORD.finditer(current)]
    max_match = min(max_words, len(prev_words), len(next_words))
    for n in range(max_match, 1, -1):
        if ([x[0] for x in prev_words[-n:]] ==
                [x[0] for x in next_words[:n]]):
            # Preserve the exact punctuation and whitespace of newly
            # recognized text; remove duplicate prefix, then strip spacers.
            return current[next_words[n - 1][2]:].lstrip(" \t,.;:!?")
    return current




def choose_asr_beam_size(quality_mode, pending_chunks):
    """Favor completeness when ASR cannot keep pace with live capture.

    Beam 5 for best recognition with spare CPU; reduce search work only when
    multiple WHOLE audio chunks are waiting, before queue drops them.
    """
    if not quality_mode:
        return 1
    if pending_chunks >= 4:
        return 2
    if pending_chunks >= 2:
        return 3
    return 5



def should_finalize_audio(sample_count, silence_count, rate, interval_seconds,
                          quality_mode=False):
    """Split natural speech near a pause instead of at a fixed clock boundary.

    Quality mode waits for at least the configured amount of audio, then
    prefers a >=0.28s quiet interval, but forces a cut after two more seconds
    to bound latency and memory.  Other providers retain fixed intervals.
    """
    if not quality_mode:
        return sample_count >= int(rate * interval_seconds)
    minimum = max(3.0, float(interval_seconds))
    maximum = min(10.0, max(5.0, minimum + 2.0))
    return (sample_count >= int(rate * minimum)
            and (silence_count >= int(rate * 0.28)
                 or sample_count >= int(rate * maximum)))

def run_stream(selected, interval_seconds, language, model, translate, emit_status,
               emit_original, emit_result, stop_event, quality_mode=False, emit_audio_level=None,
               emit_error=None):
    pending = queue.Queue(maxsize=6 if quality_mode else 4)
    # Text translation must not stop Whisper from recognizing the next audio block.
    translation_pending = queue.Queue(maxsize=40)
    translated_dropped = [0]
    capture_done = threading.Event()
    skipped = [0]
    assembler = SentenceBuffer(interval_seconds) if quality_mode else None
    seen_skipped = 0
    recent_transcript = ""
    last_recognized_audio_text = ""
    previous_lang = None

    def capture():
        pa = None
        stream = None
        try:
            pa = pyaudio.PyAudio()
            info = resolve_capture_device(pa, selected)
            stream, rate, channels, frames, sample_format = open_capture_stream(pa, info)
            emit_status(f"오디오 연결 완료: {info['name']} · {channels}채널 · 음성 수집 중...")
            parts = []
            overlap_tail = np.empty(0, dtype=np.float32)
            count = 0
            quiet_samples = 0
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
                if float(np.sqrt(np.mean(samples * samples))) < 0.0015:
                    quiet_samples += samples.size
                else:
                    quiet_samples = 0
                if not should_finalize_audio(
                    count, quiet_samples, rate, interval_seconds, quality_mode
                ):
                    continue
                block = np.concatenate(parts)
                block, overlap_tail = add_audio_context(
                    block, overlap_tail, rate, quality_mode
                )
                parts = []
                count = 0
                quiet_samples = 0
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
        """Queue finished sentences without blocking speech recognition."""
        if stop_event.is_set() or not text.strip():
            return
        task = (text, source_lang, captured_at, asr_seconds)
        try:
            translation_pending.put_nowait(task)
        except queue.Full:
            # Never hold the ASR thread waiting on text translation.
            # Drop the oldest *translation* task, not the captured audio.
            try:
                translation_pending.get_nowait()
                translated_dropped[0] += 1
            except queue.Empty:
                pass
            translation_pending.put_nowait(task)
            emit_status(
                f"번역 대기열 초과 · 오래된 번역 {translated_dropped[0]}개 생략 "
                f"(원문 인식은 계속 진행)"
            )

    def translation_worker():
        try:
            while not stop_event.is_set():
                try:
                    text, source_lang, captured_at, asr_seconds = translation_pending.get(
                        timeout=0.2
                    )
                except queue.Empty:
                    continue
                if stop_event.is_set():
                    break
                started = time.monotonic()
                result = text if source_lang == "ko" else translate(text, source_lang)
                duration = time.monotonic() - started
                if stop_event.is_set():
                    break
                emit_result(text, result)
                lag = time.monotonic() - captured_at
                overload = " · 처리 지연" if lag > 2.5 * interval_seconds else ""
                emit_status(
                    f"번역 완료 · 지연 {lag:.1f}초 · 인식 {asr_seconds:.1f}초 · "
                    f"번역 {duration:.1f}초 · 음성 대기 {pending.qsize()}개 · "
                    f"번역 대기 {translation_pending.qsize()}개 · "
                    f"음성 누락 {skipped[0]}개 · 번역 누락 {translated_dropped[0]}개"
                    f"{overload}"
                )
        except Exception as exc:
            if emit_error is not None:
                emit_error(f"번역 오류: {exc}")
            else:
                emit_status(f"번역 오류: {exc}")
            stop_event.set()

    translation_thread = threading.Thread(
        target=translation_worker, name="LiveTranslatorTranslate", daemon=True
    )
    translation_thread.start()

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
                recent_transcript = ""
                last_recognized_audio_text = ""
                previous_lang = None
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
                # Do not carry previous-topic vocabulary through an extended pause.
                recent_transcript = ""
                last_recognized_audio_text = ""
                previous_lang = None
                emit_status("오디오 수집 중 · 이 구간은 무음으로 판정됨")
                if assembler:
                    item = assembler.flush()
                    if item:
                        deliver(item[0], item[1], captured_at, 0)
                continue
            before = time.monotonic()
            audio = resample_poly(samples, 16000, rate).astype(np.float32)
            emit_status(f"음성 인식 중 · 대기 {pending.qsize()}개")
            # The last *recognized* phrase offers subject/vocabulary context
            # across speech chunks; do not reuse it across skipped audio.
            prompt = None
            if quality_mode and recent_transcript and (
                language != "auto" or previous_lang == language
            ):
                prompt = recent_transcript[-140:]
            segments, metadata = model.transcribe(
                audio,
                language=None if language == "auto" else language,
                beam_size=choose_asr_beam_size(quality_mode, pending.qsize()),
                best_of=1,
                temperature=0,
                initial_prompt=prompt,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            raw_original = " ".join(s.text.strip() for s in segments).strip()
            if stop_event.is_set():
                break
            if not raw_original:
                emit_status(
                    "오디오 신호는 수집됐지만 음성 인식 결과가 없습니다. "
                    "원본 언어·영상 음성·입력 음량을 확인하세요."
                )
                continue
            asr_seconds = time.monotonic() - before
            source_lang = metadata.language if language == "auto" else language
            original = raw_original
            if quality_mode:
                if previous_lang == source_lang:
                    original = strip_transcript_overlap(
                        last_recognized_audio_text, raw_original, source_lang
                    )
                last_recognized_audio_text = raw_original
                # Feed only new recognized text back as bounded terminology
                # context. A repeated hallucination must not be amplified.
                if original:
                    recent_transcript = original[-180:]
                previous_lang = source_lang
            if not original:
                emit_status("중복 인식된 구간을 제거했습니다 · 음성 수집은 계속됩니다.")
                continue
            emit_original(original)
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
        translation_thread.join(timeout=2)
