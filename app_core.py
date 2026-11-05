"""LiveTranslator: Windows local audio-to-Korean subtitles (prototype)."""
import datetime
import math
import os
import re
from decimal import Decimal, InvalidOperation
import subprocess
import sys
import ctypes
from ctypes import wintypes
from html import unescape
from tkinter import simpledialog
from urllib import error as urlerror
from urllib import request as urlrequest
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Single high-quality offline Argos mode; no speed/balanced mode selection.
# Set these before argostranslate is imported in the translation worker.
os.environ["ARGOS_CHUNK_TYPE"] = "MINISBD"
os.environ["ARGOS_DEVICE_TYPE"] = "cpu"
os.environ["ARGOS_BEAM_SIZE"] = "4"
os.environ["ARGOS_COMPUTE_TYPE"] = "float32"
import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
import updater

import numpy as np
from scipy.signal import resample_poly
import pyaudiowpatch as pyaudio

APP_DIR = updater.APP_DIR
APP_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS = APP_DIR / "settings.json"
TRANSCRIPTS = APP_DIR / "transcripts"
TRANSCRIPTS.mkdir(exist_ok=True)


GOOGLE_KEY_FILE = APP_DIR / "google_api_key.dpapi"
GOOGLE_TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"

class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi(data, decrypt=False):
    """Encrypt/decrypt bytes for the current Windows user, using Windows DPAPI."""
    if os.name != "nt":
        raise RuntimeError("Google API 키 보관은 Windows에서만 지원합니다.")
    dll = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    func = dll.CryptUnprotectData if decrypt else dll.CryptProtectData
    func.argtypes = [
        ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob)
    ]
    func.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    buffer = ctypes.create_string_buffer(data)
    input_blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output_blob = _DataBlob()
    if not func(ctypes.byref(input_blob), None, None, None, None, 0, ctypes.byref(output_blob)):
        raise RuntimeError("Windows 사용자 계정 보안 저장소를 사용할 수 없습니다.")
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel.LocalFree(ctypes.cast(output_blob.pbData, ctypes.c_void_p))


def store_google_key(key):
    key = key.strip()
    if not key or len(key) > 512 or "\n" in key or "\r" in key:
        raise ValueError("유효한 Google Cloud Translation API 키를 입력하세요.")
    APP_DIR.mkdir(parents=True, exist_ok=True)
    temp = APP_DIR / "google_api_key.dpapi.tmp"
    try:
        temp.write_bytes(_dpapi(key.encode("utf-8")))
        temp.replace(GOOGLE_KEY_FILE)
    finally:
        temp.unlink(missing_ok=True)


def read_google_key():
    if not GOOGLE_KEY_FILE.is_file():
        return None
    return _dpapi(GOOGLE_KEY_FILE.read_bytes(), decrypt=True).decode("utf-8")


def google_translate_text(text, source_lang, api_key):
    """Send recognized TEXT only to official Google Cloud Translation Basic v2."""
    if not text.strip():
        return ""
    if not api_key:
        raise RuntimeError("Google API 키가 등록되어 있지 않습니다.")
    if len(text) > 10000:
        raise RuntimeError("Google 번역 요청이 너무 깁니다.")
    payload = {"q": text, "target": "ko", "format": "text"}
    if source_lang and source_lang != "auto":
        payload["source"] = source_lang
    req = urlrequest.Request(
        GOOGLE_TRANSLATE_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-goog-api-key": api_key,
            "User-Agent": "LiveTranslator/0.6"
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=8) as response:
            result = json.loads(response.read(262144))
        translated = result["data"]["translations"][0]["translatedText"]
        if not isinstance(translated, str):
            raise ValueError("응답 형식 오류")
        return unescape(translated)
    except urlerror.HTTPError as exc:
        if exc.code == 400:
            raise RuntimeError("Google 번역 요청이 거부됐습니다. API 설정을 확인하세요.") from None
        if exc.code in (401, 403):
            raise RuntimeError("Google API 키 권한, 프로젝트 결제 또는 Translation API 활성화 상태를 확인하세요.") from None
        if exc.code == 429:
            raise RuntimeError("Google 번역 요청 한도에 도달했습니다. 사용량 제한을 확인하세요.") from None
        raise RuntimeError(f"Google 번역 서버 오류(HTTP {exc.code})") from None
    except (urlerror.URLError, TimeoutError):
        raise RuntimeError("Google 번역 네트워크 연결 실패 또는 8초 시간 초과") from None
    except (KeyError, IndexError, TypeError, ValueError):
        raise RuntimeError("Google 번역 응답을 해석할 수 없습니다.") from None



# A narrow safety check for financial quantities. It does not attempt to
# translate sentences or infer amounts Whisper failed to recognize.
_SCALE = {"million": Decimal("1000000"), "billion": Decimal("1000000000"),
          "trillion": Decimal("1000000000000")}
_ENGLISH_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90
}
_SCALE_EXPR = re.compile(
    r"\b(?P<number>\d+(?:,\d{3})*(?:\.\d+)?|"
    + "|".join(_ENGLISH_NUMBERS.keys())
    + r")\s+(?P<scale>million|billion|trillion)\b", re.IGNORECASE
)
_KOREAN_SCALE = re.compile(r"(?<![\d.])(?P<n>\d+(?:\.\d+)?)\s*(?P<u>조|억|만)(?![\w])")


def _format_korean_large_number(english_number, scale):
    try:
        amount = Decimal(str(_ENGLISH_NUMBERS.get(
            english_number.lower(), english_number.replace(",", "")
        ))) * _SCALE[scale.lower()]
    except (InvalidOperation, ValueError, KeyError):
        return None
    for factor, suffix in (
        (Decimal("1000000000000"), "조"),
        (Decimal("100000000"), "억"),
        (Decimal("10000"), "만"),
    ):
        if amount >= factor and amount % factor == 0:
            number = amount / factor
            return f"{number:f}".rstrip("0").rstrip(".") + suffix if number != number.to_integral_value() else f"{int(number)}{suffix}"
    return None


def guard_argos_quantity(source, translated):
    """Fix only unambiguous single-scale numeric magnitude mistranslations.

    Example: 'nearly six billion euro' translated as '거의 6억 유로'
    -> '거의 60억 유로'. Skip compound amounts and multiple quantities,
    rather than risk silently fabricating the translated meaning.
    """
    expressions = list(_SCALE_EXPR.finditer(source))
    quantities = list(_KOREAN_SCALE.finditer(translated))
    if len(expressions) != 1 or len(quantities) != 1:
        return translated
    original = expressions[0]
    expected = _format_korean_large_number(
        original.group("number"), original.group("scale")
    )
    if not expected:
        return translated
    current = quantities[0]
    # A Korean result such as "2억 5천만" is a compound quantity.
    # Never replace only its first portion.
    after = translated[current.end():current.end() + 16]
    if re.match(r"\s*\d+\s*(?:천|백|십)?\s*(?:만|억)", after):
        return translated
    if current.group(0).replace(" ", "") == expected:
        return translated
    # Only change a single simple Korean scale amount.
    return translated[:current.start()] + expected + translated[current.end():]



def find_saved_device_index(devices, choice):
    """Look up a playback loopback or mic by its stable name, never by index."""
    if not isinstance(choice, dict) or not isinstance(choice.get("name"), str):
        return None
    if not isinstance(choice.get("loopback"), bool):
        return None
    return next(
        (idx for idx, dev in enumerate(devices)
         if len(dev) > 3
         and dev[3] == choice["name"]
         and dev[2] is choice["loopback"]),
        None,
    )


def find_strongest_audio(results, min_rms=0.001):
    """Select the strongest valid signal, not a silent or broken virtual device."""
    valid = [
        item for item in results
        if "error" not in item
        and not item.get("warning")
        and isinstance(item.get("rms"), (int, float))
        and math.isfinite(item["rms"])
        and item["rms"] >= min_rms
    ]
    return max(valid, key=lambda item: item["rms"], default=None)


class App:
    def __init__(self, root):
        self.root = root
        root.title(f"LiveTranslator v{updater.current_version()} - 실시간 한국어 자막")
        root.geometry("880x710")
        self.events = queue.Queue()
        self.stop_signal = threading.Event()
        self.worker = None
        self._cached_model = None
        self._cached_model_key = None
        self._model_check_busy = False
        self._audio_scan_busy = False
        self._auto_start_pending = False
        self._needs_auto_source = False
        self.running = False
        self.devices = []
        self.settings = self.load_settings()
        self.device = tk.StringVar()
        self.language = tk.StringVar(value=self.settings.get("language", "en"))
        self.model = tk.StringVar(value=self.settings.get("model", "base"))
        self.seconds = tk.StringVar(value=str(self.settings.get("seconds", 3)))
        self.save_enabled = tk.BooleanVar(value=self.settings.get("save", False))
        self.engine = tk.StringVar(value=self.settings.get("engine", "argos"))
        self.overlay_enabled = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="장치를 선택하고 번역을 시작하세요.")
        self.audio_meter = tk.DoubleVar(value=0.0)
        self.audio_state = tk.StringVar(value="입력 신호: 번역 시작 후 확인")
        self.last_audio_meter_at = None
        self.capture_started_at = None
        self.last_original_at = None
        self.build_ui()
        self.refresh()
        self.root.after(100, self.consume)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def load_settings(self):
        try:
            return json.loads(SETTINGS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def persist(self):
        SETTINGS.write_text(json.dumps({
            "language": self.language.get(), "model": self.model.get(),
            "seconds": self.seconds.get(), "save": self.save_enabled.get(),
            "engine": self.engine.get(),
            "audio_device": self.settings.get("audio_device"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def remember_selected_device(self, event=None):
        """Save a user-selected or confirmed audio source immediately."""
        if not self.devices or self.device_combo.current() < 0:
            return
        selected = self.devices[self.device_combo.current()]
        self.settings["audio_device"] = {
            "name": selected[3],
            "loopback": bool(selected[2]),
        }
        self._needs_auto_source = False
        self.persist()

    def build_ui(self):
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"LiveTranslator v{updater.current_version()}", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(frame, text="PC 출력 장치(WASAPI Loopback) 또는 마이크를 선택해 한국어로 번역합니다.").pack(anchor="w", pady=(5, 16))
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=5)
        ttk.Label(row, text="오디오 장치", width=13).pack(side="left")
        self.device_combo = ttk.Combobox(row, textvariable=self.device, state="readonly")
        self.device_combo.pack(side="left", fill="x", expand=True, padx=5)
        self.device_combo.bind("<<ComboboxSelected>>", self.remember_selected_device)
        ttk.Button(row, text="새로고침", command=self.refresh).pack(side="left")
        ttk.Button(row, text="소리 찾기", command=self.scan_audio).pack(side="left", padx=(5, 0))
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text="원본 언어", width=13).pack(side="left")
        ttk.Combobox(row, values=["en", "ja", "ko", "auto"], state="readonly", textvariable=self.language, width=10).pack(side="left")
        ttk.Label(row, text="   음성 인식 모델").pack(side="left")
        ttk.Combobox(row, values=["tiny", "base", "small"], state="readonly", textvariable=self.model, width=9).pack(side="left")
        ttk.Label(row, text="   인식 간격(초)").pack(side="left")
        ttk.Combobox(row, values=["2", "3", "4", "5", "6", "8"], state="readonly", textvariable=self.seconds, width=6).pack(side="left")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(3, 8))
        ttk.Label(row, text="번역 엔진", width=13).pack(side="left")
        ttk.Radiobutton(row, text="Argos 고품질 (오프라인)", variable=self.engine, value="argos").pack(side="left", padx=(0, 12))
        ttk.Radiobutton(row, text="Google Cloud (온라인)", variable=self.engine, value="google").pack(side="left")
        ttk.Button(row, text="Google API 키 설정", command=self.configure_google_key).pack(side="left", padx=(15, 0))
        ttk.Button(row, text="모델 진단", command=self.diagnose_models).pack(side="left", padx=(9, 0))
        ttk.Label(frame, text="Google 모드에서는 인식된 텍스트만 Google에 전송되며 API 사용 요금이 발생할 수 있습니다.",
                  foreground="#855800").pack(anchor="w", pady=(0, 4))
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(8, 14))
        ttk.Checkbutton(row, text="번역 내역 TXT 저장", variable=self.save_enabled).pack(side="left")
        ttk.Checkbutton(row, text="항상 위에 자막", variable=self.overlay_enabled, command=self.toggle_overlay).pack(side="left", padx=12)
        ttk.Button(row, text="기록 폴더", command=lambda: messagebox.showinfo("저장 위치", str(TRANSCRIPTS))).pack(side="right")
        ttk.Button(row, text="업데이트 확인", command=self.check_update).pack(side="right", padx=10)
        ttk.Button(row, text="자막 지우기", command=self.clear_output).pack(side="right", padx=5)
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=4)
        self.start_btn = ttk.Button(row, text="▶ 번역 시작", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(row, text="■ 중지", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=7)
        ttk.Button(row, text="영어→한국어 모델 설치", command=lambda: self.install_pack("en")).pack(side="left", padx=3)
        ttk.Button(row, text="일본어→한국어 모델 설치", command=lambda: self.install_pack("ja")).pack(side="left", padx=3)
        ttk.Label(frame, textvariable=self.status, foreground="#215ba5").pack(anchor="w", pady=(10, 4))
        meter_row = ttk.Frame(frame)
        meter_row.pack(fill="x", pady=(0, 9))
        ttk.Label(meter_row, text="오디오 입력", width=13).pack(side="left")
        ttk.Progressbar(meter_row, variable=self.audio_meter, maximum=100, length=145).pack(side="left", padx=(0, 10))
        self.audio_meter_label = ttk.Label(meter_row, textvariable=self.audio_state)
        self.audio_meter_label.pack(side="left", fill="x", expand=True)
        ttk.Label(frame, text="원문", font=("Malgun Gothic", 11, "bold")).pack(anchor="w")
        self.original = tk.Text(frame, height=8, wrap="word")
        self.original.pack(fill="both", expand=True, pady=(4, 10))
        ttk.Label(frame, text="한국어 번역", font=("Malgun Gothic", 11, "bold")).pack(anchor="w")
        self.translated = tk.Text(frame, height=8, wrap="word")
        self.translated.pack(fill="both", expand=True, pady=(4, 0))
        self.overlay = tk.Toplevel(self.root)
        self.overlay.title("LiveTranslator 자막")
        self.overlay.geometry("950x150+150+650")
        self.overlay.configure(bg="#121820")
        self.overlay.attributes("-topmost", True)
        self.overlay_text = tk.Label(self.overlay, text="한국어 자막", font=("Malgun Gothic", 22, "bold"), bg="#121820", fg="white", wraplength=900)
        self.overlay_text.pack(fill="both", expand=True)
        self.overlay.protocol("WM_DELETE_WINDOW", self.hide_overlay)
        self.overlay.withdraw()

    def clear_output(self):
        """Clear displayed captions without stopping capture or deleting TXT logs."""
        self.original.delete("1.0", "end")
        self.translated.delete("1.0", "end")
        self.overlay_text.config(text="")
        retained = []
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] not in ("original", "result"):
                    retained.append(event)
        except queue.Empty:
            pass
        for event in retained:
            self.events.put_nowait(event)
        self.status.set("화면 자막을 지웠습니다. 번역이 실행 중이면 다음 자막부터 다시 표시됩니다.")

    def hide_overlay(self):
        self.overlay_enabled.set(False)
        self.overlay.withdraw()

    def toggle_overlay(self):
        if self.overlay_enabled.get():
            self.overlay.deiconify()
        else:
            self.overlay.withdraw()

    def refresh(self):
        # Keep a stable name so Bluetooth and virtual device index changes do not
        # silently switch the capture source. Use the currently active output
        # as the first/default option for new users.
        old_name = None
        old_loopback = None
        if self.devices and self.device_combo.current() >= 0:
            old = self.devices[self.device_combo.current()]
            old_name = old[3] if len(old) > 3 else None
            old_loopback = old[2]
        devices = []
        p = pyaudio.PyAudio()
        try:
            loopbacks = [
                info for info in p.get_loopback_device_info_generator()
                if int(info.get("maxInputChannels", 0)) > 0
            ]
            default_index = None
            try:
                default = p.get_default_wasapi_loopback()
                default_index = int(default["index"])
            except (OSError, LookupError, AttributeError, KeyError, TypeError):
                pass
            loopbacks.sort(key=lambda info: 0 if int(info["index"]) == default_index else 1)
            for info in loopbacks:
                name = str(info["name"])
                is_default = int(info["index"]) == default_index
                label = ("🔊 기본 출력 | " if is_default else "🔊 출력 | ") + name
                devices.append((label, int(info["index"]), True, name))
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0 and not info.get("isLoopbackDevice", False):
                    name = str(info["name"])
                    devices.append((f"🎤 입력 | {name}", int(info["index"]), False, name))
        except Exception as exc:
            self.status.set(f"오디오 장치 조회 오류: {exc}")
            return
        finally:
            p.terminate()
        self.devices = devices
        self.device_combo["values"] = [d[0] for d in devices]
        if not devices:
            self.status.set("오디오 장치를 찾을 수 없습니다. Windows 출력/입력 장치를 확인하세요.")
            return
        # Keep a current in-session selection. Otherwise restore the last
        # working device, which survives EXE restarts and PortAudio reindexing.
        match = next(
            (i for i, device in enumerate(devices) if device[3] == old_name and device[2] == old_loopback),
            None,
        )
        saved = self.settings.get("audio_device")
        saved_index = find_saved_device_index(devices, saved)
        # A persisted choice has priority over a temporary/default selection.
        chosen = saved_index if saved_index is not None else match
        self.device_combo.current(chosen if chosen is not None else 0)
        self._needs_auto_source = saved_index is None and match is None
        if chosen is None:
            self.status.set(
                "저장된 오디오 출력이 없거나 연결이 끊어졌습니다. "
                "번역 시작 시 실제 소리가 나는 출력 장치를 자동으로 찾습니다."
            )
        else:
            self.status.set(f"오디오 장치 복원 완료: {devices[chosen][3]}")

    def scan_audio(self, auto_start=False):
        """Search actual playback loopbacks; optionally start after detection."""
        if self.running:
            messagebox.showinfo("오디오 검사", "번역을 중지한 뒤 소리 찾기를 실행해 주세요.")
            return
        if self._audio_scan_busy:
            return
        if not self.devices:
            messagebox.showerror("오디오 검사", "오디오 장치를 먼저 새로고침해 주세요.")
            return
        selected_idx = self.device_combo.current()
        if selected_idx < 0:
            return
        selected = self.devices[selected_idx]
        self._audio_scan_busy = True
        self._auto_start_pending = bool(auto_start)
        if auto_start:
            self.status.set("오디오 자동 감지 중 · 영상 소리가 나는 출력 장치를 찾는 중...")
        else:
            self.status.set("오디오 신호 검사 중 · 유튜브 영상을 소리와 함께 재생해 주세요...")
        devices = list(self.devices)

        def worker():
            try:
                from streaming_pipeline import find_active_loopback, probe_with_timeout
                if selected[2]:
                    results = find_active_loopback(
                        devices, seconds=1.0,
                        progress=lambda i, count, label: self.events.put(
                            ("status", f"오디오 검사 중 ({i}/{count}) · {label}")
                        ),
                    )
                else:
                    self.events.put(("status", "선택한 마이크 신호 확인 중..."))
                    result = probe_with_timeout(selected, seconds=1.5, timeout=3.5)
                    results = [{"device": selected, **result}]
                self.events.put(("scan_completed", results))
            except Exception as exc:
                self.events.put(("scan_failed", str(exc)))

        threading.Thread(target=worker, name="AudioSignalProbe", daemon=True).start()

    def use_detected_audio_and_start(self, results):
        """Pick a proven playback source, persist it, then start translation."""
        best = find_strongest_audio(results)
        if best is None:
            self.status.set("오디오 자동 감지 실패 · 재생 중인 영상 소리가 감지되지 않았습니다.")
            messagebox.showwarning(
                "오디오 자동 감지",
                "소리가 감지되지 않았습니다. 영상을 재생하고 Windows 출력 장치를 확인해 주세요.\n"
                "필요하면 '소리 찾기'에서 검사 결과를 볼 수 있습니다.",
            )
            return
        index = next(
            (i for i, device in enumerate(self.devices) if device == best["device"]),
            None,
        )
        if index is None:
            self.status.set("출력 장치 목록이 바뀌었습니다. 새로고침 후 다시 시도하세요.")
            return
        self.device_combo.current(index)
        self.remember_selected_device()
        self.status.set(
            f"오디오 자동 감지 완료: {best['device'][3]} "
            "· 이 장치를 다음 실행에도 사용합니다."
        )
        self.root.after(50, self.start)

    def show_audio_scan(self, results):
        """Never automatically switch audio sources without confirmation."""
        if not results:
            messagebox.showinfo("오디오 검사", "검사할 오디오 장치가 없습니다.")
            return
        lines = []
        best = None
        for item in results:
            label = item["device"][0]
            if "error" in item:
                lines.append(f"· {label}: {item['error']}")
                continue
            if item.get("warning"):
                lines.append(f"· {label}: {item['warning']}")
                continue
            rms = float(item["rms"])
            db = 20.0 * math.log10(max(rms, 1e-7))
            lines.append(f"· {label}: {db:.0f} dBFS")
            if best is None or rms > best["rms"]:
                best = item
        best = find_strongest_audio(results)
        playing = best is not None
        details = "\n".join(lines[:12])
        if not playing:
            self.status.set("오디오 검사 완료 · 소리 신호가 감지되지 않았습니다.")
            messagebox.showwarning(
                "오디오 검사",
                "현재 선택 가능한 장치에서 뚜렷한 소리를 감지하지 못했습니다.\n"
                "유튜브를 재생한 채 Windows 볼륨 믹서에서 Chrome 출력 장치를 확인해 주세요.\n\n"
                + details,
            )
            return
        best_name = best["device"][3] if len(best["device"]) > 3 else best["device"][0]
        self.status.set(f"오디오 검사 완료 · 신호 감지: {best_name}")
        current = self.device_combo.current()
        current_device = self.devices[current] if current >= 0 else None
        if current_device != best["device"]:
            if messagebox.askyesno(
                "오디오 장치 추천",
                f"가장 큰 소리 신호가 감지된 장치:\n{best_name}\n\n"
                f"이 장치로 변경할까요?\n\n{details}",
            ):
                match = next(
                    (i for i, item in enumerate(self.devices) if item == best["device"]),
                    None,
                )
                if match is not None:
                    self.device_combo.current(match)
                    self.remember_selected_device()
                    self.status.set(f"오디오 장치 저장 완료: {best_name} · 다음 실행부터 자동 선택합니다.")
        else:
            self.remember_selected_device()
            messagebox.showinfo(
                "오디오 검사",
                f"선택한 장치에서 소리를 감지했습니다.\n"
                "이제 번역 시작을 눌러주세요.\n\n" + details,
            )

    def configure_google_key(self):
        if self.running:
            messagebox.showinfo("Google API 키", "번역을 중지한 후 API 키를 설정하세요.")
            return
        key = simpledialog.askstring(
            "Google Cloud Translation API 키",
            "Google Cloud Translation Basic(v2) API 키를 입력하세요.\n"
            "이 키는 Windows 사용자 계정으로 암호화 저장되며 GitHub에 올라가지 않습니다.\n"
            "빈 값으로 저장하면 기존 키가 삭제됩니다.",
            parent=self.root, show="*",
        )
        if key is None:
            return
        try:
            if key.strip():
                store_google_key(key)
                messagebox.showinfo("Google API 키", "API 키를 Windows에 암호화 저장했습니다.")
            else:
                GOOGLE_KEY_FILE.unlink(missing_ok=True)
                messagebox.showinfo("Google API 키", "저장된 API 키를 삭제했습니다.")
        except Exception as exc:
            messagebox.showerror("Google API 키", f"키 저장 실패: {exc}")

    def diagnose_models(self):
        """Check selected speech and translation models without downloading new ones."""
        if self.running:
            messagebox.showinfo(
                "모델 진단",
                "번역 중입니다. Whisper 모델은 이미 로드됐습니다.\n"
                "먼저 중지하면 Argos 모델도 검사할 수 있습니다.",
            )
            return
        if self._model_check_busy:
            return
        self._model_check_busy = True
        model_name = self.model.get()
        lang = self.language.get()
        self.status.set("모델 진단 중 · 새 모델을 다운로드하지 않습니다...")

        def job():
            notes = []
            try:
                local = APP_DIR / "models" / model_name
                files = ("model.bin", "config.json", "tokenizer.json")
                complete = all((local / filename).is_file() for filename in files)
                if self._cached_model is not None and self._cached_model_key:
                    notes.append("Whisper: 이 프로그램에서 로드에 성공한 모델이 있습니다 (정상)")
                elif complete:
                    notes.append(f"Whisper {model_name}: 로컬 파일 확인됨. 실제 로딩은 번역 시작 시 검증")
                else:
                    notes.append(
                        f"Whisper {model_name}: 별도 모델 폴더에는 없음. "
                        "Hugging Face 캐시에 있을 수 있으므로 미설치로 단정할 수 없습니다."
                    )
                import argostranslate.translate
                if lang == "ko":
                    notes.append("Argos: 원본 언어가 한국어이므로 번역 모델이 필요하지 않습니다.")
                else:
                    for source in (("en", "ja") if lang == "auto" else (lang,)):
                        try:
                            translator = argostranslate.translate.get_translation_from_codes(source, "ko")
                            if translator is None:
                                notes.append(f"Argos {source}→ko: 미설치 또는 인식 불가. 모델 설치 버튼을 사용하세요.")
                                continue
                            test_phrase = "Hello." if source == "en" else "こんにちは。"
                            translated = translator.translate(test_phrase)
                            if not isinstance(translated, str) or not translated.strip():
                                raise RuntimeError("번역 결과가 비어 있습니다.")
                            notes.append(f"Argos {source}→ko: 테스트 번역 성공 · {translated.strip()[:80]}")
                        except Exception as exc:
                            notes.append(f"Argos {source}→ko: 번역 실패 · {type(exc).__name__}: {exc}")
            except Exception as exc:
                notes.append(f"모델 진단 실패 · {type(exc).__name__}: {exc}")
            finally:
                self.events.put(("model_report", "\n".join(notes)))

        threading.Thread(target=job, daemon=True, name="ModelDiagnostic").start()

    def install_pack(self, lang):
        if self.running:
            messagebox.showwarning("번역 중", "먼저 번역을 중지하세요.")
            return
        self.status.set(f"{lang}→ko 언어 모델 설치 중...")
        def task():
            try:
                import argostranslate.package
                available = argostranslate.package.get_available_packages()
                chosen = next(p for p in available if p.from_code == lang and p.to_code == "ko")
                path = chosen.download()
                argostranslate.package.install_from_path(path)
                self.events.put(("status", f"{lang}→ko 번역 모델 설치 완료"))
            except Exception as exc:
                self.events.put(("error", f"모델 설치 실패: {exc}"))
        threading.Thread(target=task, daemon=True).start()

    def start(self):
        if self.running:
            return
        if not self.devices or self.device_combo.current() < 0:
            messagebox.showerror("오디오", "장치를 먼저 선택하세요.")
            return
        selected = self.devices[self.device_combo.current()]
        if self._audio_scan_busy:
            messagebox.showinfo("오디오 검사", "진행 중인 소리 찾기가 끝난 뒤 번역을 시작해 주세요.")
            return
        # On a first launch (or when a saved device is disconnected), find
        # the real active playback loopback once, without requiring a click on
        # '소리 찾기'. Manual mic selections bypass automatic output scanning.
        if selected[2] and self._needs_auto_source:
            self.scan_audio(auto_start=True)
            return
        engine = self.engine.get()
        if engine not in ("argos", "google"):
            messagebox.showerror("번역 엔진", "번역 엔진을 선택하세요.")
            return
        api_key = None
        if engine == "google":
            try:
                api_key = read_google_key()
            except Exception:
                messagebox.showerror("Google API 키", "저장된 API 키를 읽을 수 없습니다. 다시 등록하세요.")
                return
            if not api_key:
                messagebox.showinfo("Google API 키", "먼저 'Google API 키 설정'에서 본인 Google Cloud API 키를 등록하세요.")
                return
            if not messagebox.askyesno("온라인 번역 확인",
                "Google Cloud로 인식된 음성의 텍스트가 전송됩니다.\n"
                "결제 계정의 API 요금이 발생할 수 있습니다.\n"
                "회사 보안 정책 및 녹음 동의 여부를 확인했나요?"):
                return
        self.persist()
        self.stop_signal = threading.Event()
        self.running = True
        self.audio_meter.set(0)
        self.audio_state.set("입력 신호: 연결 대기 중...")
        self.last_audio_meter_at = None
        self.capture_started_at = None
        self.last_original_at = None
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.set("음성 인식 모델을 초기화하는 중...")
        self.worker = threading.Thread(target=self.capture_and_translate, args=(selected, self.language.get(), self.model.get(), int(self.seconds.get()), engine, api_key), daemon=True)
        self.worker.start()

    def stop(self):
        if not self.running:
            return
        self.stop_signal.set()
        self.stop_btn.config(state="disabled")
        self.status.set("중지 요청됨. 진행 중인 AI 처리가 끝나면 종료됩니다.")

    def capture_and_translate(self, selected, language, model_name, seconds, engine, api_key):
        try:
            from faster_whisper import WhisperModel
            if engine == "argos":
                import argostranslate.translate
            from streaming_pipeline import run_stream

            local_model = APP_DIR / "models" / model_name
            path = str(local_model) if all(
                (local_model / name).is_file()
                for name in ("model.bin", "config.json", "tokenizer.json")
            ) else model_name
            if self._cached_model is None or self._cached_model_key != path:
                self.events.put(("status", "음성 인식 모델을 로드하는 중..."))
                self._cached_model = WhisperModel(
                    path, device="cpu", compute_type="int8", cpu_threads=6
                )
                self._cached_model_key = path
            self.events.put(("model_loaded", model_name))
            if self.stop_signal.is_set():
                return
            translators = {}

            def translate_cached(original, lang):
                if engine == "google":
                    return google_translate_text(original, lang, api_key)
                if lang not in translators:
                    translator = argostranslate.translate.get_translation_from_codes(lang, "ko")
                    if translator is None:
                        raise RuntimeError(f"{lang}→ko 번역 모델이 없습니다. 먼저 설치해 주세요.")
                    translators[lang] = translator
                candidate = translators[lang].translate(original)
                return guard_argos_quantity(original, candidate)

            run_stream(
                selected=selected,
                interval_seconds=seconds,
                language=language,
                model=self._cached_model,
                translate=translate_cached,
                quality_mode=(engine == "argos"),
                emit_status=lambda status: self.events.put(
                    ("error" if status.startswith("오디오 캡처 오류:") else "status", status)
                ),
                emit_audio_level=lambda rms, peak: self.events.put(("audio_level", (rms, peak))),
                emit_original=lambda original: self.events.put(("original", original)),
                emit_result=lambda original, translated: self.events.put(("result", (original, translated))),
                stop_event=self.stop_signal,
            )
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            self.events.put(("ended", None))

    def check_update(self):
        if self.running:
            messagebox.showinfo("업데이트", "번역을 중지한 후 업데이트해 주세요.")
            return
        self.status.set("GitHub에서 최신 업데이트 확인 중...")
        def job():
            try:
                info = updater.find_update()
                self.events.put(("update_found", info))
            except Exception as exc:
                self.events.put(("error", f"업데이트 확인 실패: {exc}"))
        threading.Thread(target=job, name="UpdateCheck", daemon=True).start()

    def apply_update(self, info):
        self.status.set(f"v{info['version']} 코드 다운로드 및 검증 중...")
        def job():
            try:
                version = updater.install_update(info)
                self.events.put(("update_ready", version))
            except Exception as exc:
                self.events.put(("error", f"업데이트 적용 실패: {exc}"))
        threading.Thread(target=job, name="UpdateInstall", daemon=True).start()

    def restart_app(self):
        try:
            if getattr(sys, "frozen", False):
                command = [sys.executable]
                working = str(Path(sys.executable).parent)
            else:
                command = [sys.executable, str(Path(__file__).resolve().parent / "launcher.py")]
                working = str(Path(__file__).resolve().parent)
            env = os.environ.copy()
            env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
            subprocess.Popen(command, cwd=working, env=env)
            self.root.destroy()
        except Exception as exc:
            messagebox.showerror("재시작 오류", str(exc))

    def consume(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "status":
                    self.status.set(data)
                    if data.startswith("오디오 연결 완료:"):
                        self.capture_started_at = time.monotonic()
                elif kind == "scan_completed":
                    self._audio_scan_busy = False
                    auto = self._auto_start_pending
                    self._auto_start_pending = False
                    if auto:
                        self.use_detected_audio_and_start(data)
                    else:
                        self.show_audio_scan(data)
                elif kind == "scan_failed":
                    self._audio_scan_busy = False
                    self._auto_start_pending = False
                    self.status.set("오디오 신호 검사 실패")
                    messagebox.showerror("오디오 검사 실패", data)
                elif kind == "model_loaded":
                    self.status.set(f"Whisper {data} 모델 로드 성공 · 오디오 수집 준비 중...")
                elif kind == "model_report":
                    self._model_check_busy = False
                    self.status.set("모델 진단 완료 · 결과 팝업을 확인하세요.")
                    messagebox.showinfo("LiveTranslator 모델 진단", data)
                elif kind == "audio_level":
                    rms, peak = data
                    self.last_audio_meter_at = time.monotonic()
                    dbfs = 20 * math.log10(max(float(rms), 1e-7))
                    level = max(0, min(100, int((dbfs + 65) / 55 * 100)))
                    self.audio_meter.set(level)
                    if rms < 0.0005:
                        state = "거의 무음 · 소리 출력 장치가 맞는지 확인"
                    elif rms < 0.002:
                        state = "매우 작은 신호 · Windows 출력 음량 확인"
                    else:
                        state = "오디오 입력 정상"
                    if (rms >= 0.0005 and self.last_original_at is None
                            and self.capture_started_at is not None
                            and time.monotonic() - self.capture_started_at > 15):
                        state += " · 음성 인식이 오래 걸림"
                    self.audio_state.set(f"{state} ({dbfs:.0f} dBFS)")
                elif kind == "update_found":
                    if data is None:
                        self.status.set(f"최신 버전입니다. (v{updater.current_version()})")
                    elif messagebox.askyesno("LiveTranslator 업데이트",
                        f"새 버전 v{data['version']}을 설치할까요?\n"
                        "모델, 설정, 번역 기록은 그대로 유지됩니다."):
                        self.apply_update(data)
                    else:
                        self.status.set("업데이트 취소됨")
                elif kind == "update_ready":
                    self.status.set(f"v{data} 업데이트 준비 완료")
                    if messagebox.askyesno("업데이트 완료",
                        f"코드 업데이트 v{data} 다운로드를 완료했습니다.\n지금 재시작할까요?"):
                        self.restart_app()
                elif kind == "error":
                    self.status.set(f"오류: {data}")
                elif kind == "original":
                    self.last_original_at = time.monotonic()
                    self.original.insert("end", data + "\n")
                    self.original.see("end")
                elif kind == "result":
                    original, translated = data
                    self.translated.insert("end", translated + "\n")
                    self.translated.see("end")
                    self.overlay_text.config(text=translated)
                    if self.save_enabled.get():
                        filename = TRANSCRIPTS / (datetime.date.today().isoformat() + ".txt")
                        with filename.open("a", encoding="utf-8") as f:
                            f.write(f"[{datetime.datetime.now():%H:%M:%S}] {original}\n{translated}\n\n")
                elif kind == "ended":
                    self.running = False
                    self.capture_started_at = None
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    if self.stop_signal.is_set() and not self.status.get().startswith(("오류:", "오디오 캡처 오류:")):
                        self.status.set("중지 완료")
        except queue.Empty:
            pass
        if (self.running and self.capture_started_at is not None
                and (self.last_audio_meter_at is None
                     or time.monotonic() - self.last_audio_meter_at > 5)
                and time.monotonic() - self.capture_started_at > 5):
            self.audio_state.set("오디오 데이터 수신 없음 · 다른 출력 장치 선택 또는 드라이버 확인")
            self.audio_meter.set(0)
        self.root.after(100, self.consume)

    def close(self):
        if self.running:
            self.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
