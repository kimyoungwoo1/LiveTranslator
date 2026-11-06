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
HYM_DIR = APP_DIR / "hy_mt2"


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
        self._hym_install_busy = False
        self._hym_install_cancel = threading.Event()
        self._auto_start_pending = False
        self.running = False
        self.devices = []
        self.all_devices = []
        self.settings = self.load_settings()
        self.mode = tk.StringVar(value="output" if (self.settings.get("audio_device") or {}).get("loopback", True) else "input")
        self.device = tk.StringVar()
        self.language = tk.StringVar(value=self.settings.get("language", "en"))
        self.model = tk.StringVar(value=self.settings.get("model", "base"))
        self.seconds = tk.StringVar(value=str(self.settings.get("seconds", 3)))
        self.save_enabled = tk.BooleanVar(value=self.settings.get("save", False))
        self.engine = tk.StringVar(value=self.settings.get("engine", "hymt"))
        self.overlay_enabled = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="장치를 선택하고 번역을 시작하세요.")
        self.audio_meter = tk.DoubleVar(value=0.0)
        self.audio_state = tk.StringVar(value="입력 신호: 번역 시작 후 확인")
        self.last_audio_meter_at = None
        self.capture_started_at = None
        self.last_original_at = None
        self.session_started_at = None
        self.session_record_path = None
        self.session_summary_path = None
        self.session_ko_lines = []
        self.summary_running = False
        self.session_engine = None
        self.session_language = None
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
            "audio_devices_by_mode": self.settings.get("audio_devices_by_mode", {}),
            "hym_server": self.settings.get("hym_server"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def remember_selected_device(self, event=None):
        """Save a user-selected or confirmed audio source immediately."""
        if not self.devices or self.device_combo.current() < 0:
            return
        selected = self.devices[self.device_combo.current()]
        choice = {"name": selected[3], "loopback": bool(selected[2])}
        self.settings["audio_device"] = choice
        self.settings.setdefault("audio_devices_by_mode", {})[
            "output" if selected[2] else "input"
        ] = choice
        self.persist()

    def build_ui(self):
        """Compact dark dashboard using only the bundled Tkinter/ttk runtime."""
        screen_h = max(650, int(self.root.winfo_screenheight()))
        screen_w = max(980, int(self.root.winfo_screenwidth()))
        initial_h = max(650, min(1050, screen_h - 105))
        initial_w = max(980, min(1260, screen_w - 55))
        self.root.geometry(f"{initial_w}x{initial_h}")
        self.root.minsize(920, min(700, initial_h))
        colors = {
            "bg": "#0c1220",
            "card": "#161f31",
            "field": "#212e44",
            "text": "#f1f5fb",
            "muted": "#a0aec3",
            "border": "#2d3a50",
            "blue": "#468bff",
            "blue_hover": "#3475eb",
            "green": "#36d2a0",
            "red": "#f1848a",
        }
        self.root.configure(bg=colors["bg"])
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("LT.TFrame", background=colors["bg"])
        style.configure("LTCard.TFrame", background=colors["card"])
        style.configure("LT.TLabel", background=colors["bg"],
                        foreground=colors["text"], font=("Malgun Gothic", 10))
        style.configure("LTMuted.TLabel", background=colors["bg"],
                        foreground=colors["muted"], font=("Malgun Gothic", 9))
        style.configure("LTCard.TLabel", background=colors["card"],
                        foreground=colors["text"], font=("Malgun Gothic", 10))
        style.configure("LTCardMuted.TLabel", background=colors["card"],
                        foreground=colors["muted"], font=("Malgun Gothic", 9))
        style.configure("LTRadio.TRadiobutton", background=colors["card"],
                        foreground=colors["text"], font=("Malgun Gothic", 10))
        style.configure("LTCheck.TCheckbutton", background=colors["bg"],
                        foreground=colors["text"], font=("Malgun Gothic", 10))
        style.map("LTRadio.TRadiobutton", background=[("active", colors["card"])],
                  foreground=[("active", colors["text"])])
        style.map("LTCheck.TCheckbutton", background=[("active", colors["bg"])],
                  foreground=[("active", colors["text"])])
        style.configure("LT.TCombobox", fieldbackground=colors["field"],
                        background=colors["field"], foreground=colors["text"],
                        arrowcolor=colors["text"], bordercolor=colors["border"],
                        lightcolor=colors["border"], darkcolor=colors["border"],
                        padding=5)
        style.map("LT.TCombobox",
                  fieldbackground=[("readonly", colors["field"])],
                  foreground=[("readonly", colors["text"])])
        style.configure("LT.TButton", font=("Malgun Gothic", 9),
                        padding=(11, 8), background=colors["field"],
                        foreground=colors["text"], borderwidth=0)
        style.map("LT.TButton", background=[("active", "#334562")])
        style.configure("LTPrimary.TButton", font=("Malgun Gothic", 10, "bold"),
                        padding=(18, 10), background=colors["blue"],
                        foreground="#ffffff", borderwidth=0)
        style.map("LTPrimary.TButton",
                  background=[("active", colors["blue_hover"]),
                              ("disabled", "#37475d")],
                  foreground=[("disabled", "#8e99ab")])
        style.configure("LTDanger.TButton", font=("Malgun Gothic", 9, "bold"),
                        padding=(13, 8), background="#593037",
                        foreground="#ffdde0", borderwidth=0)
        style.map("LTDanger.TButton", background=[("active", "#74383f")])
                        
        outer = ttk.Frame(self.root, style="LT.TFrame", padding=(20, 15, 20, 14))
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="LT.TFrame")
        header.pack(fill="x", pady=(0, 11))
        title_col = ttk.Frame(header, style="LT.TFrame")
        title_col.pack(side="left", fill="x", expand=True)
        ttk.Label(title_col, text="LiveTranslator", style="LT.TLabel",
                  font=("Segoe UI", 22, "bold")).pack(anchor="w")
        ttk.Label(title_col, text=f"v{updater.current_version()}  ·  PC 음성을 실시간 한국어 자막으로",
                  style="LTMuted.TLabel").pack(anchor="w", pady=(2, 0))
        self.live_label = tk.Label(header, text="● 대기 중", font=("Malgun Gothic", 10, "bold"),
                                   bg=colors["card"], fg=colors["green"],
                                   padx=14, pady=8, bd=0)
        self.live_label.pack(side="right", pady=(8, 0))

        # Large, clickable cards are easier to distinguish than plain radios.
        mode_card = tk.Frame(outer, bg=colors["card"], padx=14, pady=8)
        mode_card.pack(fill="x", pady=(0, 9))
        mode_card.grid_columnconfigure(1, weight=1, uniform="mode")
        mode_card.grid_columnconfigure(2, weight=1, uniform="mode")
        intro = tk.Frame(mode_card, bg=colors["card"], width=165)
        intro.grid(row=0, column=0, sticky="nw", padx=(2, 12), pady=(8, 0))
        tk.Label(intro, text="◉  사용 모드", bg=colors["card"], fg=colors["text"],
                 font=("Malgun Gothic", 12, "bold")).pack(anchor="w")
        tk.Label(intro, text="어떤 소리를 번역할까요?", bg=colors["card"],
                 fg=colors["muted"], font=("Malgun Gothic", 9)).pack(anchor="w", pady=(5, 0))
        self.mode_tiles = {}
        for column, key, icon, heading, detail in (
            (1, "output", "▣", "온라인 회의 · PC 출력 번역",
             "Teams, Zoom, 영상의 재생 소리"),
            (2, "input", "♩", "오프라인 회의 · 마이크 번역",
             "발표자, 회의실 등 주변 음성"),
        ):
            shell = tk.Frame(mode_card, bg=colors["border"], padx=2, pady=2, cursor="hand2")
            shell.grid(row=0, column=column, sticky="nsew",
                       padx=(0, 9) if column == 1 else (0, 0))
            inner = tk.Frame(shell, bg=colors["field"], padx=12, pady=7)
            inner.pack(fill="both", expand=True)
            line = tk.Frame(inner, bg=colors["field"])
            line.pack(fill="x")
            symbol = tk.Label(line, text=icon, bg=colors["field"], fg=colors["text"],
                              font=("Segoe UI Symbol", 16))
            symbol.pack(side="left", padx=(0, 8))
            name = tk.Label(line, text=heading, bg=colors["field"], fg=colors["text"],
                            font=("Malgun Gothic", 10, "bold"), anchor="w")
            name.pack(side="left", fill="x", expand=True)
            mark = tk.Label(line, text="○", bg=colors["field"], fg=colors["muted"],
                            font=("Segoe UI Symbol", 13))
            mark.pack(side="right")
            desc = tk.Label(inner, text=detail, bg=colors["field"],
                            fg=colors["muted"], font=("Malgun Gothic", 9),
                            anchor="w")
            desc.pack(fill="x", pady=(6, 0))
            self.mode_tiles[key] = (shell, inner, line, symbol, name, mark, desc)
            for widget in (shell, inner, line, symbol, name, mark, desc):
                widget.bind("<Button-1>", lambda event, selected=key:
                            self.select_audio_mode(selected))
        self.mode_description = tk.StringVar()
        self._update_mode_ui()

        source = tk.Frame(outer, bg=colors["card"], padx=15, pady=8)
        source.pack(fill="x", pady=(0, 9))
        heading_row = tk.Frame(source, bg=colors["card"])
        heading_row.pack(fill="x", pady=(0, 8))
        tk.Label(heading_row, text="◖  입력 소스", bg=colors["card"],
                 fg=colors["text"], font=("Malgun Gothic", 11, "bold")
                 ).pack(side="left", padx=(0, 12))
        tk.Label(heading_row, text="번역할 오디오 장치를 선택하세요",
                 bg=colors["card"], fg=colors["muted"],
                 font=("Malgun Gothic", 9)).pack(side="left")
        self.source_kind_label = tk.Label(
            heading_row, text="", bg=colors["card"], fg=colors["blue"],
            font=("Malgun Gothic", 9, "bold"))
        self.source_kind_label.pack(side="right")
        row = tk.Frame(source, bg=colors["card"])
        row.pack(fill="x", pady=(0, 7))
        self.source_tabs = {}
        for key, label in (("output", "◖  PC 출력"), ("input", "♩  마이크 입력")):
            tab = tk.Label(row, text=label, cursor="hand2",
                           padx=15, pady=8, font=("Malgun Gothic", 9, "bold"),
                           bd=1, relief="solid")
            tab.pack(side="left", padx=(0, 5))
            tab.bind("<Button-1>", lambda event, selected=key:
                     self.select_audio_mode(selected))
            self.source_tabs[key] = tab
        self.device_combo = ttk.Combobox(
            row, textvariable=self.device, state="readonly", style="LT.TCombobox")
        self.device_combo.pack(side="left", fill="x", expand=True, padx=(10, 7))
        self.device_combo.bind("<<ComboboxSelected>>", self.remember_selected_device)
        ttk.Button(row, text="↻ 새로고침", command=self.refresh,
                   style="LT.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(row, text="소리 찾기", command=self.scan_audio,
                   style="LT.TButton").pack(side="left")
        ttk.Separator(source, orient="horizontal").pack(fill="x", pady=(0, 10))
        row = tk.Frame(source, bg=colors["card"])
        row.pack(fill="x")
        tk.Label(row, text="⚙  번역 설정", font=("Malgun Gothic", 10, "bold"),
                 bg=colors["card"], fg=colors["text"]).pack(side="left", padx=(0, 25))
        ttk.Label(row, text="원본 언어", style="LTCardMuted.TLabel").pack(side="left", padx=(0, 7))
        ttk.Combobox(row, values=["en", "ja", "ko", "auto"], state="readonly",
                     textvariable=self.language, width=8,
                     style="LT.TCombobox").pack(side="left", padx=(0, 22))
        ttk.Label(row, text="Whisper 모델", style="LTCardMuted.TLabel").pack(side="left", padx=(0, 7))
        ttk.Combobox(row, values=["tiny", "base", "small"], state="readonly",
                     textvariable=self.model, width=9,
                     style="LT.TCombobox").pack(side="left", padx=(0, 22))
        ttk.Label(row, text="인식 간격", style="LTCardMuted.TLabel").pack(side="left", padx=(0, 7))
        ttk.Combobox(row, values=["2", "3", "4", "5", "6", "8"],
                     state="readonly", textvariable=self.seconds, width=6,
                     style="LT.TCombobox").pack(side="left")
        ttk.Label(row, text="초", style="LTCardMuted.TLabel").pack(side="left", padx=(5, 0))
        self._update_mode_ui()

        engine_card = ttk.Frame(outer, style="LTCard.TFrame", padding=(15, 8))
        engine_card.pack(fill="x", pady=(0, 9))
        row = ttk.Frame(engine_card, style="LTCard.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="번역 엔진", style="LTCard.TLabel",
                  font=("Malgun Gothic", 11, "bold")).pack(side="left", padx=(0, 20))
        ttk.Radiobutton(row, text="Hy-MT2 · 기본 추천 · 오프라인", variable=self.engine,
                        value="hymt", style="LTRadio.TRadiobutton").pack(side="left", padx=(0, 16))
        ttk.Radiobutton(row, text="Google Cloud · 온라인", variable=self.engine,
                        value="google", style="LTRadio.TRadiobutton").pack(side="left", padx=(0, 16))
        ttk.Radiobutton(row, text="Argos · 경량 오프라인", variable=self.engine,
                        value="argos", style="LTRadio.TRadiobutton").pack(side="left")
        self.engine_tools_visible = False
        self.engine_tools_toggle = ttk.Button(
            engine_card, text="설치 / 고급 설정 펼치기 ▾",
            command=self.toggle_engine_tools, style="LT.TButton")
        self.engine_tools_toggle.pack(anchor="w", pady=(6, 0))
        tools = ttk.Frame(engine_card, style="LTCard.TFrame")
        self.engine_tools_frame = tools
        self.hym_install_btn = ttk.Button(
            tools, text="Hy-MT2 설치", command=self.install_hymt, style="LT.TButton"
        )
        self.hym_install_btn.pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="설치 오류 확인", command=self.show_hymt_install_error,
                   style="LT.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="llama.cpp 실행기 선택",
                   command=self.choose_hymt_server, style="LT.TButton").pack(side="left")
        ttk.Button(tools, text="Google API 키 설정",
                   command=self.configure_google_key, style="LT.TButton").pack(side="right")
        self.engine_hint_label = ttk.Label(
            engine_card,
            text="Hy-MT2 최초 설치에는 모델 및 CPU 실행기 다운로드가 필요합니다. "
                 "Google Cloud는 API 요금 및 외부 텍스트 전송이 발생할 수 있습니다.",
            style="LTCardMuted.TLabel")

        actions = ttk.Frame(outer, style="LT.TFrame")
        actions.pack(fill="x", pady=(1, 9))
        self.start_btn = ttk.Button(actions, text="▶  번역 시작", command=self.start,
                                    style="LTPrimary.TButton")
        self.start_btn.pack(side="left", padx=(0, 7))
        self.stop_btn = ttk.Button(actions, text="■  중지", command=self.stop,
                                   state="disabled", style="LT.TButton")
        self.stop_btn.pack(side="left", padx=(0, 7))
        ttk.Button(actions, text="자막 지우기", command=self.clear_output,
                   style="LT.TButton").pack(side="left")
        ttk.Checkbutton(actions, text="항상 위에 자막", variable=self.overlay_enabled,
                        command=self.toggle_overlay,
                        style="LTCheck.TCheckbutton").pack(side="right")
        ttk.Checkbutton(actions, text="원문·번역 + AI 요약 TXT 2개 저장", variable=self.save_enabled,
                        style="LTCheck.TCheckbutton").pack(side="right", padx=(0, 16))

        status_panel = ttk.Frame(outer, style="LTCard.TFrame", padding=(15, 7))
        status_panel.pack(fill="x", pady=(0, 9))
        status_heading = ttk.Frame(status_panel, style="LTCard.TFrame")
        status_heading.pack(fill="x", pady=(0, 7))
        tk.Label(status_heading, text="●  LIVE", bg=colors["card"],
                 fg=colors["green"], font=("Malgun Gothic", 10, "bold")
                 ).pack(side="left", padx=(0, 17))
        ttk.Label(status_heading, textvariable=self.status, style="LTCard.TLabel",
                  wraplength=940).pack(side="left", fill="x", expand=True)
        meter = ttk.Frame(status_panel, style="LTCard.TFrame")
        meter.pack(fill="x")
        ttk.Label(meter, text="입력 레벨", style="LTCardMuted.TLabel").pack(side="left", padx=(0, 12))
        ttk.Progressbar(meter, variable=self.audio_meter, maximum=100,
                        length=150).pack(side="left", padx=(0, 12))
        self.audio_meter_label = ttk.Label(meter, textvariable=self.audio_state,
                                            style="LTCardMuted.TLabel")
        self.audio_meter_label.pack(side="left", fill="x", expand=True)

        footer = ttk.Frame(outer, style="LT.TFrame")
        footer.pack(side="bottom", fill="x")
        ttk.Button(footer, text="EN 모델 설치",
                   command=lambda: self.install_pack("en"),
                   style="LT.TButton").pack(side="left", padx=(0, 5))
        ttk.Button(footer, text="JA 모델 설치",
                   command=lambda: self.install_pack("ja"),
                   style="LT.TButton").pack(side="left", padx=(0, 5))
        ttk.Button(footer, text="모델 진단", command=self.diagnose_models,
                   style="LT.TButton").pack(side="left")
        ttk.Button(footer, text="프로그램 종료", command=self.close,
                   style="LTDanger.TButton").pack(side="right")
        ttk.Button(footer, text="업데이트 확인", command=self.check_update,
                   style="LT.TButton").pack(side="right", padx=(0, 7))
        ttk.Button(footer, text="기록 폴더 열기",
                   command=self.open_transcripts_folder,
                   style="LT.TButton").pack(side="right", padx=(0, 7))

        transcript = ttk.Panedwindow(outer, orient="vertical")
        transcript.pack(fill="both", expand=True, pady=(0, 7))

        def transcript_pane(label_text):
            pane = ttk.Frame(transcript, style="LTCard.TFrame", padding=(13, 10))
            ttk.Label(pane, text=label_text, style="LTCard.TLabel",
                      font=("Malgun Gothic", 11, "bold")).pack(anchor="w", pady=(0, 6))
            text_row = ttk.Frame(pane, style="LTCard.TFrame")
            text_row.pack(fill="both", expand=True)
            text_box = tk.Text(
                text_row, height=9, wrap="word", undo=False,
                font=("Malgun Gothic", 11), bg=colors["field"], fg=colors["text"],
                insertbackground=colors["text"], selectbackground="#315c9c",
                selectforeground="#ffffff", padx=11, pady=10,
                relief="flat", bd=0, highlightthickness=0,
            )
            text_box.pack(side="left", fill="both", expand=True)
            scroll = ttk.Scrollbar(text_row, orient="vertical", command=text_box.yview)
            scroll.pack(side="right", fill="y")
            text_box.configure(yscrollcommand=scroll.set)
            return pane, text_box

        orig_pane, self.original = transcript_pane("원문  /  Speech to text")
        ko_pane, self.translated = transcript_pane("한국어 번역  /  Translation")
        transcript.add(orig_pane, weight=1)
        transcript.add(ko_pane, weight=1)

        self.overlay = tk.Toplevel(self.root)
        self.overlay.title("LiveTranslator 자막")
        self.overlay.geometry("950x150+150+650")
        self.overlay.configure(bg="#121820")
        self.overlay.attributes("-topmost", True)
        self.overlay_text = tk.Label(
            self.overlay, text="한국어 자막",
            font=("Malgun Gothic", 22, "bold"),
            bg="#121820", fg="white", wraplength=900,
        )
        self.overlay_text.pack(fill="both", expand=True)
        self.overlay.protocol("WM_DELETE_WINDOW", self.hide_overlay)
        self.overlay.withdraw()

    def toggle_engine_tools(self):
        """Fold optional installation controls to reserve caption space."""
        self.engine_tools_visible = not self.engine_tools_visible
        if self.engine_tools_visible:
            self.engine_tools_frame.pack(fill="x", pady=(6, 0))
            self.engine_hint_label.pack(anchor="w", pady=(5, 0))
            self.engine_tools_toggle.configure(text="설치 / 고급 설정 접기 ▴")
        else:
            self.engine_tools_frame.pack_forget()
            self.engine_hint_label.pack_forget()
            self.engine_tools_toggle.configure(text="설치 / 고급 설정 펼치기 ▾")

    def set_live_indicator(self, state):
        if not hasattr(self, "live_label"):
            return
        if state == "running":
            self.live_label.configure(text="● 번역 중", fg="#36d2a0")
        elif state == "stopping":
            self.live_label.configure(text="● 중지 중", fg="#f9bd79")
        else:
            self.live_label.configure(text="● 대기 중", fg="#36d2a0")

    def open_transcripts_folder(self):
        """Open the saved records folder, not just its path."""
        try:
            TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(str(TRANSCRIPTS))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(TRANSCRIPTS)])
            else:
                subprocess.Popen(["xdg-open", str(TRANSCRIPTS)])
        except (OSError, ValueError) as exc:
            messagebox.showerror("기록 폴더 열기 실패", str(exc))

    def _create_transcript_session(self):
        """One dated TXT file per translation session, never a daily append log."""
        TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
        started = self.session_started_at or datetime.datetime.now()
        engine = {"argos": "Argos", "hymt": "Hy-MT2", "google": "Google"}.get(
            self.session_engine, "Unknown"
        )
        lang = self.session_language if self.session_language in ("en", "ja", "ko", "auto") else "auto"
        base_name = f"{started:%Y-%m-%d_%H-%M-%S}_{engine}_{lang}"
        header = (
            f"LiveTranslator · {started:%Y-%m-%d %H:%M:%S}\n"
            f"번역 엔진: {engine} | 원본 언어: {lang}\n"
            + "=" * 48 + "\n\n"
        )
        for index in range(1000):
            suffix = f"_{index:02d}" if index else ""
            path = TRANSCRIPTS / (base_name + suffix + ".txt")
            summary_path = TRANSCRIPTS / (base_name + suffix + "_AI요약.txt")
            if summary_path.exists():
                continue
            try:
                # x is atomic: never overwrite records from a prior session.
                with path.open("x", encoding="utf-8-sig") as handle:
                    handle.write(header)
                self.session_record_path = path
                self.session_summary_path = summary_path
                try:
                    with summary_path.open("x", encoding="utf-8-sig") as summary:
                        summary.write(
                            f"LiveTranslator AI 회의 요약 · {started:%Y-%m-%d %H:%M:%S}\n"
                            "상태: 번역 종료 후 로컬 AI 요약 생성 대기\n"
                        )
                except OSError:
                    path.unlink(missing_ok=True)
                    self.session_record_path = None
                    self.session_summary_path = None
                    raise
                return
            except FileExistsError:
                continue
        raise OSError("번역 세션 기록 파일을 생성할 수 없습니다.")

    def save_transcript(self, original, translated):
        """Keep each translated English/Japanese sentence next to its Korean."""
        if not self.save_enabled.get():
            return
        if not isinstance(original, str) or not isinstance(translated, str):
            return
        if not original.strip() and not translated.strip():
            return
        try:
            if self.session_record_path is None:
                self._create_transcript_session()
            with self.session_record_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"[{datetime.datetime.now():%H:%M:%S}]\n"
                    f"[원문]\n{original.strip()}\n"
                    f"[한국어 번역]\n{translated.strip()}\n"
                    + "-" * 48 + "\n\n"
                )
            self.session_ko_lines.append(translated.strip())
        except OSError as exc:
            self.save_enabled.set(False)
            self.status.set(f"TXT 기록 저장 실패 · 저장 옵션 해제됨: {exc}")

    @staticmethod
    def _summary_chunks(lines, max_chars=1100):
        """Chunk all spoken Korean in order without silently losing long calls."""
        chunks = []
        current = ""
        for line in lines:
            line = " ".join(str(line).split())
            if not line:
                continue
            while line:
                space = max_chars - len(current) - (1 if current else 0)
                if space <= 0:
                    chunks.append(current)
                    current = ""
                    continue
                if len(line) <= space:
                    current += (" " if current else "") + line
                    break
                cut = line.rfind(" ", 0, space + 1)
                if cut <= 0:
                    cut = space
                current += (" " if current else "") + line[:cut]
                chunks.append(current)
                current = ""
                line = line[cut:].lstrip()
        if current:
            chunks.append(current)
        return chunks

    def _create_ai_summary(self, summary_path, korean_lines, server_path):
        """Run after the last caption has been persisted, without blocking Tk."""
        from streaming_pipeline import HyMT2Backend, hym_ready
        if not hym_ready(HYM_DIR, server_path):
            raise RuntimeError(
                "로컬 Hy-MT2가 설치되지 않아 AI 요약을 생성할 수 없습니다. "
                "설치 / 고급 설정에서 Hy-MT2를 설치한 뒤 새 회의를 기록하세요."
            )
        backend = HyMT2Backend(HYM_DIR, server_path=server_path)
        try:
            backend.start(
                threading.Event(),
                lambda msg: self.events.put(("summary_status", msg)),
            )
            parts = self._summary_chunks(korean_lines)
            if not parts:
                raise RuntimeError("번역된 음성이 없어 요약할 내용이 없습니다.")
            # Summarize every passage; do not truncate the full meeting.
            partials = []
            for i, part in enumerate(parts, 1):
                self.events.put(("summary_status", f"AI 요약 작성 중 ({i}/{len(parts)})"))
                partials.append(backend.summarize_korean(part))
            if len(partials) == 1:
                result = partials[0]
            else:
                # Preserve per-section findings even if the tiny model cannot
                # reliably consolidate an hours-long meeting into one answer.
                result = "\n\n".join(
                    f"[구간 {i}]\n{item}" for i, item in enumerate(partials, 1)
                )
            with summary_path.open("w", encoding="utf-8-sig") as handle:
                handle.write(
                    "LiveTranslator · AI 회의 요약\n"
                    "요약 엔진: Hy-MT2 (로컬 / 오프라인)\n"
                    "참고: 음성 인식·번역 오류가 요약에도 반영될 수 있습니다.\n"
                    + "=" * 48 + "\n\n" + result + "\n"
                )
        finally:
            backend.stop()

    def _start_ai_summary(self):
        if self.summary_running or not self.session_summary_path:
            return
        summary_path = self.session_summary_path
        lines = list(self.session_ko_lines)
        server_path = self.settings.get("hym_server")
        self.summary_running = True
        def worker():
            try:
                self._create_ai_summary(summary_path, lines, server_path)
                self.events.put(("summary_complete", summary_path.name))
            except Exception as exc:
                # Keep the second file, but never claim a failed inference is AI output.
                try:
                    with summary_path.open("w", encoding="utf-8-sig") as file:
                        file.write(
                            "LiveTranslator · AI 회의 요약\n"
                            "상태: 생성 실패\n"
                            f"원인: {exc}\n"
                        )
                except OSError:
                    pass
                self.events.put(("summary_error", str(exc)))
        threading.Thread(target=worker, name="LiveTranslatorSummary", daemon=True).start()

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

    def _update_mode_ui(self):
        offline = self.mode.get() == "input"
        if hasattr(self, "mode_description"):
            self.mode_description.set(
                "오프라인 회의: 마이크로 발표자의 소리를 번역합니다."
                if offline else "온라인 회의: PC에서 재생되는 소리를 번역합니다."
            )
        if hasattr(self, "source_kind_label"):
            self.source_kind_label.config(
                text="마이크 입력" if offline else "PC 출력 / Loopback")
        for key, widgets in getattr(self, "mode_tiles", {}).items():
            shell, inner, line, symbol, heading, mark, desc = widgets
            selected = key == self.mode.get()
            shell.config(bg="#368dff" if selected else "#3b4b66")
            background = "#1a3a70" if selected else "#212e44"
            for widget in (inner, line, symbol, heading, mark, desc):
                widget.config(bg=background)
            mark.config(text="●" if selected else "○",
                        fg="#80c1ff" if selected else "#a0aec3")
            heading.config(fg="#ffffff" if selected else "#e1eaf7")
        for key, widget in getattr(self, "source_tabs", {}).items():
            selected = key == self.mode.get()
            widget.config(bg="#2779ed" if selected else "#212e44",
                          fg="#ffffff" if selected else "#b9c9e0",
                          highlightbackground="#2779ed" if selected else "#3b4b66",
                          highlightthickness=1)

    def select_audio_mode(self, new_mode):
        if new_mode == self.mode.get():
            return
        if self.running or self._audio_scan_busy:
            messagebox.showinfo("입력 모드", "번역과 소리 찾기를 중지한 후 모드를 바꿔주세요.")
            return
        self.mode.set(new_mode)
        self.switch_audio_mode()

    def switch_audio_mode(self):
        """Change source type, never capture microphone and loopback together."""
        if self.running or self._audio_scan_busy:
            current = self.devices[self.device_combo.current()] if (
                self.devices and self.device_combo.current() >= 0
            ) else None
            if current is not None:
                self.mode.set("output" if current[2] else "input")
            self._update_mode_ui()
            messagebox.showinfo("입력 모드", "번역과 소리 찾기를 중지한 후 모드를 바꿔주세요.")
            return
        self._update_mode_ui()
        self._show_mode_devices()
        self.persist()

    def _show_mode_devices(self):
        if not hasattr(self, "mode"):
            return
        want_loopback = self.mode.get() != "input"
        self.devices = [
            d for d in getattr(self, "all_devices", []) if bool(d[2]) == want_loopback
        ]
        self.device_combo["values"] = [d[0] for d in self.devices]
        preferred = self.settings.get("audio_devices_by_mode", {}).get(
            self.mode.get())
        if not preferred:
            saved = self.settings.get("audio_device")
            if isinstance(saved, dict) and saved.get("loopback") is want_loopback:
                preferred = saved
        chosen = find_saved_device_index(self.devices, preferred)
        label = "PC 출력" if want_loopback else "마이크 입력"
        if self.devices:
            self.device_combo.current(chosen if chosen is not None else 0)
            self.status.set(
                f"{label} 모드 · 장치 선택: {self.devices[self.device_combo.current()][3]}"
            )
        else:
            self.device_combo.set("")
            self.status.set(f"{label} 장치를 찾을 수 없습니다. Windows 오디오 설정을 확인하세요.")

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
        if hasattr(self, "mode"):
            self.all_devices = devices
            self._show_mode_devices()
            return
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
        if chosen is None:
            self.status.set(
                f"기본 오디오 장치 선택: {devices[0][3]} · "
                "번역 시작 시 바로 연결합니다. 소리가 없으면 '소리 찾기'로 확인하세요."
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
                        devices, seconds=1.0, stop_on_signal=bool(auto_start),
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

    def choose_hymt_server(self):
        if self.running or self._hym_install_busy:
            messagebox.showinfo("Hy-MT2 실행기", "번역과 설치를 중지한 뒤 실행기를 지정하세요.")
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="llama.cpp 실행기 선택 (llama-server.exe)",
            filetypes=[("llama-server.exe", "llama-server.exe"), ("Windows 실행기", "*.exe")]
        )
        if not path:
            return
        if Path(path).name.lower() != "llama-server.exe":
            messagebox.showerror("Hy-MT2 실행기", "공식 llama-server.exe를 선택해야 합니다.")
            return
        self.settings["hym_server"] = path
        self.persist()
        self.status.set("Hy-MT2 실행기 경로 저장 완료. 모델이 없다면 설치를 진행하세요.")

    def show_hymt_install_error(self):
        file = HYM_DIR / "install_error.log"
        if not file.is_file():
            messagebox.showinfo(
                "Hy-MT2 설치 진단",
                "저장된 설치 오류가 없습니다. 먼저 'Hy-MT2 설치'를 눌러주세요."
            )
            return
        try:
            contents = file.read_text(encoding="utf-8")[-3500:]
            messagebox.showerror(
                "Hy-MT2 설치 오류 상세",
                contents + "\n\n오류 기록 위치: " + str(file)
            )
        except OSError as exc:
            messagebox.showerror("Hy-MT2 설치 진단", str(exc))

    def install_hymt(self):
        if self.running:
            messagebox.showinfo("Hy-MT2 설치", "번역을 중지한 후 모델을 설치하세요.")
            return
        if self._hym_install_busy:
            return
        if not messagebox.askyesno(
            "Hy-MT2 무료 모델 설치",
            "Hy-MT2 Q4_K_M 모델(약 1.1GB)과 공식 llama.cpp Windows CPU 실행기를 "
            "사용자 PC에 다운로드합니다.\n"
            "외부 서버로 음성을 전송하지 않으며, 처음 설치할 때만 인터넷이 필요합니다.\n\n"
            "지금 다운로드할까요?",
        ):
            return
        self._hym_install_busy = True
        self._hym_install_cancel.clear()
        self.hym_install_btn.config(state="disabled")
        self.status.set("Hy-MT2 설치를 시작합니다...")

        def install_task():
            stage = ["설치 준비"]
            def on_progress(message):
                stage[0] = message
                self.events.put(("hym_progress", message))
            try:
                from streaming_pipeline import setup_hym
                setup_hym(
                    HYM_DIR,
                    progress=on_progress,
                    custom_path=self.settings.get("hym_server"),
                    cancel=self._hym_install_cancel,
                )
                (HYM_DIR / "install_error.log").unlink(missing_ok=True)
                self.events.put(("hym_installed", None))
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                try:
                    HYM_DIR.mkdir(parents=True, exist_ok=True)
                    (HYM_DIR / "install_error.log").write_text(
                        f"단계: {stage[0]}\n오류: {detail}\n"
                        "재시도 시 기존 다운로드 파일은 유지됩니다.\n",
                        encoding="utf-8"
                    )
                except OSError:
                    pass
                self.events.put(("hym_install_error", detail))

        threading.Thread(
            target=install_task, name="HyMT2Installer", daemon=True
        ).start()

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
        engine = self.engine.get()
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
                if engine == "hymt":
                    from streaming_pipeline import hym_ready
                    if hym_ready(HYM_DIR, self.settings.get("hym_server")):
                        notes.append("Hy-MT2: GGUF 모델과 llama.cpp 실행기 파일 확인됨. "
                                     "실제 실행은 번역 시작 시 검증합니다.")
                    else:
                        notes.append("Hy-MT2: 모델 또는 실행기 미설치 · "
                                     "'Hy-MT2 설치'를 눌러 설치하세요.")
                    self.events.put(("model_report", "\n".join(notes)))
                    return
                if engine == "google":
                    notes.append("Google Cloud: 온라인 번역 모드 · "
                                 "로컬 번역 모델이 필요하지 않습니다.")
                    self.events.put(("model_report", "\n".join(notes)))
                    return
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
        if getattr(self, "summary_running", False):
            messagebox.showinfo(
                "AI 회의 요약", "이전 회의의 AI 요약이 끝난 뒤 새 번역을 시작해 주세요."
            )
            return
        if not self.devices or self.device_combo.current() < 0:
            # The Windows playback device can appear just after startup.
            # Retry enumeration once rather than requiring a manual refresh.
            self.refresh()
        if not self.devices or self.device_combo.current() < 0:
            messagebox.showerror("오디오", "사용 가능한 오디오 장치가 없습니다. Windows 출력 장치를 확인하세요.")
            return
        selected = self.devices[self.device_combo.current()]
        if self._audio_scan_busy:
            messagebox.showinfo("오디오 검사", "진행 중인 소리 찾기가 끝난 뒤 번역을 시작해 주세요.")
            return
        # Never gate the first translation on a 1-second loudness probe:
        # the playback may still be silent when the user clicks Start.
        # Capture the selected default/saved output immediately and wait for
        # audio in the normal loop. '소리 찾기' remains a manual diagnostic.
        engine = self.engine.get()
        if engine not in ("argos", "google", "hymt"):
            messagebox.showerror("번역 엔진", "번역 엔진을 선택하세요.")
            return
        api_key = None
        if engine == "hymt":
            if self._hym_install_busy:
                messagebox.showinfo("Hy-MT2", "모델 설치가 끝나면 번역을 시작하세요.")
                return
            try:
                from streaming_pipeline import hym_ready
                installed = hym_ready(HYM_DIR, self.settings.get("hym_server"))
            except Exception as exc:
                messagebox.showerror("Hy-MT2", f"설치 상태 확인 실패: {exc}")
                return
            if not installed:
                messagebox.showinfo(
                    "Hy-MT2 설치 필요",
                    "Hy-MT2 모델 또는 llama.cpp 실행기가 설치되지 않았습니다. "
                    "'Hy-MT2 설치'를 눌러 설치한 후 다시 시작하세요.",
                )
                return
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
        # Starting confirms the visible source, so future launches restore it
        # even if it was the Windows default on this very first run.
        self.remember_selected_device()
        self.session_started_at = datetime.datetime.now()
        self.session_record_path = None
        self.session_summary_path = None
        self.session_ko_lines = []
        self.session_engine = engine
        self.session_language = self.language.get()
        self.stop_signal = threading.Event()
        self.running = True
        self.set_live_indicator("running")
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
        self.set_live_indicator("stopping")
        self.stop_btn.config(state="disabled")
        self.status.set("중지 요청됨. 진행 중인 AI 처리가 끝나면 종료됩니다.")

    def capture_and_translate(self, selected, language, model_name, seconds, engine, api_key):
        hy_backend = None
        try:
            from faster_whisper import WhisperModel
            if engine == "argos":
                import argostranslate.translate
            from streaming_pipeline import run_stream, HyMT2Backend

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
            if engine == "hymt":
                hy_backend = HyMT2Backend(
                    HYM_DIR, server_path=self.settings.get("hym_server")
                )
                hy_backend.start(
                    self.stop_signal,
                    lambda msg: self.events.put(("status", msg)),
                )

            def translate_cached(original, lang):
                if engine == "google":
                    return google_translate_text(original, lang, api_key)
                if engine == "hymt":
                    return hy_backend.translate(original, lang)
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
                quality_mode=(engine in ("argos", "hymt")),
                emit_status=lambda status: self.events.put(
                    ("error" if status.startswith("오디오 캡처 오류:") else "status", status)
                ),
                emit_error=lambda message: self.events.put(("error", message)),
                emit_audio_level=lambda rms, peak: self.events.put(("audio_level", (rms, peak))),
                emit_original=lambda original: self.events.put(("original", original)),
                emit_result=lambda original, translated: self.events.put(("result", (original, translated))),
                stop_event=self.stop_signal,
            )
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            if hy_backend is not None:
                try:
                    hy_backend.stop()
                except Exception:
                    pass
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
                elif kind == "hym_progress":
                    self.status.set(data)
                elif kind == "hym_installed":
                    self._hym_install_busy = False
                    self.hym_install_btn.config(state="normal")
                    self.status.set("Hy-MT2 설치 완료 · Hy-MT2 엔진을 선택하고 번역을 시작하세요.")
                    messagebox.showinfo("Hy-MT2 설치", "설치 완료. Hy-MT2를 선택하고 번역을 시작할 수 있습니다.")
                elif kind == "hym_install_error":
                    self._hym_install_busy = False
                    self.hym_install_btn.config(state="normal")
                    self.status.set(f"Hy-MT2 설치 실패 · {data[:190]}")
                    messagebox.showerror(
                        "Hy-MT2 설치 오류",
                        data + "\n\n세부 내용은 '설치 오류 확인'에서 다시 볼 수 있습니다."
                    )
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
                    self.save_transcript(original, translated)
                elif kind == "summary_status":
                    self.status.set(data)
                elif kind == "summary_complete":
                    self.summary_running = False
                    self.status.set(f"AI 요약 TXT 저장 완료 · {data}")
                elif kind == "summary_error":
                    self.summary_running = False
                    self.status.set(f"AI 요약 TXT 생성 실패 · {data[:140]}")
                elif kind == "ended":
                    self._start_ai_summary()
                    self.running = False
                    self.set_live_indicator("idle")
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
        """Exit from the dedicated button or title-bar X; stop capture first."""
        if self.summary_running and not messagebox.askyesno(
            "AI 요약 생성 중",
            "AI 요약을 작성 중입니다. 지금 종료하면 요약 TXT가 미완성일 수 있습니다. 종료할까요?",
        ):
            return
        if self._hym_install_busy and not messagebox.askyesno(
            "Hy-MT2 설치 중", "모델 다운로드를 중단하고 프로그램을 종료할까요?"
        ):
            return
        if self.running and not messagebox.askyesno(
            "LiveTranslator 종료",
            "번역이 진행 중입니다. 번역을 중지하고 프로그램을 종료할까요?",
        ):
            return
        if self.running:
            self.stop_signal.set()
        self._hym_install_cancel.set()
        try:
            self.persist()
        except (OSError, ValueError):
            pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
