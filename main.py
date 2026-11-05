"""LiveTranslator: Windows local audio-to-Korean subtitles (prototype)."""
import datetime
import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

import numpy as np
from scipy.signal import resample_poly
import pyaudiowpatch as pyaudio

APP_DIR = Path.home() / "AppData" / "Local" / "LiveTranslator"
APP_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS = APP_DIR / "settings.json"
TRANSCRIPTS = APP_DIR / "transcripts"
TRANSCRIPTS.mkdir(exist_ok=True)

class App:
    def __init__(self, root):
        self.root = root
        root.title("LiveTranslator - 오프라인 한국어 자막")
        root.geometry("880x710")
        self.events = queue.Queue()
        self.stop_signal = threading.Event()
        self.worker = None
        self.running = False
        self.devices = []
        self.settings = self.load_settings()
        self.device = tk.StringVar()
        self.language = tk.StringVar(value=self.settings.get("language", "en"))
        self.model = tk.StringVar(value=self.settings.get("model", "base"))
        self.seconds = tk.StringVar(value=str(self.settings.get("seconds", 4)))
        self.save_enabled = tk.BooleanVar(value=self.settings.get("save", False))
        self.overlay_enabled = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="장치를 선택하고 번역을 시작하세요.")
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
            "seconds": self.seconds.get(), "save": self.save_enabled.get()
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def build_ui(self):
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="LiveTranslator", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(frame, text="PC 출력 장치(WASAPI Loopback) 또는 마이크를 선택해 한국어로 번역합니다.").pack(anchor="w", pady=(5, 16))
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=5)
        ttk.Label(row, text="오디오 장치", width=13).pack(side="left")
        self.device_combo = ttk.Combobox(row, textvariable=self.device, state="readonly")
        self.device_combo.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row, text="새로고침", command=self.refresh).pack(side="left")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text="원본 언어", width=13).pack(side="left")
        ttk.Combobox(row, values=["en", "ja", "ko", "auto"], state="readonly", textvariable=self.language, width=10).pack(side="left")
        ttk.Label(row, text="   음성 인식 모델").pack(side="left")
        ttk.Combobox(row, values=["tiny", "base", "small"], state="readonly", textvariable=self.model, width=9).pack(side="left")
        ttk.Label(row, text="   인식 간격(초)").pack(side="left")
        ttk.Combobox(row, values=["2", "3", "4", "5", "6", "8"], state="readonly", textvariable=self.seconds, width=6).pack(side="left")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(8, 14))
        ttk.Checkbutton(row, text="번역 내역 TXT 저장", variable=self.save_enabled).pack(side="left")
        ttk.Checkbutton(row, text="항상 위에 자막", variable=self.overlay_enabled, command=self.toggle_overlay).pack(side="left", padx=12)
        ttk.Button(row, text="기록 폴더", command=lambda: messagebox.showinfo("저장 위치", str(TRANSCRIPTS))).pack(side="right")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=4)
        self.start_btn = ttk.Button(row, text="▶ 번역 시작", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(row, text="■ 중지", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=7)
        ttk.Button(row, text="영어→한국어 모델 설치", command=lambda: self.install_pack("en")).pack(side="left", padx=3)
        ttk.Button(row, text="일본어→한국어 모델 설치", command=lambda: self.install_pack("ja")).pack(side="left", padx=3)
        ttk.Label(frame, textvariable=self.status, foreground="#215ba5").pack(anchor="w", pady=(10, 12))
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

    def hide_overlay(self):
        self.overlay_enabled.set(False)
        self.overlay.withdraw()

    def toggle_overlay(self):
        if self.overlay_enabled.get():
            self.overlay.deiconify()
        else:
            self.overlay.withdraw()

    def refresh(self):
        devices = []
        p = pyaudio.PyAudio()
        try:
            for info in p.get_loopback_device_info_generator():
                devices.append((f"🔊 출력 | {info['name']} [Loopback]", int(info["index"]), True))
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0 and not info.get("isLoopbackDevice", False):
                    devices.append((f"🎤 입력 | {info['name']}", int(info["index"]), False))
        finally:
            p.terminate()
        self.devices = devices
        self.device_combo["values"] = [d[0] for d in devices]
        if devices:
            self.device_combo.current(0)
        else:
            self.status.set("오디오 장치를 찾을 수 없습니다.")

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
        self.persist()
        self.stop_signal = threading.Event()
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.set("음성 인식 모델을 초기화하는 중...")
        self.worker = threading.Thread(target=self.capture_and_translate, args=(selected,), daemon=True)
        self.worker.start()

    def stop(self):
        if not self.running:
            return
        self.stop_signal.set()
        self.stop_btn.config(state="disabled")
        self.status.set("중지 요청됨. 진행 중인 AI 처리가 끝나면 종료됩니다.")

    def capture_and_translate(self, selected):
        stream = None
        pa = None
        try:
            from faster_whisper import WhisperModel
            import argostranslate.translate
            self.events.put(("status", "음성 인식 모델을 로드 중입니다. 첫 실행에는 다운로드가 필요합니다."))
            model = WhisperModel(self.model.get(), device="cpu", compute_type="int8", cpu_threads=6)
            if self.stop_signal.is_set():
                return
            pa = pyaudio.PyAudio()
            info = pa.get_device_info_by_index(selected[1])
            rate = int(info["defaultSampleRate"])
            channels = min(int(info["maxInputChannels"]), 2)
            if channels < 1:
                raise RuntimeError("선택한 장치에 입력 채널이 없습니다.")
            frames = max(512, int(rate * .1))
            stream = pa.open(format=pyaudio.paFloat32, channels=channels, rate=rate, input=True,
                             input_device_index=selected[1], frames_per_buffer=frames)
            self.events.put(("status", "실시간 음성 수집 및 번역 중"))
            chunk = []
            length = 0
            while not self.stop_signal.is_set():
                samples = np.frombuffer(stream.read(frames, exception_on_overflow=False), dtype=np.float32)
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)
                chunk.append(samples)
                length += samples.size
                if length < rate * int(self.seconds.get()):
                    continue
                samples = np.concatenate(chunk)
                chunk, length = [], 0
                if self.stop_signal.is_set() or np.sqrt(np.mean(samples * samples)) < .002:
                    continue
                samples = resample_poly(samples, 16000, rate).astype(np.float32)
                self.events.put(("status", "음성 인식 중..."))
                segments, metadata = model.transcribe(samples, language=None if self.language.get() == "auto" else self.language.get(),
                                                       vad_filter=True, beam_size=1, condition_on_previous_text=False)
                original = " ".join(segment.text.strip() for segment in segments).strip()
                if self.stop_signal.is_set():
                    break
                if not original:
                    self.events.put(("status", "음성 수집 중"))
                    continue
                source_lang = metadata.language if self.language.get() == "auto" else self.language.get()
                if source_lang == "ko":
                    translated = original
                else:
                    translated = argostranslate.translate.translate(original, source_lang, "ko")
                if not self.stop_signal.is_set():
                    self.events.put(("result", (original, translated)))
                    self.events.put(("status", "음성 수집 및 번역 중"))
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if pa is not None:
                pa.terminate()
            self.events.put(("ended", None))

    def consume(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "status":
                    self.status.set(data)
                elif kind == "error":
                    self.status.set(f"오류: {data}")
                elif kind == "result":
                    original, translated = data
                    self.original.insert("end", original + "\n")
                    self.original.see("end")
                    self.translated.insert("end", translated + "\n")
                    self.translated.see("end")
                    self.overlay_text.config(text=translated)
                    if self.save_enabled.get():
                        filename = TRANSCRIPTS / (datetime.date.today().isoformat() + ".txt")
                        with filename.open("a", encoding="utf-8") as f:
                            f.write(f"[{datetime.datetime.now():%H:%M:%S}] {original}\n{translated}\n\n")
                elif kind == "ended":
                    self.running = False
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    if self.stop_signal.is_set():
                        self.status.set("중지 완료")
        except queue.Empty:
            pass
        self.root.after(100, self.consume)

    def close(self):
        if self.running:
            self.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
