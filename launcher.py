"""Stable LiveTranslator executable bootstrap; source updates are loaded as modules."""
import importlib.util
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("ARGOS_CHUNK_TYPE", "MINISBD")
os.environ.setdefault("ARGOS_DEVICE_TYPE", "cpu")
os.environ.setdefault("ARGOS_BEAM_SIZE", "1")

# Static imports ensure all dependencies are bundled by PyInstaller.
import app_core as bundled_core
import streaming_pipeline as bundled_pipeline
import updater


def _load_python(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("패치 모듈을 불러올 수 없습니다")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def choose_app():
    info = updater.installed_patch()
    if not info:
        return bundled_core, None
    try:
        _load_python("streaming_pipeline", updater.PATCH_DIR / "streaming_pipeline.py")
        patched = _load_python("live_patch_app_core", updater.PATCH_DIR / "app_core.py")
        return patched, None
    except Exception as exc:
        sys.modules["streaming_pipeline"] = bundled_pipeline
        return bundled_core, f"업데이트된 코드 실행 실패. 기본 버전으로 실행합니다: {exc}"


def main():
    if "--self-test" in sys.argv:
        try:
            import argostranslate.package
            import argostranslate.translate
            import faster_whisper
            from minisbd import SBDetect
            assert bundled_core.App
            assert bundled_pipeline.run_stream
            assert updater.BASE_VERSION
        except Exception:
            import traceback
            traceback.print_exc()
            return 1
        print("Frozen runtime imports OK")
        return 0

    import tkinter as tk
    from tkinter import messagebox
    core, warning = choose_app()
    root = tk.Tk()
    core.App(root)
    if warning:
        root.after(200, lambda: messagebox.showwarning("업데이트 실행 오류", warning))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
