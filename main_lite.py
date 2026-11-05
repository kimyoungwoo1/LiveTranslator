"""Experimental lightweight Windows packaging entry point."""
import os
import sys
os.environ["ARGOS_CHUNK_TYPE"] = "MINISBD"
os.environ["ARGOS_DEVICE_TYPE"] = "cpu"

if "--self-test" in sys.argv:
    try:
        from faster_whisper import WhisperModel
        import argostranslate.package
        import argostranslate.translate
        from minisbd import SBDetect
        langs = argostranslate.translate.get_installed_languages()
        print("Frozen import smoke test passed", len(langs))
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)

import tkinter as tk
from main import App
root = tk.Tk()
App(root)
root.mainloop()
