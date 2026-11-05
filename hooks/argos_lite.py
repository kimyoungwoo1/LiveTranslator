"""Experimental PyInstaller runtime hook for MiniSBD-only Argos translation.
Only used by the separate lite build, not the normal build.
Argos 1.11 sbd.py imports stanza even when MiniSBD is selected.
The placeholder avoids bundling Stanza and its large torch dependency.
"""
import os
import sys
import types
os.environ["ARGOS_CHUNK_TYPE"] = "MINISBD"
os.environ["ARGOS_DEVICE_TYPE"] = "cpu"
if "stanza" not in sys.modules:
    stanza_stub = types.ModuleType("stanza")
    stanza_stub.__path__ = []
    sys.modules["stanza"] = stanza_stub
