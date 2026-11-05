"""Offline updater tests. Run: python -m unittest discover -s tests -v"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import updater


ROOT = Path(__file__).resolve().parent.parent


def published_bytes(path):
    """Compare GitHub Raw LF bytes, not checkout-dependent Windows CRLF bytes."""
    return path.read_bytes().replace(b"\r\n", b"\n")


class UpdateTests(unittest.TestCase):
    def test_manifest_matches_repository_files(self):
        manifest = json.loads((ROOT / "updates" / "stable.json").read_text(encoding="utf-8"))
        updater._validate_manifest(manifest)
        for name, info in manifest["files"].items():
            file_bytes = published_bytes(ROOT / info["path"])
            self.assertEqual(updater._object_sha(file_bytes), info["git_sha1"])
            compile(file_bytes.decode("utf-8"), name, "exec")

    def test_install_then_verify_local_patch(self):
        manifest = json.loads((ROOT / "updates" / "stable.json").read_text(encoding="utf-8"))
        files = {
            updater.MANIFEST_URL: json.dumps(manifest).encode("utf-8"),
        }
        files.update({
            updater.REPO_RAW + item["path"]: published_bytes(ROOT / item["path"])
            for item in manifest["files"].values()
        })
        with tempfile.TemporaryDirectory() as td, patch.object(updater, "APP_DIR", Path(td)), patch.object(
            updater, "PATCH_DIR", Path(td) / "code_patch"
        ), patch.object(updater, "_read_https", side_effect=lambda url, max_bytes: files[url]):
            self.assertEqual(updater.current_version(), updater.BASE_VERSION)
            self.assertIsNotNone(updater.find_update())
            updater.install_update(manifest)
            self.assertEqual(updater.current_version(), manifest["version"])
            with (updater.PATCH_DIR / "app_core.py").open("ab") as f:
                f.write(b"tamper")
            self.assertEqual(updater.current_version(), updater.BASE_VERSION)

    def test_bad_manifest_rejected(self):
        manifest = json.loads((ROOT / "updates" / "stable.json").read_text(encoding="utf-8"))
        manifest["files"]["app_core.py"]["path"] = "../main.py"
        with self.assertRaises(ValueError):
            updater._validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
