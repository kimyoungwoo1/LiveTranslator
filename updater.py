"""Small, opt-in source-code patch updater for the LiveTranslator EXE.

The executable is an immutable runtime; only the two allowlisted Python
modules can be replaced by verified copies downloaded from the project repo.
New binary dependencies still require a new executable build.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import Request, urlopen

BASE_VERSION = "0.6.0"
REPO_RAW = "https://raw.githubusercontent.com/kimyoungwoo1/LiveTranslator/main/"
MANIFEST_URL = REPO_RAW + "updates/stable.json"
APP_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "LiveTranslator"
PATCH_DIR = APP_DIR / "code_patch"
PATCH_FILES = ("app_core.py", "streaming_pipeline.py")


def _version_tuple(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise ValueError("유효하지 않은 버전 번호")
    return tuple(map(int, value.split(".")))


def _object_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _read_https(url, max_bytes):
    req = Request(url, headers={"User-Agent": "LiveTranslator-updater/0.6"})
    with urlopen(req, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError("다운로드 서버 오류")
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise RuntimeError("업데이트 파일 크기 제한 초과")
        return data


def _validate_manifest(info):
    if not isinstance(info, dict):
        raise ValueError("업데이트 정보 형식 오류")
    version = info.get("version")
    if _version_tuple(version) < _version_tuple(BASE_VERSION):
        raise ValueError("이전 버전 업데이트는 설치할 수 없습니다")
    if _version_tuple(info.get("min_bootstrap", BASE_VERSION)) > _version_tuple(BASE_VERSION):
        raise RuntimeError("새로운 EXE가 필요한 업데이트입니다")
    files = info.get("files")
    if not isinstance(files, dict) or set(files) != set(PATCH_FILES):
        raise ValueError("허용되지 않은 파일 목록")
    for name in PATCH_FILES:
        spec = files[name]
        if not isinstance(spec, dict):
            raise ValueError("파일 정보 오류")
        if spec.get("path") != f"updates/v{version}/{name}":
            raise ValueError("업데이트 경로가 올바르지 않습니다")
        digest = spec.get("git_sha1")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{40}", digest):
            raise ValueError("파일 무결성 정보 오류")
    return info


def installed_patch():
    try:
        info = _validate_manifest(json.loads((PATCH_DIR / "manifest.json").read_text(encoding="utf-8")))
        if _version_tuple(info["version"]) <= _version_tuple(BASE_VERSION):
            return None
        for name in PATCH_FILES:
            path = PATCH_DIR / name
            if not path.is_file() or path.is_symlink():
                return None
            data = path.read_bytes()
            if len(data) > 300_000 or _object_sha(data) != info["files"][name]["git_sha1"]:
                return None
        return info
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return None


def current_version():
    patch = installed_patch()
    return patch["version"] if patch else BASE_VERSION


def find_update():
    info = _validate_manifest(json.loads(_read_https(MANIFEST_URL, 128_000)))
    if _version_tuple(info["version"]) <= _version_tuple(current_version()):
        return None
    return info


def install_update(info):
    _validate_manifest(info)
    # Never trust a stale UI snapshot; compare it to the current server release.
    live = _validate_manifest(json.loads(_read_https(MANIFEST_URL, 128_000)))
    if info != live:
        raise RuntimeError("업데이트 정보가 변경되었습니다. 다시 확인해 주세요")
    if _version_tuple(info["version"]) <= _version_tuple(current_version()):
        raise RuntimeError("이미 최신 버전입니다")

    APP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="update-", dir=APP_DIR))
    backup = APP_DIR / "code_patch.backup"
    moved_old = False
    try:
        for name in PATCH_FILES:
            spec = info["files"][name]
            data = _read_https(REPO_RAW + spec["path"], 300_000)
            if _object_sha(data) != spec["git_sha1"]:
                raise RuntimeError(f"{name}: 파일 검증 실패")
            # Compile before swapping to reject syntax-corrupted updates.
            compile(data.decode("utf-8"), name, "exec")
            (tmp / name).write_bytes(data)
        (tmp / "manifest.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        if backup.exists():
            shutil.rmtree(backup)
        if PATCH_DIR.exists():
            os.replace(PATCH_DIR, backup)
            moved_old = True
        try:
            os.replace(tmp, PATCH_DIR)
        except Exception:
            if moved_old:
                os.replace(backup, PATCH_DIR)
            raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return info["version"]
