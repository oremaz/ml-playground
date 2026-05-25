from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

_KAGGLE_INPUT_ROOT = Path("/kaggle/input")
_ARCHIVE_SUFFIXES = {".zip"}
_IGNORED_DATA_SUFFIXES = _ARCHIVE_SUFFIXES | {".invalid"}


def _running_on_kaggle() -> bool:
    """Return True when executing inside a Kaggle Kernel environment."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None and _KAGGLE_INPUT_ROOT.exists()


def _has_data(directory: Path) -> bool:
    try:
        return directory.exists() and any(directory.iterdir())
    except (OSError, StopIteration):
        return directory.exists()


def _has_non_archive_data(directory: Path) -> bool:
    try:
        return any(
            entry.is_dir() or entry.suffix.lower() not in _IGNORED_DATA_SUFFIXES
            for entry in directory.iterdir()
        )
    except (OSError, StopIteration):
        return False


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Refusing to extract unsafe path {member.filename!r} from {zip_path}")
        archive.extractall(destination)


def _quarantine_bad_archive(archive: Path) -> Path:
    quarantined = archive.with_suffix(f"{archive.suffix}.invalid")
    counter = 1
    while quarantined.exists():
        quarantined = archive.with_name(f"{archive.name}.{counter}.invalid")
        counter += 1
    archive.rename(quarantined)
    return quarantined


def _extract_archives_if_needed(directory: Path) -> None:
    if not directory.is_dir() or _has_non_archive_data(directory):
        return

    archives = sorted(
        entry for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in _ARCHIVE_SUFFIXES
    )
    for archive in archives:
        try:
            _safe_extract_zip(archive, directory)
        except zipfile.BadZipFile:
            quarantined = _quarantine_bad_archive(archive)
            print(f"⚠️ Skipped invalid zip archive {archive.name}; moved it to {quarantined.name}")
            continue
        print(f"✅ Extracted {archive.name} in {directory}")


def _download_with_kaggle_cli(dataset_slug: str, destination: Path) -> None:
    kaggle_cli = shutil.which("kaggle")
    if not kaggle_cli:
        raise RuntimeError("kaggle CLI not found. Install the 'kaggle' package or ensure it is on PATH.")

    destination.mkdir(parents=True, exist_ok=True)
    cmd = [
        kaggle_cli,
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(destination),
        "--unzip",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)


def ensure_kaggle_dataset(
    dataset_slug: str,
    local_dir: Path | str,
    description: str,
    kaggle_subdir: Optional[str] = None,
) -> Path:
    """
    Ensure a Kaggle dataset is available locally or via Kaggle Kernel inputs.

    Returns the path that should be used by downstream code.
    """
    local_path = Path(local_dir)
    kaggle_dir_name = kaggle_subdir or dataset_slug.split("/")[-1]

    if _running_on_kaggle():
        kaggle_input = _KAGGLE_INPUT_ROOT / kaggle_dir_name
        if _has_data(kaggle_input):
            print(f"✅ Using Kaggle input for {description} at {kaggle_input}")
            return kaggle_input

    _extract_archives_if_needed(local_path)
    if _has_non_archive_data(local_path):
        print(f"ℹ️ {description} already present at {local_path}")
        return local_path

    try:
        _download_with_kaggle_cli(dataset_slug, local_path)
        _extract_archives_if_needed(local_path)
        if _has_non_archive_data(local_path):
            print(f"✅ Downloaded {description} to {local_path}")
    except Exception as exc:  # pragma: no cover - requires network/CLI
        print(f"⚠️ Kaggle download for {description} skipped: {exc}")

    return local_path
