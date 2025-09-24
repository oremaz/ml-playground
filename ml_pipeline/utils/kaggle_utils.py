from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

_KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def _running_on_kaggle() -> bool:
    """Return True when executing inside a Kaggle Kernel environment."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None and _KAGGLE_INPUT_ROOT.exists()


def _has_data(directory: Path) -> bool:
    try:
        return directory.exists() and any(directory.iterdir())
    except (OSError, StopIteration):
        return directory.exists()


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

    if _has_data(local_path):
        print(f"ℹ️ {description} already present at {local_path}")
        return local_path

    try:
        _download_with_kaggle_cli(dataset_slug, local_path)
        if _has_data(local_path):
            print(f"✅ Downloaded {description} to {local_path}")
    except Exception as exc:  # pragma: no cover - requires network/CLI
        print(f"⚠️ Kaggle download for {description} skipped: {exc}")

    return local_path
