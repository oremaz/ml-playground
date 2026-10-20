"""Recreate the uploaded CSVs from the exact public snapshots used in this pilot."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd


SOURCES = {
    "energy.csv": ("https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/e3e4c27a4ae2b41b21f84b4c9ac2d822d3d0ace1/energydata_complete.csv",
                   "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"),
    "bikes.zip": ("https://www.kaggle.com/api/v1/datasets/download/lakshmi25npathi/bike-sharing-dataset",
                  "436d50be2070bb22b651a607697147dffdf733117601a21bfbdccd129ef1f2a1"),
    "activity.zip": ("https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip",
                     "50dabbc800629611831a85b8b71c87040525ca4af6a152c6ba9360ecee6b92dc"),
}


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in SOURCES.items():
        path = root / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Public snapshot changed: {name}; inspect before accepting a new version")
            path.write_bytes(content)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    with zipfile.ZipFile(root / "bikes.zip") as archive:
        (root / "bikes.csv").write_bytes(archive.read("hour.csv"))
    with zipfile.ZipFile(root / "activity.zip") as archive:
        names = pd.read_csv(io.BytesIO(archive.read("UCI HAR Dataset/features.txt")), sep=r"\s+", header=None)[1]
        labels = dict(pd.read_csv(io.BytesIO(archive.read("UCI HAR Dataset/activity_labels.txt")), sep=r"\s+", header=None).itertuples(index=False, name=None))
        columns = [f"{name}__{index + 1}" for index, name in enumerate(names)]
        frames = []
        for split in ["train", "test"]:
            values = pd.read_csv(io.BytesIO(archive.read(f"UCI HAR Dataset/{split}/X_{split}.txt")), sep=r"\s+", header=None, names=columns)
            activity = pd.read_csv(io.BytesIO(archive.read(f"UCI HAR Dataset/{split}/y_{split}.txt")), header=None)[0].map(labels)
            subject = pd.read_csv(io.BytesIO(archive.read(f"UCI HAR Dataset/{split}/subject_{split}.txt")), header=None)[0]
            frames.append(pd.concat([pd.DataFrame({"activity": activity, "subject": subject}), values], axis=1))
        pd.concat(frames, ignore_index=True).to_csv(root / "activity.csv", index=False)
    manifest = {}
    for name, number in [("energy", 374), ("bikes", 275), ("activity", 240)]:
        path = root / f"{name}.csv"
        frame = pd.read_csv(path)
        manifest[name] = {"uci_id": number, "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "shape": list(frame.shape)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("ml_pipeline/data/agent_domains"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))
