"""Tests for Kaggle dataset resolution helpers."""
import zipfile

from utils.kaggle_utils import ensure_kaggle_dataset


def test_existing_local_zip_is_extracted(tmp_path):
    data_dir = tmp_path / "artifact-real-fake"
    data_dir.mkdir()
    zip_path = data_dir / "artifact-dataset.zip"

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("real/sample.txt", "real")
        archive.writestr("fake/sample.txt", "fake")

    resolved = ensure_kaggle_dataset(
        dataset_slug="owner/dataset",
        local_dir=data_dir,
        description="Test dataset",
    )

    assert resolved == data_dir
    assert (data_dir / "real" / "sample.txt").exists()
    assert (data_dir / "fake" / "sample.txt").exists()
