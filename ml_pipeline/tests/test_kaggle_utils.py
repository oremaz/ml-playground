"""Tests for Kaggle dataset resolution helpers."""
import zipfile

from utils import kaggle_utils


def test_existing_local_zip_is_extracted(tmp_path):
    data_dir = tmp_path / "artifact-real-fake"
    data_dir.mkdir()
    zip_path = data_dir / "artifact-dataset.zip"

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("real/sample.txt", "real")
        archive.writestr("fake/sample.txt", "fake")

    resolved = kaggle_utils.ensure_kaggle_dataset(
        dataset_slug="owner/dataset",
        local_dir=data_dir,
        description="Test dataset",
    )

    assert resolved == data_dir
    assert (data_dir / "real" / "sample.txt").exists()
    assert (data_dir / "fake" / "sample.txt").exists()


def test_bad_local_zip_is_ignored_and_download_attempted(tmp_path, monkeypatch):
    data_dir = tmp_path / "artifact-real-fake"
    data_dir.mkdir()
    bad_zip_path = data_dir / "artifact-dataset.zip"
    bad_zip_path.write_text("<html>not a zip</html>")
    downloads = []

    def fake_download(dataset_slug, destination):
        downloads.append((dataset_slug, destination))
        (destination / "real").mkdir()
        (destination / "real" / "sample.txt").write_text("real")

    monkeypatch.setattr(kaggle_utils, "_download_with_kaggle_cli", fake_download)

    resolved = kaggle_utils.ensure_kaggle_dataset(
        dataset_slug="owner/dataset",
        local_dir=data_dir,
        description="Test dataset",
    )

    assert resolved == data_dir
    assert downloads == [("owner/dataset", data_dir)]
    assert (data_dir / "artifact-dataset.zip.invalid").exists()
    assert (data_dir / "real" / "sample.txt").exists()
