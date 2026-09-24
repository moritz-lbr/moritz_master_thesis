"""Download the benchmark datasets used in the thesis into ``datasets/``.

Usage (from the repository root):

    pixi run python datasets/download.py cifar10 svhn

Each dataset is stored in its own folder together with a ``dataset_overview.yaml``
that the training code reads to infer the input/output dimension of the network:

    datasets/cifar-10-batches-py/   CIFAR-10 (python version, 50k train / 10k test)
    datasets/svhn/                  SVHN cropped digits (73,257 train / 26,032 test)

The experiment configs refer to these folders via ``task: datasets/<folder>``.
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import yaml

DATASETS_DIR = Path(__file__).resolve().parent

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
SVHN_URL = "http://ufldl.stanford.edu/housenumbers/{split}_32x32.mat"


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as file:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            file.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(target)


def _write_overview(folder: Path, overview: dict) -> None:
    with (folder / "dataset_overview.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump({"training": overview}, file, sort_keys=False)


def download_cifar10(force: bool = False) -> Path:
    folder = DATASETS_DIR / "cifar-10-batches-py"
    expected = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    if force or not all((folder / name).is_file() for name in expected):
        with tempfile.TemporaryDirectory(dir=DATASETS_DIR) as tmp:
            archive = Path(tmp) / "cifar-10-python.tar.gz"
            _download(CIFAR10_URL, archive)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(tmp)
            if folder.exists():
                shutil.rmtree(folder)
            shutil.move(str(Path(tmp) / "cifar-10-batches-py"), folder)
    else:
        print(f"CIFAR-10 already present in {folder}")

    _write_overview(
        folder,
        {
            "data_samples": 60000,
            "input_dimension": 3072,
            "output_dimension": 10,
            "source_url": CIFAR10_URL,
            "preprocessing": {
                "float_conversion": "uint8_to_float32_div255",
                "feature_standardization": "train_split_channelwise_zscore",
                "label_encoding": "class_index",
            },
            "split": {"type": "predefined", "n_train": 50000, "n_test": 10000},
        },
    )
    return folder


def download_svhn(force: bool = False) -> Path:
    folder = DATASETS_DIR / "svhn"
    for split in ("train", "test"):
        target = folder / f"{split}_32x32.mat"
        if force or not target.is_file():
            _download(SVHN_URL.format(split=split), target)
        else:
            print(f"SVHN {split} split already present in {target}")

    _write_overview(
        folder,
        {
            "data_samples": 99289,
            "input_dimension": 3072,
            "output_dimension": 10,
            "variant": "cropped_32x32",
            "source_url": "http://ufldl.stanford.edu/housenumbers/",
            "preprocessing": {
                "float_conversion": "uint8_to_float32_div255",
                "feature_standardization": "train_split_channelwise_zscore",
                "label_remapping": "svhn_10_to_digit_0",
                "label_encoding": "one_hot",
            },
            "split": {"type": "predefined", "n_train": 73257, "n_test": 26032},
            "files": {"train": "train_32x32.mat", "test": "test_32x32.mat"},
        },
    )
    return folder


DOWNLOADERS = {"cifar10": download_cifar10, "svhn": download_svhn}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", choices=sorted(DOWNLOADERS), default=sorted(DOWNLOADERS))
    parser.add_argument("--force", action="store_true", help="Re-download even if the files exist.")
    args = parser.parse_args()

    for name in args.datasets:
        folder = DOWNLOADERS[name](force=args.force)
        print(f"{name}: ready in {folder}")


if __name__ == "__main__":
    main()
