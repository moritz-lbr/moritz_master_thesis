"""Dataset loaders used for training and for the post-training analysis.

The thesis experiments use CIFAR-10 and SVHN (download them with
``datasets/download.py``). Both loaders return ``(x_train, y_train, x_test, y_test)`` as
float32 arrays with flattened inputs that are standardized channel-wise with statistics of
the training split, using the predefined train/test splits of the datasets.

``load_dataset`` inspects the files inside a dataset directory and dispatches to the
matching loader, so the experiment configs only need to point ``task`` at a folder below
``datasets/``. To train on further benchmark datasets (e.g. from the UCI Machine Learning
Repository, https://archive.ics.uci.edu), add a loader with the same return signature, a
detection function and a branch in ``load_dataset``, and place a ``dataset_overview.yaml``
with the input and output dimension in the dataset folder (see ``datasets/download.py``).
"""
from pathlib import Path
from typing import List, Tuple
import pickle

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"


def resolve_dataset_path(task_path: str | Path) -> Path:
    """Resolve a dataset path from a config or a saved simulation_config.yaml.

    Relative paths are tried relative to the working directory, the repository root and
    ``datasets/``. If an absolute path recorded in an old log no longer exists (e.g. the
    logs were copied from a cluster), ``datasets/<folder name>`` is used as a fallback.
    """
    candidate = Path(task_path).expanduser()
    candidates: List[Path] = []

    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend([candidate, REPO_ROOT / candidate, DATASETS_DIR / candidate])
    candidates.append(DATASETS_DIR / candidate.name)

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        f"Could not resolve dataset path from {task_path!r}. "
        "Download the data with `pixi run python datasets/download.py`."
    )


def getImageData(filename):
    with open(filename, 'rb') as fo:
        batch = pickle.load(fo, encoding='latin1')
        X = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        y = np.array(batch['labels'])
    return X, y


def load_all_cifar10_data(dataset_path: str | Path = "cifar-10-batches-py"):
    """Loads all CIFAR-10 batches and concatenates them into train/test datasets."""
    base_dir = resolve_dataset_path(dataset_path)
    train_paths = [base_dir / f"data_batch_{batch_idx}" for batch_idx in range(1, 6)]
    test_path = base_dir / "test_batch"

    x_train_list, y_train_list = [], []

    # Load all training batches
    for path in train_paths:
        X, y = getImageData(path)
        x_train_list.append(X)
        y_train_list.append(y)

    # Concatenate all training data
    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # Load test data
    x_test, y_test = getImageData(test_path)

    # Convert to float and scale to [0,1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32) / 255.0

    # Channel-wise z-score using TRAIN stats only
    # shapes: mean/std -> (1,1,1,3)
    mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    std = x_train.std(axis=(0, 1, 2), keepdims=True)
    std = np.maximum(std, 1e-7)  # avoid divide-by-zero
    x_train = (x_train - mean) / std
    x_test  = (x_test  - mean) / std

    x_train = x_train.reshape(x_train.shape[0], 3072)
    x_test = x_test.reshape(x_test.shape[0], 3072)

    return x_train, y_train, x_test, y_test


def _read_svhn_mat(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.io import loadmat

    data = loadmat(path)
    if "X" not in data or "y" not in data:
        raise ValueError(f"Expected SVHN .mat file with 'X' and 'y' arrays at {path}.")

    images = np.asarray(data["X"])
    labels = np.asarray(data["y"]).reshape(-1).astype(np.int32)

    if images.ndim != 4 or images.shape[:3] != (32, 32, 3):
        raise ValueError(
            f"Expected SVHN images with shape (32, 32, 3, n) in {path}, got {images.shape}."
        )

    images = images.transpose(3, 0, 1, 2)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"SVHN image/label count mismatch in {path}: {images.shape[0]} images, "
            f"{labels.shape[0]} labels."
        )

    labels[labels == 10] = 0
    if np.any((labels < 0) | (labels > 9)):
        raise ValueError(f"Expected SVHN labels in 0..9 after remapping 10 to 0 in {path}.")

    return images, labels


def load_svhn(
    dataset_path: str | Path = "svhn",
    *,
    one_hot: bool = True,
    include_extra: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load SVHN cropped 32x32 train/test splits with channel-wise normalization."""
    base_dir = resolve_dataset_path(dataset_path)
    train_path = base_dir / "train_32x32.mat"
    test_path = base_dir / "test_32x32.mat"

    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Expected SVHN train_32x32.mat and test_32x32.mat under {base_dir}."
        )

    x_train, y_train = _read_svhn_mat(train_path)
    x_test, y_test = _read_svhn_mat(test_path)

    if include_extra:
        extra_path = base_dir / "extra_32x32.mat"
        if not extra_path.is_file():
            raise FileNotFoundError(f"Requested SVHN extra split, but {extra_path} was not found.")
        x_extra, y_extra = _read_svhn_mat(extra_path)
        x_train = np.concatenate([x_train, x_extra], axis=0)
        y_train = np.concatenate([y_train, y_extra], axis=0)

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    std = x_train.std(axis=(0, 1, 2), keepdims=True)
    std = np.maximum(std, 1e-7)
    x_train = ((x_train - mean) / std).reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = ((x_test - mean) / std).reshape(x_test.shape[0], -1).astype(np.float32)

    if one_hot:
        eye = np.eye(10, dtype=np.float32)
        y_train = eye[y_train.astype(np.int32)]
        y_test = eye[y_test.astype(np.int32)]

    return x_train, y_train, x_test, y_test


# ----------------------------
# Dataset detection
# ----------------------------

def _is_cifar10_dataset(dataset_path: Path) -> bool:
    return (
        dataset_path.is_dir()
        and (dataset_path / "data_batch_1").is_file()
        and (dataset_path / "test_batch").is_file()
    )

def _is_svhn_dataset(dataset_path: Path) -> bool:
    return (
        dataset_path.is_dir()
        and (dataset_path / "train_32x32.mat").is_file()
        and (dataset_path / "test_32x32.mat").is_file()
    )


def load_dataset(task: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load the dataset referenced by ``task`` with its predefined train/test split.

    Returns ``(x_train, y_train, x_test, y_test, test_split)``, where ``test_split`` is the
    fraction of test samples.
    """
    dataset_path = resolve_dataset_path(task)

    if _is_svhn_dataset(dataset_path):
        data = load_svhn(dataset_path)
    elif _is_cifar10_dataset(dataset_path):
        data = load_all_cifar10_data(dataset_path)
    else:
        raise ValueError(
            f"Unsupported dataset in {dataset_path}. Supported are CIFAR-10 "
            "(data_batch_1..5, test_batch) and SVHN (train_32x32.mat, test_32x32.mat); "
            "see the module docstring of src/data.py on how to add further datasets."
        )

    x_train, y_train, x_test, y_test = (np.asarray(a, dtype=np.float32) for a in data)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    test_split = float(x_test.shape[0] / (x_train.shape[0] + x_test.shape[0]))
    return x_train, y_train, x_test, y_test, test_split
