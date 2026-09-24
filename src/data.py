"""Dataset loaders used for training and for the post-training analysis.

Every loader returns ``(x_train, y_train, x_test, y_test)`` as float32 arrays with
flattened, standardized inputs (statistics computed on the training split only).
Classification targets are one-hot encoded.

``load_dataset`` inspects the files inside a dataset directory and dispatches to the
matching loader, so the experiment configs only need to point ``task`` at a folder
below ``datasets/`` (see ``datasets/download.py``).
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


def _read_idx_images(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    magic, num_images, rows, cols = np.frombuffer(raw[:16], dtype=">i4")
    if magic != 2051:
        raise ValueError(f"Expected IDX image file at {path}, found magic number {magic}.")
    images = np.frombuffer(raw[16:], dtype=np.uint8)
    return images.reshape(num_images, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    magic, num_labels = np.frombuffer(raw[:8], dtype=">i4")
    if magic != 2049:
        raise ValueError(f"Expected IDX label file at {path}, found magic number {magic}.")
    labels = np.frombuffer(raw[8:], dtype=np.uint8)
    if labels.shape[0] != num_labels:
        raise ValueError(f"Expected {num_labels} labels in {path}, found {labels.shape[0]}.")
    return labels


def load_mnist(dataset_path: str | Path = "mnist") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the predefined MNIST train/test splits with per-pixel normalization."""
    base_dir = resolve_dataset_path(dataset_path)
    train_images_path = base_dir / "train-images-idx3-ubyte"
    train_labels_path = base_dir / "train-labels-idx1-ubyte"
    test_images_path = base_dir / "t10k-images-idx3-ubyte"
    test_labels_path = base_dir / "t10k-labels-idx1-ubyte"

    x_train = _read_idx_images(train_images_path).astype(np.float32) / 255.0
    y_train = _read_idx_labels(train_labels_path)
    x_test = _read_idx_images(test_images_path).astype(np.float32) / 255.0
    y_test = _read_idx_labels(test_labels_path)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-7)
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)

    eye = np.eye(10, dtype=np.float32)
    y_train = eye[y_train.astype(np.int32)]
    y_test = eye[y_test.astype(np.int32)]

    return x_train, y_train, x_test, y_test


def _resolve_emnist_byclass_dir(dataset_path: str | Path) -> Path:
    base_dir = resolve_dataset_path(dataset_path)
    byclass_dir = base_dir / "byclass"
    return byclass_dir if byclass_dir.is_dir() else base_dir


def load_emnist_byclass(dataset_path: str | Path = "emnist") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the predefined EMNIST byclass train/test splits with per-pixel normalization."""
    base_dir = _resolve_emnist_byclass_dir(dataset_path)
    train_images_path = base_dir / "emnist-byclass-train-images-idx3-ubyte"
    train_labels_path = base_dir / "emnist-byclass-train-labels-idx1-ubyte"
    test_images_path = base_dir / "emnist-byclass-test-images-idx3-ubyte"
    test_labels_path = base_dir / "emnist-byclass-test-labels-idx1-ubyte"

    x_train = _read_idx_images(train_images_path).reshape(-1, 28 * 28).astype(np.float32)
    x_train /= 255.0
    y_train = _read_idx_labels(train_labels_path)
    x_test = _read_idx_images(test_images_path).reshape(-1, 28 * 28).astype(np.float32)
    x_test /= 255.0
    y_test = _read_idx_labels(test_labels_path)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-7)
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std

    num_classes = int(max(y_train.max(), y_test.max())) + 1
    eye = np.eye(num_classes, dtype=np.float32)
    y_train = eye[y_train.astype(np.int32)]
    y_test = eye[y_test.astype(np.int32)]

    return x_train, y_train, x_test, y_test


def load_winequality(
    path: str | Path = "winequality",
    *,
    test_split: float = 0.2,
    seed: int = 0,
    one_hot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the wine quality classification dataset with train-split standardization."""
    base_path = resolve_dataset_path(path)

    if base_path.is_file():
        csv_paths = [base_path]
    else:
        csv_paths = [
            csv_path
            for csv_path in (
                base_path / "winequality-red.csv",
                base_path / "winequality-white.csv",
            )
            if csv_path.is_file()
        ]
        if not csv_paths:
            raise FileNotFoundError(
                f"Could not find winequality CSV files under {base_path}."
            )

    feature_blocks: List[np.ndarray] = []
    label_blocks: List[np.ndarray] = []
    include_wine_type_feature = len(csv_paths) > 1

    for csv_path in csv_paths:
        data = np.loadtxt(csv_path, delimiter=";", skiprows=1, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(
                f"Expected tabular data with at least one feature and one target column in {csv_path}."
            )

        features = data[:, :-1].astype(np.float32)
        labels = data[:, -1].astype(np.int32)

        if include_wine_type_feature:
            stem = csv_path.stem.lower()
            if "red" in stem:
                wine_type_value = 1.0
            elif "white" in stem:
                wine_type_value = 0.0
            else:
                raise ValueError(
                    f"Could not infer wine type from {csv_path.name}. Expected 'red' or 'white' in the filename."
                )
            wine_type_feature = np.full((features.shape[0], 1), wine_type_value, dtype=np.float32)
            features = np.concatenate([features, wine_type_feature], axis=1)

        feature_blocks.append(features)
        label_blocks.append(labels)

    features_all = np.concatenate(feature_blocks, axis=0)
    labels_all = np.concatenate(label_blocks, axis=0)

    if features_all.shape[0] <= 1:
        raise ValueError("Need at least two winequality samples to construct a train/test split.")

    n_test = int(round(features_all.shape[0] * float(test_split)))
    n_test = max(1, min(features_all.shape[0] - 1, n_test))

    rng = np.random.default_rng(seed=seed)
    permutation = rng.permutation(features_all.shape[0])
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]

    if train_idx.size == 0:
        train_idx = permutation[:-1]
        test_idx = permutation[-1:]

    x_train = features_all[train_idx]
    x_test = features_all[test_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-7)
    x_train = ((x_train - mean) / std).reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = ((x_test - mean) / std).reshape(x_test.shape[0], -1).astype(np.float32)

    class_values = np.unique(labels_all)
    label_to_index = {int(label): idx for idx, label in enumerate(class_values.tolist())}
    labels_encoded = np.asarray([label_to_index[int(label)] for label in labels_all], dtype=np.int32)
    y_train_idx = labels_encoded[train_idx]
    y_test_idx = labels_encoded[test_idx]

    if one_hot:
        eye = np.eye(len(class_values), dtype=np.float32)
        y_train = eye[y_train_idx]
        y_test = eye[y_test_idx]
    else:
        y_train = y_train_idx.astype(np.int32)
        y_test = y_test_idx.astype(np.int32)

    return x_train, y_train, x_test, y_test


def load_drybean(
    path: str | Path = "drybean",
    *,
    test_split: float = 0.2,
    seed: int = 0,
    one_hot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the dry bean classification dataset with train-split standardization."""
    base_path = resolve_dataset_path(path)
    arff_path = base_path if base_path.is_file() else base_path / "Dry_Bean_Dataset.arff"

    if not arff_path.is_file():
        raise FileNotFoundError(f"Could not find Dry_Bean_Dataset.arff under {base_path}.")

    feature_rows: List[List[float]] = []
    label_rows: List[str] = []
    class_names: List[str] | None = None
    in_data = False

    with arff_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lower_line = line.lower()
            if not in_data:
                if lower_line.startswith("@attribute class"):
                    class_block = line[line.index("{") + 1:line.rindex("}")]
                    class_names = [name.strip() for name in class_block.split(",")]
                elif lower_line == "@data":
                    in_data = True
                continue

            values = [value.strip() for value in line.split(",")]
            feature_rows.append([float(value) for value in values[:-1]])
            label_rows.append(values[-1])

    features_all = np.asarray(feature_rows, dtype=np.float32)
    if features_all.shape[0] <= 1:
        raise ValueError("Need at least two drybean samples to construct a train/test split.")

    if class_names is None:
        class_names = sorted(set(label_rows))
    label_to_index = {name: idx for idx, name in enumerate(class_names)}
    labels_encoded = np.asarray([label_to_index[label] for label in label_rows], dtype=np.int32)

    n_test = int(round(features_all.shape[0] * float(test_split)))
    n_test = max(1, min(features_all.shape[0] - 1, n_test))

    rng = np.random.default_rng(seed=seed)
    permutation = rng.permutation(features_all.shape[0])
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]

    if train_idx.size == 0:
        train_idx = permutation[:-1]
        test_idx = permutation[-1:]

    x_train = features_all[train_idx]
    x_test = features_all[test_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-7)
    x_train = ((x_train - mean) / std).reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = ((x_test - mean) / std).reshape(x_test.shape[0], -1).astype(np.float32)

    y_train_idx = labels_encoded[train_idx]
    y_test_idx = labels_encoded[test_idx]

    if one_hot:
        eye = np.eye(len(class_names), dtype=np.float32)
        y_train = eye[y_train_idx]
        y_test = eye[y_test_idx]
    else:
        y_train = y_train_idx.astype(np.int32)
        y_test = y_test_idx.astype(np.int32)

    return x_train, y_train, x_test, y_test


def load_classification_dataset(
    path: str | Path = "classification_data",
    *,
    one_hot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an artificial classification dataset with predefined train/test indices."""
    base_dir = resolve_dataset_path(path)
    data_path = base_dir if base_dir.is_file() else base_dir / "data.npz"

    if not data_path.is_file():
        raise FileNotFoundError(f"Classification dataset file not found at {data_path}.")

    with np.load(data_path) as data:
        required_keys = {"inputs", "outputs", "train_idx", "test_idx"}
        if not required_keys <= set(data.files):
            raise ValueError(
                f"Expected {sorted(required_keys)} in {data_path}, found {sorted(data.files)}."
            )
        inputs = data["inputs"].astype(np.float32)
        outputs = np.asarray(data["outputs"])
        train_idx = data["train_idx"].astype(np.int64)
        test_idx = data["test_idx"].astype(np.int64)

    x_train = inputs[train_idx].reshape(train_idx.shape[0], -1)
    x_test = inputs[test_idx].reshape(test_idx.shape[0], -1)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-7)
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)

    labels = outputs.reshape(outputs.shape[0], -1)
    if labels.shape[1] == 1:
        labels = labels[:, 0].astype(np.int32)
        y_train_idx = labels[train_idx]
        y_test_idx = labels[test_idx]
        if one_hot:
            eye = np.eye(int(labels.max()) + 1, dtype=np.float32)
            y_train = eye[y_train_idx]
            y_test = eye[y_test_idx]
        else:
            y_train = y_train_idx.astype(np.int32)
            y_test = y_test_idx.astype(np.int32)
    else:
        y_train = labels[train_idx].astype(np.float32)
        y_test = labels[test_idx].astype(np.float32)

    return x_train, y_train, x_test, y_test


CALIFORNIA_HOUSING_NUMERIC_COLUMNS = (
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
)
CALIFORNIA_HOUSING_TARGET_COLUMN = "median_house_value"
CALIFORNIA_HOUSING_OCEAN_PROXIMITY_CATEGORIES = (
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN",
)


def load_california_housing_prices(
    path: str | Path = "california_housing_prices",
    *,
    test_split: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Kaggle California housing prices as a standardized regression dataset."""
    base_path = resolve_dataset_path(path)
    csv_path = base_path if base_path.is_file() else base_path / "housing.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Could not find housing.csv under {base_path}.")

    import pandas as pd

    dataframe = pd.read_csv(csv_path)
    required_columns = set(CALIFORNIA_HOUSING_NUMERIC_COLUMNS) | {
        CALIFORNIA_HOUSING_TARGET_COLUMN,
        "ocean_proximity",
    }
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"Missing expected California housing columns in {csv_path}: {missing_columns}."
        )

    if dataframe.shape[0] <= 1:
        raise ValueError("Need at least two California housing rows to construct a train/test split.")

    numeric_features = dataframe.loc[:, CALIFORNIA_HOUSING_NUMERIC_COLUMNS].to_numpy(
        dtype=np.float32
    )
    targets_all = dataframe.loc[:, [CALIFORNIA_HOUSING_TARGET_COLUMN]].to_numpy(
        dtype=np.float32
    )

    ocean_values = dataframe["ocean_proximity"].astype(str).to_numpy()
    category_to_index = {
        category: idx
        for idx, category in enumerate(CALIFORNIA_HOUSING_OCEAN_PROXIMITY_CATEGORIES)
    }
    unknown_categories = sorted(set(ocean_values) - set(category_to_index))
    if unknown_categories:
        raise ValueError(
            f"Unexpected ocean_proximity categories in {csv_path}: {unknown_categories}."
        )

    categorical_features = np.zeros(
        (ocean_values.shape[0], len(CALIFORNIA_HOUSING_OCEAN_PROXIMITY_CATEGORIES)),
        dtype=np.float32,
    )
    categorical_indices = np.asarray(
        [category_to_index[category] for category in ocean_values],
        dtype=np.int64,
    )
    categorical_features[np.arange(ocean_values.shape[0]), categorical_indices] = 1.0

    n_test = int(round(numeric_features.shape[0] * float(test_split)))
    n_test = max(1, min(numeric_features.shape[0] - 1, n_test))

    rng = np.random.default_rng(seed=seed)
    permutation = rng.permutation(numeric_features.shape[0])
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]

    if train_idx.size == 0:
        train_idx = permutation[:-1]
        test_idx = permutation[-1:]

    x_train_numeric = numeric_features[train_idx]
    x_test_numeric = numeric_features[test_idx]

    train_medians = np.nanmedian(x_train_numeric, axis=0, keepdims=True).astype(np.float32)
    if np.any(np.isnan(train_medians)):
        raise ValueError("At least one California housing numeric column is all-NaN in the train split.")
    x_train_numeric = np.where(np.isnan(x_train_numeric), train_medians, x_train_numeric)
    x_test_numeric = np.where(np.isnan(x_test_numeric), train_medians, x_test_numeric)

    numeric_mean = x_train_numeric.mean(axis=0, keepdims=True)
    numeric_std = x_train_numeric.std(axis=0, keepdims=True)
    numeric_std = np.maximum(numeric_std, 1e-7)
    x_train_numeric = (x_train_numeric - numeric_mean) / numeric_std
    x_test_numeric = (x_test_numeric - numeric_mean) / numeric_std

    x_train = np.concatenate([x_train_numeric, categorical_features[train_idx]], axis=1)
    x_test = np.concatenate([x_test_numeric, categorical_features[test_idx]], axis=1)

    y_train = targets_all[train_idx]
    y_test = targets_all[test_idx]
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.maximum(y_std, 1e-7)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    return (
        x_train.reshape(x_train.shape[0], -1).astype(np.float32),
        y_train.astype(np.float32),
        x_test.reshape(x_test.shape[0], -1).astype(np.float32),
        y_test.astype(np.float32),
    )


def load_ct_slice_localization(
    path: str | Path = "ct_slice_localization",
    *,
    test_split: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load UCI CT slice localization as a standardized regression dataset."""
    base_path = resolve_dataset_path(path)
    csv_path = base_path if base_path.is_file() else base_path / "slice_localization_data.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Could not find slice_localization_data.csv under {base_path}.")

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 386:
        raise ValueError(
            "Expected CT slice localization data with shape (num_examples, 386), "
            f"got {data.shape} from {csv_path}."
        )
    if data.shape[0] <= 1:
        raise ValueError("Need at least two CT slice rows to construct a train/test split.")

    features_all = data[:, 1:-1]
    targets_all = data[:, -1:]

    if features_all.shape[1] != 384:
        raise ValueError(
            f"Expected 384 CT slice histogram features, found {features_all.shape[1]}."
        )

    n_test = int(round(features_all.shape[0] * float(test_split)))
    n_test = max(1, min(features_all.shape[0] - 1, n_test))

    rng = np.random.default_rng(seed=seed)
    permutation = rng.permutation(features_all.shape[0])
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]

    if train_idx.size == 0:
        train_idx = permutation[:-1]
        test_idx = permutation[-1:]

    x_train = features_all[train_idx]
    x_test = features_all[test_idx]
    y_train = targets_all[train_idx]
    y_test = targets_all[test_idx]

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.maximum(x_std, 1e-7)
    x_train = ((x_train - x_mean) / x_std).reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = ((x_test - x_mean) / x_std).reshape(x_test.shape[0], -1).astype(np.float32)

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.maximum(y_std, 1e-7)
    y_train = ((y_train - y_mean) / y_std).astype(np.float32)
    y_test = ((y_test - y_mean) / y_std).astype(np.float32)

    return x_train, y_train, x_test, y_test


def load_yearpredictionmsd(path: str | Path = "yearpredictionmsd") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load YearPredictionMSD with the fixed publication train/test split."""
    base_path = resolve_dataset_path(path)
    data_path = base_path if base_path.is_file() else base_path / "YearPredictionMSD.txt"

    if not data_path.is_file():
        raise FileNotFoundError(f"Could not find YearPredictionMSD.txt under {base_path}.")

    data = np.loadtxt(data_path, delimiter=",", dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 91:
        raise ValueError(
            "Expected YearPredictionMSD data with shape (num_examples, 91), "
            f"got {data.shape} from {data_path}."
        )

    n_train = 463_715
    n_test = 51_630
    total_expected = n_train + n_test
    if data.shape[0] != total_expected:
        raise ValueError(
            f"Expected {total_expected} YearPredictionMSD rows, found {data.shape[0]} in {data_path}."
        )

    x_train = data[:n_train, 1:]
    y_train = data[:n_train, :1]
    x_test = data[n_train:, 1:]
    y_test = data[n_train:, :1]

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.maximum(x_std, 1e-7)
    x_train = ((x_train - x_mean) / x_std).reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = ((x_test - x_mean) / x_std).reshape(x_test.shape[0], -1).astype(np.float32)

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.maximum(y_std, 1e-7)
    y_train = ((y_train - y_mean) / y_std).astype(np.float32)
    y_test = ((y_test - y_mean) / y_std).astype(np.float32)

    return x_train, y_train, x_test, y_test


def load_teacher_dataset(path) -> Tuple[np.ndarray, np.ndarray]:
    base_dir = resolve_dataset_path(path)

    # expected_folder = f"scheme-{cfg.param_scheme}_widths-{format_widths(cfg.widths)}"
    if base_dir.is_file():
        data_path = base_dir
    else:
        data_path = base_dir / "data.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Teacher dataset not found at {data_path}. Generate it with generate_teacher_data.py."
        )
    with np.load(data_path) as data:
        if "inputs" in data and "outputs" in data:
            inputs = data["inputs"].astype(np.float32)
            outputs = data["outputs"].astype(np.float32)
        elif all(key in data for key in ("x_train", "y_train", "x_test", "y_test")):
            x_train = data["x_train"].astype(np.float32)
            y_train = data["y_train"].astype(np.float32)
            x_test = data["x_test"].astype(np.float32)
            y_test = data["y_test"].astype(np.float32)
            inputs = np.concatenate([x_train, x_test], axis=0)
            outputs = np.concatenate([y_train, y_test], axis=0)
        else:
            raise ValueError(
                "Unsupported teacher dataset format. Expected 'inputs'/'outputs' arrays "
                "or legacy 'x_train'/'y_train'/'x_test'/'y_test'."
            )
    return inputs, outputs


def load_teacher_2048x1_dataset(
    path: str | Path = "teacher_data/2048x1",
    *,
    test_split: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load teacher_data/2048x1 with train-split feature and target normalization."""
    inputs, outputs = load_teacher_dataset(path)
    inputs = inputs.reshape(inputs.shape[0], -1).astype(np.float32)
    outputs = outputs.reshape(outputs.shape[0], -1).astype(np.float32)

    if inputs.ndim != 2 or inputs.shape[1] != 512:
        raise ValueError(
            f"Expected teacher_data/2048x1 inputs with shape (n, 512), got {inputs.shape}."
        )
    if outputs.ndim != 2 or outputs.shape[1] != 1:
        raise ValueError(
            f"Expected teacher_data/2048x1 outputs with shape (n, 1), got {outputs.shape}."
        )
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError(
            f"Teacher input/output count mismatch: {inputs.shape[0]} inputs, "
            f"{outputs.shape[0]} outputs."
        )
    if inputs.shape[0] <= 1:
        raise ValueError("Need at least two teacher samples to construct a train/test split.")

    n_test = int(round(inputs.shape[0] * float(test_split)))
    n_test = max(1, min(inputs.shape[0] - 1, n_test))

    rng = np.random.default_rng(seed=seed)
    permutation = rng.permutation(inputs.shape[0])
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]

    if train_idx.size == 0:
        train_idx = permutation[:-1]
        test_idx = permutation[-1:]

    x_train = inputs[train_idx]
    y_train = outputs[train_idx]
    x_test = inputs[test_idx]
    y_test = outputs[test_idx]

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.maximum(x_std, 1e-7)
    x_train = ((x_train - x_mean) / x_std).astype(np.float32)
    x_test = ((x_test - x_mean) / x_std).astype(np.float32)

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.maximum(y_std, 1e-7)
    y_train = ((y_train - y_mean) / y_std).astype(np.float32)
    y_test = ((y_test - y_mean) / y_std).astype(np.float32)

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

def _is_mnist_dataset(dataset_path: Path) -> bool:
    return (
        dataset_path.is_dir()
        and (dataset_path / "train-images-idx3-ubyte").is_file()
        and (dataset_path / "train-labels-idx1-ubyte").is_file()
        and (dataset_path / "t10k-images-idx3-ubyte").is_file()
        and (dataset_path / "t10k-labels-idx1-ubyte").is_file()
    )

def _is_emnist_byclass_dataset(dataset_path: Path) -> bool:
    base_dir = dataset_path / "byclass" if (dataset_path / "byclass").is_dir() else dataset_path
    return (
        base_dir.is_dir()
        and (base_dir / "emnist-byclass-train-images-idx3-ubyte").is_file()
        and (base_dir / "emnist-byclass-train-labels-idx1-ubyte").is_file()
        and (base_dir / "emnist-byclass-test-images-idx3-ubyte").is_file()
        and (base_dir / "emnist-byclass-test-labels-idx1-ubyte").is_file()
    )

def _is_svhn_dataset(dataset_path: Path) -> bool:
    return (
        dataset_path.is_dir()
        and (dataset_path / "train_32x32.mat").is_file()
        and (dataset_path / "test_32x32.mat").is_file()
    )

def _is_winequality_dataset(dataset_path: Path) -> bool:
    if dataset_path.is_file():
        return dataset_path.suffix.lower() == ".csv" and dataset_path.name.startswith("winequality-")
    return (
        dataset_path.is_dir()
        and any(
            (dataset_path / filename).is_file()
            for filename in ("winequality-red.csv", "winequality-white.csv")
        )
    )

def _is_drybean_dataset(dataset_path: Path) -> bool:
    if dataset_path.is_file():
        return dataset_path.suffix.lower() == ".arff" and dataset_path.name == "Dry_Bean_Dataset.arff"
    return dataset_path.is_dir() and (dataset_path / "Dry_Bean_Dataset.arff").is_file()

def _is_classification_dataset(dataset_path: Path) -> bool:
    data_path = dataset_path if dataset_path.is_file() else dataset_path / "data.npz"
    if not data_path.is_file():
        return False
    with np.load(data_path) as data:
        return {"inputs", "outputs", "train_idx", "test_idx"} <= set(data.files)

def _is_yearpredictionmsd_dataset(dataset_path: Path) -> bool:
    if dataset_path.is_file():
        return dataset_path.name == "YearPredictionMSD.txt"
    return dataset_path.is_dir() and (dataset_path / "YearPredictionMSD.txt").is_file()

def _is_california_housing_prices_dataset(dataset_path: Path) -> bool:
    csv_path = dataset_path if dataset_path.is_file() else dataset_path / "housing.csv"
    if not csv_path.is_file():
        return False

    required_columns = {
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "median_house_value",
        "ocean_proximity",
    }
    with csv_path.open("r", encoding="utf-8") as file:
        header = file.readline().strip().split(",")
    return required_columns <= set(header)

def _is_ct_slice_localization_dataset(dataset_path: Path) -> bool:
    csv_path = dataset_path if dataset_path.is_file() else dataset_path / "slice_localization_data.csv"
    if not csv_path.is_file():
        return False

    with csv_path.open("r", encoding="utf-8") as file:
        header = file.readline().strip().split(",")
    return (
        len(header) == 386
        and header[0] == "patientId"
        and header[1] == "value0"
        and header[-2] == "value383"
        and header[-1] == "reference"
    )

def _is_teacher_2048x1_dataset(dataset_path: Path) -> bool:
    data_path = dataset_path if dataset_path.is_file() else dataset_path / "data.npz"
    return (
        data_path.is_file()
        and data_path.name == "data.npz"
        and data_path.parent.name == "2048x1"
    )


def _random_split(
    inputs: np.ndarray,
    targets: np.ndarray,
    test_split: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = int(inputs.shape[0])
    if total <= 1:
        raise ValueError("Need at least two samples to construct a train/test split.")
    n_test = int(round(total * float(test_split)))
    n_test = max(1, min(total - 1, n_test))
    permutation = rng.permutation(total)
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]
    return inputs[train_idx], targets[train_idx], inputs[test_idx], targets[test_idx]


def load_dataset(
    task: str | Path,
    test_split: float = 0.2,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load the dataset referenced by ``task``.

    Returns ``(x_train, y_train, x_test, y_test, test_split)`` where ``test_split`` is the
    fraction of test samples actually used (datasets with a predefined split ignore the
    requested ``test_split``). Datasets without a predefined split are split with
    ``seed=0`` so that the analysis scripts can recover the exact test set; only the
    generic teacher datasets fall back to ``rng``.
    """
    dataset_path = resolve_dataset_path(task)
    test_split = float(test_split)

    if _is_yearpredictionmsd_dataset(dataset_path):
        data = load_yearpredictionmsd(dataset_path)
    elif _is_california_housing_prices_dataset(dataset_path):
        data = load_california_housing_prices(dataset_path, test_split=test_split, seed=0)
    elif _is_ct_slice_localization_dataset(dataset_path):
        data = load_ct_slice_localization(dataset_path, test_split=test_split, seed=0)
    elif _is_drybean_dataset(dataset_path):
        data = load_drybean(dataset_path, test_split=test_split, seed=0, one_hot=True)
    elif _is_classification_dataset(dataset_path):
        data = load_classification_dataset(dataset_path, one_hot=True)
    elif _is_winequality_dataset(dataset_path):
        data = load_winequality(dataset_path, test_split=test_split, seed=0, one_hot=True)
    elif _is_mnist_dataset(dataset_path):
        data = load_mnist(dataset_path)
    elif _is_emnist_byclass_dataset(dataset_path):
        data = load_emnist_byclass(dataset_path)
    elif _is_svhn_dataset(dataset_path):
        data = load_svhn(dataset_path)
    elif _is_cifar10_dataset(dataset_path):
        data = load_all_cifar10_data(dataset_path)
    elif _is_teacher_2048x1_dataset(dataset_path):
        data = load_teacher_2048x1_dataset(dataset_path, test_split=test_split, seed=0)
    else:
        inputs, targets = load_teacher_dataset(dataset_path)
        data = _random_split(inputs, targets, test_split, rng or np.random.default_rng())

    x_train, y_train, x_test, y_test = (np.asarray(a, dtype=np.float32) for a in data)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    effective_split = float(x_test.shape[0] / (x_train.shape[0] + x_test.shape[0]))
    return x_train, y_train, x_test, y_test, effective_split
