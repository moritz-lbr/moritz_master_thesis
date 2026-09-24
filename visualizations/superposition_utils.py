"""Shared helpers for the superposition-matrix analysis (thesis Section 5.3).

Extracted from the plotting script plot_superposition.py of the original working
repository; only the functions used by analyze_superposition.py and
mp_spectra_corr.py are kept. The helpers read Dense-layer weight snapshots directly
from the zarr v3 stores written during training (weights.zarr), evaluate
(reconstructed) models on the test split and provide common plot styling.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from math import isqrt
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from numcodecs.zstd import Zstd
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from scipy.sparse.linalg import eigsh
except Exception:  # pragma: no cover - scipy is optional for this plotting script.
    eigsh = None


LEGEND_FONTSIZE = 12


@dataclass(frozen=True)
class ImageLayout:
    height: int
    width: int
    channels: int
    channel_names: tuple[str, ...]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data if isinstance(data, dict) else {}


def training_step_stride(run_dir: Path) -> int:
    training = load_yaml(run_dir / "simulation_config.yaml").get("training", {})
    value = training.get("save_loss_frequency", 1)
    if isinstance(value, str):
        if value.strip().lower() == "epoch":
            batch_size = int(training.get("batch_size", 1))
            n_train = int(training.get("training_data", {}).get("n_train", batch_size))
            return 1 if batch_size >= n_train else n_train // batch_size
        return int(float(value))
    return int(value)


def task_name(run_dir: Path) -> str | None:
    task = (
        load_yaml(run_dir / "simulation_config.yaml")
        .get("training", {})
        .get("training_data", {})
        .get("task")
    )
    return Path(task).name if task else None


def task_path(run_dir: Path) -> Path | None:
    task = (
        load_yaml(run_dir / "simulation_config.yaml")
        .get("training", {})
        .get("training_data", {})
        .get("task")
    )
    if not task:
        return None
    try:
        return import_data_module().resolve_dataset_path(task)
    except FileNotFoundError:
        return Path(str(task)).expanduser()


def square_layout(unit_count: int) -> ImageLayout | None:
    side = isqrt(unit_count)
    if side * side != unit_count:
        return None
    return ImageLayout(side, side, 1, ("gray",))


def rgb_layout(unit_count: int) -> ImageLayout | None:
    if unit_count % 3 != 0:
        return None
    side = isqrt(unit_count // 3)
    if side * side * 3 != unit_count:
        return None
    return ImageLayout(side, side, 3, ("R", "G", "B"))


def infer_image_layout(run_dir: Path, unit_count: int) -> ImageLayout | None:
    path = task_path(run_dir)
    overview = load_yaml(path / "dataset_overview.yaml") if path and path.is_dir() else {}
    task_text = " ".join(
        str(value).lower()
        for value in (
            path.name if path else "",
            path.as_posix() if path else "",
            overview.get("training", {}).get("variant", ""),
        )
    )

    if unit_count == 784 and ("mnist" in task_text or "emnist" in task_text):
        return ImageLayout(28, 28, 1, ("gray",))
    if unit_count == 3072 and (
        "svhn" in task_text or "cifar" in task_text or "32x32" in task_text
    ):
        return ImageLayout(32, 32, 3, ("R", "G", "B"))

    layout = square_layout(unit_count)
    if layout is not None:
        return layout
    return rgb_layout(unit_count)


def zarr_paths(input_dir: Path) -> list[Path]:
    if input_dir.name == "weights.zarr" and input_dir.is_dir():
        return [input_dir]
    return sorted(path for path in input_dir.rglob("weights.zarr") if path.is_dir())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def load_json_top_level_field(path: Path, field: str) -> Any:
    marker = f'"{field}"'
    decoder = json.JSONDecoder()
    buffer = ""
    value_start: int | None = None

    with path.open("r", encoding="utf-8") as file:
        while True:
            chunk = file.read(65536)
            if chunk:
                buffer += chunk

            if value_start is None:
                marker_index = buffer.find(marker)
                if marker_index < 0:
                    if not chunk:
                        return None
                    keep = len(marker) + 8
                    if len(buffer) > keep:
                        buffer = buffer[-keep:]
                    continue
                colon_index = buffer.find(":", marker_index + len(marker))
                if colon_index < 0:
                    if not chunk:
                        return None
                    continue
                value_start = colon_index + 1
                while value_start < len(buffer) and buffer[value_start].isspace():
                    value_start += 1

            try:
                value, _ = decoder.raw_decode(buffer, value_start)
                return value
            except json.JSONDecodeError:
                if not chunk:
                    raise


def load_training_final_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "training_log.json"
    if not path.exists():
        return {}
    value = load_json_top_level_field(path, "final_metrics")
    return value if isinstance(value, dict) else {}


def scheme_from_run(run: str) -> str:
    text = str(run).lower()
    if "standard" in text:
        return "standard"
    if "mup" in text or "mu_p" in text or "mu-p" in text:
        return "muP"
    return "muP"


def label_from_run(run: str) -> str:
    return run.rsplit("-", maxsplit=1)[-1]


def width_key(label: str) -> Sequence[int]:
    try:
        return [int(part) for part in label.split("x")]
    except ValueError:
        return [int(match) for match in re.findall(r"\d+", label)] or [10**9]


def combined_legend(legend_entries: dict[str, list[tuple[Any, str]]]) -> tuple[list[Any], list[str], list[int]]:
    handles: list[Any] = []
    labels: list[str] = []
    header_indices: list[int] = []
    titles = {"standard": "Standard parametrization", "muP": "muP parametrization"}
    for scheme in ("standard", "muP"):
        entries = sorted(legend_entries[scheme], key=lambda item: tuple(width_key(item[1])))
        if not entries:
            continue
        handles.append(plt.Line2D([], [], linestyle="", marker="", linewidth=0))
        labels.append(titles[scheme])
        header_indices.append(len(labels) - 1)
        for handle, label in entries:
            handles.append(handle)
            labels.append(label)
    return handles, labels, header_indices


def keep_log_step(step: int, min_step: int) -> bool:
    if step == 0:
        return True
    if step < min_step:
        return False
    decade = 10 ** (len(str(step)) - 1)
    return step % decade == 0


def selected_times(length: int, frequency: int, min_step: int) -> list[tuple[int, int]]:
    return [
        (index, int(step))
        for index, step in enumerate(np.arange(length, dtype=np.int64) * frequency)
        if keep_log_step(int(step), min_step)
    ]


def set_nonnegative_loglike_xscale(ax: plt.Axes, arrays: Sequence[Any]) -> None:
    finite_values = []
    for values in arrays:
        array = np.asarray(values, dtype=float).ravel()
        finite = array[np.isfinite(array)]
        if finite.size:
            finite_values.append(finite)
    if not finite_values:
        return
    values = np.concatenate(finite_values)
    if np.all(values > 0):
        ax.set_xscale("log")
        return
    positive = values[values > 0]
    if np.all(values >= 0) and positive.size:
        min_positive = float(np.min(positive))
        max_positive = float(np.max(positive))
        zero_position = min_positive / 10.0
        for line in ax.get_lines():
            xdata = np.asarray(line.get_xdata(), dtype=float)
            if xdata.size:
                line.set_xdata(np.where(xdata == 0.0, zero_position, xdata))
        ax.set_xscale("log")
        decade_min = int(np.floor(np.log10(min_positive)))
        decade_max = int(np.ceil(np.log10(max_positive)))
        positive_ticks = [
            10.0**exponent
            for exponent in range(decade_min, decade_max + 1)
            if min_positive <= 10.0**exponent <= max_positive
        ]
        ticks = [zero_position] + positive_ticks
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: (
                    "0"
                    if np.isclose(value, zero_position)
                    else rf"$10^{{{int(round(np.log10(value)))}}}$"
                    if value > 0 and np.isclose(np.log10(value), round(np.log10(value)))
                    else f"{value:g}"
                )
            )
        )
        ax.set_xlim(zero_position / np.sqrt(10.0), max_positive * np.sqrt(10.0))


def matrix_keys(store: Path) -> list[str]:
    attrs = load_json(store / "zarr.json").get("attributes", {})
    keys = list(attrs.get("param_keys", []))
    return [
        key
        for key in keys
        if len(load_json(store / key / "zarr.json").get("shape", [])) >= 3
    ]


def layer_name(key: str) -> str:
    parts = key.split("/")
    return parts[-2] if parts[-1] == "kernel" and len(parts) > 1 else key.replace("/", "_")


def selected_reference_units(
    unit_count: int,
    count: int,
    *,
    seed: int = 0,
    layout: ImageLayout | None = None,
) -> np.ndarray:
    if unit_count < 1 or count < 1:
        return np.empty(0, dtype=np.int64)
    count = min(int(count), int(unit_count))
    rng = np.random.default_rng(seed)
    if layout is not None and layout.channels > 1:
        selected: list[int] = []
        for channel in range(min(count, layout.channels)):
            channel_units = np.arange(channel, unit_count, layout.channels, dtype=np.int64)
            if channel_units.size:
                selected.append(int(rng.choice(channel_units)))
        remaining = count - len(selected)
        if remaining > 0:
            available = np.setdiff1d(np.arange(unit_count, dtype=np.int64), np.asarray(selected, dtype=np.int64))
            selected.extend(int(unit) for unit in rng.choice(available, size=remaining, replace=False))
        return np.asarray(selected, dtype=np.int64)
    return np.sort(rng.choice(unit_count, size=count, replace=False).astype(np.int64))


def reference_highlight_colors(reference_units: np.ndarray, layout: ImageLayout) -> list[str]:
    channel_colors = {
        "R": "red",
        "G": "green",
        "B": "blue",
        "red": "red",
        "green": "green",
        "blue": "blue",
    }
    fallback = ["red", "green", "blue"]
    colors: list[str] = []
    for index, ref_unit in enumerate(reference_units):
        _, _, channel = unit_image_coordinates(int(ref_unit), layout)
        channel_name = layout.channel_names[channel] if channel < len(layout.channel_names) else ""
        colors.append(channel_colors.get(channel_name, fallback[index % len(fallback)]))
    return colors


def normalized_input_vectors(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=norms > 0,
    )


def test_loss_history(run_dir: Path) -> np.ndarray:
    metrics = load_training_final_metrics(run_dir)
    history = metrics.get("history", {})
    if not isinstance(history, dict):
        return np.empty(0, dtype=float)
    return np.asarray(history.get("test_loss", []), dtype=float)


def history_index_to_time_index(
    history_index: int,
    *,
    history_length: int,
    time_count: int,
) -> int:
    if int(time_count) == int(history_length) + 1:
        return int(history_index) + 1
    if int(time_count) == int(history_length):
        return int(history_index)
    preferred = int(history_index) + 1
    return preferred if preferred < int(time_count) else int(history_index)


def best_test_loss_snapshot(
    run_dir: Path,
    time_count: int,
    step_stride: int,
) -> dict[str, Any] | None:
    losses = test_loss_history(run_dir)
    finite = np.flatnonzero(np.isfinite(losses))
    if finite.size == 0:
        return None

    best_history_index = int(finite[np.argmin(losses[finite])])
    best_time_index = history_index_to_time_index(
        best_history_index,
        history_length=int(losses.size),
        time_count=int(time_count),
    )

    if best_time_index < 0 or best_time_index >= int(time_count):
        print(
            f"Skipping best-loss spectrum for {run_dir}: best test-loss index "
            f"{best_history_index} has no saved weight snapshot"
        )
        return None

    return {
        "best_test_loss": float(losses[best_history_index]),
        "best_history_index": best_history_index,
        "best_time_index": int(best_time_index),
        "best_step": int(best_time_index) * int(step_stride),
    }


def read_snapshot(array_dir: Path, meta: dict[str, Any], time_index: int) -> np.ndarray:
    shape = tuple(int(v) for v in meta["shape"])
    chunk_shape = tuple(int(v) for v in meta["chunk_grid"]["configuration"]["chunk_shape"])
    if chunk_shape[1:] != shape[1:]:
        raise ValueError(f"Only full-spatial chunks are supported: {array_dir}")

    chunk_index = time_index // chunk_shape[0]
    inner_index = time_index % chunk_shape[0]
    sep = meta.get("chunk_key_encoding", {}).get("configuration", {}).get("separator", "/")
    chunk_key = [str(chunk_index)] + ["0"] * (len(shape) - 1)
    chunk_path = (
        array_dir / "c" / Path(*chunk_key)
        if sep == "/"
        else array_dir / "c" / sep.join(chunk_key)
    )
    raw = chunk_path.read_bytes()
    if any(codec.get("name") == "zstd" for codec in meta.get("codecs", [])):
        raw = Zstd().decode(raw)

    dtype = np.dtype(meta["data_type"])
    endian = next(
        (
            codec.get("configuration", {}).get("endian")
            for codec in meta.get("codecs", [])
            if codec.get("name") == "bytes"
        ),
        None,
    )
    if endian == "little":
        dtype = dtype.newbyteorder("<")
    elif endian == "big":
        dtype = dtype.newbyteorder(">")

    chunk = np.frombuffer(raw, dtype=dtype).reshape(chunk_shape)
    return np.asarray(chunk[inner_index], dtype=np.float32)


def unit_image_coordinates(unit: int, layout: ImageLayout) -> tuple[int, int, int]:
    pixel_index = int(unit) // layout.channels
    channel = int(unit) % layout.channels
    row = pixel_index // layout.width
    col = pixel_index % layout.width
    return row, col, channel


def sorted_series_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda item: tuple(width_key(label_from_run(item["run"]))))


def series_colors(
    entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[tuple[Any, str]]]]:
    scheme_groups = {
        "standard": [entry for entry in entries if scheme_from_run(entry["run"]) == "standard"],
        "muP": [entry for entry in entries if scheme_from_run(entry["run"]) == "muP"],
    }
    color_map = {
        "standard": plt.cm.autumn(np.linspace(0, 1, max(len(scheme_groups["standard"]), 1))),
        "muP": plt.cm.winter(np.linspace(0, 1, max(len(scheme_groups["muP"]), 1))),
    }
    color_indices = {"standard": 0, "muP": 0}
    colors: dict[str, Any] = {}
    legend_entries: dict[str, list[tuple[Any, str]]] = {"standard": [], "muP": []}
    for entry in sorted_series_entries(entries):
        scheme = scheme_from_run(entry["run"])
        colors[entry["run"]] = color_map[scheme][color_indices[scheme]]
        color_indices[scheme] += 1
    return colors, legend_entries


def apply_combined_legend(
    ax: plt.Axes,
    legend_entries: dict[str, list[tuple[Any, str]]],
) -> None:
    handles, labels, header_indices = combined_legend(legend_entries)
    if handles:
        legend = ax.legend(handles, labels, fontsize=LEGEND_FONTSIZE, frameon=False)
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        for index in header_indices:
            legend.get_texts()[index].set_fontweight("bold")


def import_data_module() -> Any:
    src_dir = REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import data  # type: ignore[import-not-found]

    return data


def load_reconstruction_eval_data(
    run_dir: Path,
    cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the test split that was used to evaluate the run during training."""
    data = import_data_module()
    config = load_yaml(run_dir / "simulation_config.yaml")
    training = config.get("training", {})
    training_data = training.get("training_data", {})
    task = training_data.get("task")
    if task is None:
        raise ValueError(f"Missing training_data.task in {run_dir / 'simulation_config.yaml'}")

    dataset_path = data.resolve_dataset_path(task)
    cache_key = (str(dataset_path), "predefined_split", 0.0)
    if cache_key in cache:
        return cache[cache_key]

    _, _, x_test, y_test, _ = data.load_dataset(dataset_path)
    result = (np.asarray(x_test, dtype=np.float32), np.asarray(y_test, dtype=np.float32))
    cache[cache_key] = result
    return result


def numpy_activation(name: str) -> Any:
    if name == "identity":
        return lambda x: x
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh
    if name == "gelu":
        return lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    if name == "sigmoid":
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    raise ValueError(f"Unsupported activation {name!r} for spectral reconstruction")


def classification_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    labels_array = np.asarray(labels)
    if labels_array.ndim == 2 and labels_array.shape[-1] == logits.shape[-1]:
        labels_onehot = labels_array.astype(np.float32)
    elif labels_array.ndim == 2 and labels_array.shape[-1] == 1:
        labels_onehot = np.eye(logits.shape[-1], dtype=np.float32)[labels_array[:, 0].astype(np.int64)]
    else:
        labels_onehot = np.eye(logits.shape[-1], dtype=np.float32)[labels_array.reshape(-1).astype(np.int64)]

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return float(-np.mean(np.sum(labels_onehot * log_probs, axis=1)))


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((predictions - targets) ** 2))


def evaluate_dense_two_layer_loss(
    x_test: np.ndarray,
    y_test: np.ndarray,
    dense0_kernel: np.ndarray,
    dense1_kernel: np.ndarray,
    *,
    activation0: Any,
    activation1: Any,
    loss_type: str,
    batch_size: int,
) -> float:
    losses = []
    weights = []
    for start in range(0, x_test.shape[0], batch_size):
        stop = min(start + batch_size, x_test.shape[0])
        xb = x_test[start:stop]
        logits = activation1(activation0(xb @ dense0_kernel) @ dense1_kernel)
        yb = y_test[start:stop]
        if loss_type == "cross_entropy":
            loss = classification_cross_entropy(logits, yb)
        elif loss_type == "mse":
            loss = mean_squared_error(logits, yb)
        else:
            raise ValueError(f"Unsupported loss_type {loss_type!r}")
        losses.append(loss)
        weights.append(stop - start)
    return float(np.average(losses, weights=weights))


def dense_layer_index(layer: str) -> int:
    match = re.fullmatch(r"Dense_(\d+)", str(layer))
    if match is None:
        return 10**9
    return int(match.group(1))


def dense_kernel_entries(zarr_path: Path) -> list[tuple[str, str, Path, dict[str, Any], tuple[int, ...]]]:
    entries: list[tuple[str, str, Path, dict[str, Any], tuple[int, ...]]] = []
    for key in matrix_keys(zarr_path):
        layer = layer_name(key)
        if not layer.startswith("Dense_"):
            continue
        array_dir = zarr_path / key
        meta = load_json(array_dir / "zarr.json")
        shape = tuple(int(v) for v in meta["shape"])
        if len(shape) < 3:
            continue
        entries.append((layer, key, array_dir, meta, shape))
    return sorted(entries, key=lambda item: dense_layer_index(item[0]))


def top_svd(matrix: np.ndarray, max_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_modes = max(1, min(int(max_modes), min(matrix.shape)))
    min_dim = min(matrix.shape)
    try:
        if max_modes < min_dim and max_modes <= max(1, min_dim // 2):
            from scipy.sparse.linalg import svds

            u, s, vt = svds(matrix, k=max_modes, which="LM")
            order = np.argsort(s)[::-1]
            return (
                np.asarray(u[:, order], dtype=np.float32),
                np.asarray(s[order], dtype=np.float32),
                np.asarray(vt[order, :], dtype=np.float32),
            )
    except Exception as exc:
        print(f"Falling back to full SVD for spectral reconstruction: {exc}")

    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return (
        np.asarray(u[:, :max_modes], dtype=np.float32),
        np.asarray(s[:max_modes], dtype=np.float32),
        np.asarray(vt[:max_modes, :], dtype=np.float32),
    )


def spectral_basis_for_dense0(
    dense0_kernel: np.ndarray,
    max_modes: int,
    *,
    raw_analysis: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    original_row_norms = np.linalg.norm(dense0_kernel, axis=1).astype(np.float32)
    nonzero_rows = int(np.count_nonzero(original_row_norms > 0))
    if raw_analysis:
        row_norms = np.ones(dense0_kernel.shape[0], dtype=np.float32)
        basis_matrix = dense0_kernel.astype(np.float32, copy=False)
    else:
        row_norms = original_row_norms
        basis_matrix = normalized_input_vectors(dense0_kernel)
    u, singular_values, vt = top_svd(basis_matrix, max_modes)
    return row_norms, u, singular_values, vt, nonzero_rows


def evaluate_reconstructed_loss(
    x_test: np.ndarray,
    y_test: np.ndarray,
    dense1_kernel: np.ndarray,
    *,
    row_norms: np.ndarray,
    u: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
    modes: int,
    renormalize: bool,
    activation0: Any,
    activation1: Any,
    loss_type: str,
    batch_size: int,
) -> tuple[float, dict[str, float]]:
    k_eff = min(int(modes), singular_values.size)
    u_k = u[:, :k_eff]
    s_k = singular_values[:k_eff]
    vt_k = vt[:k_eff, :]
    retained_per_feature = np.sum((u_k * s_k[None, :]) ** 2, axis=1)
    retained_total = float(np.sum(s_k.astype(np.float64) ** 2))
    total_norm_sq = float(np.count_nonzero(row_norms > 0))
    err_q = float(np.sqrt(max(total_norm_sq - retained_total, 0.0) / max(total_norm_sq, 1.0)))

    if renormalize:
        reconstructed_norms = np.sqrt(np.maximum(retained_per_feature, 0.0)).astype(np.float32)
        scale = np.divide(
            row_norms,
            reconstructed_norms,
            out=np.zeros_like(row_norms, dtype=np.float32),
            where=reconstructed_norms > 0,
        )
    else:
        scale = row_norms

    left = u_k * s_k[None, :]
    losses = []
    weights = []
    for start in range(0, x_test.shape[0], batch_size):
        stop = min(start + batch_size, x_test.shape[0])
        xb = x_test[start:stop]
        hidden_pre = ((xb * scale[None, :]) @ left) @ vt_k
        logits = activation1(activation0(hidden_pre) @ dense1_kernel)
        yb = y_test[start:stop]
        if loss_type == "cross_entropy":
            loss = classification_cross_entropy(logits, yb)
        elif loss_type == "mse":
            loss = mean_squared_error(logits, yb)
        else:
            raise ValueError(f"Unsupported loss_type {loss_type!r}")
        losses.append(loss)
        weights.append(stop - start)

    diagnostics = {
        "k_effective": float(k_eff),
        "retained_mean": float(np.mean(retained_per_feature)),
        "retained_min": float(np.min(retained_per_feature)),
        "retained_max": float(np.max(retained_per_feature)),
        "err_Q": err_q,
    }
    return float(np.average(losses, weights=weights)), diagnostics


def spectral_reconstructed_dense0_kernel(
    row_norms: np.ndarray,
    u: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
    modes: int,
) -> tuple[np.ndarray, int]:
    k_eff = min(int(modes), singular_values.size)
    left = u[:, :k_eff] * singular_values[:k_eff][None, :]
    reconstructed = left @ vt[:k_eff, :]
    reconstructed = reconstructed * row_norms[:, None]
    return np.asarray(reconstructed, dtype=np.float32), k_eff


def set_padded_ylim(ax: plt.Axes, arrays: Sequence[np.ndarray]) -> None:
    finite = [np.asarray(values, dtype=float)[np.isfinite(values)] for values in arrays]
    finite = [values for values in finite if values.size]
    if not finite:
        return
    values = np.concatenate(finite)
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    span = ymax - ymin
    pad = max(0.08 * span, 0.02 * max(abs(ymin), abs(ymax), 1.0), 1e-3)
    ax.set_ylim(ymin - pad, ymax + pad)
