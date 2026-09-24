"""Hidden-layer activation changes across network widths (thesis Fig. 3.1).

For every run of a job directory, the Dense_0 weight snapshots in ``weights.zarr`` are
used to compute the mean absolute change of the hidden pre-activations h and
post-activations z on the test set between consecutive saved training steps. The
changes are plotted against the width n (log-log) together with a fitted scaling
exponent: under SP they decay as O(n^-1/2), under muP they stay O(1).

Outputs (in --output-dir):
    <job>_dense0_activation_deltas.png   Fig. 3.1 (one panel per parametrization)
    <job>.png                            mean |activation| per layer over training
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import sys
repo_root = Path(__file__).resolve().parent.parent
visualizations_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root) + "/src")
sys.path.insert(0, str(visualizations_root))
from data import load_dataset

_CACHE_DIR = Path(".cache/matplotlib")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR.resolve()))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

from run_utils import (
    load_yaml_as_dict,
    _build_combined_legend,
    _label_from_log_path,
    _prepare_color_map,
    _scheme_from_path,
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DENSE0_DELTA_COUNT = 3
DENSE0_DELTA_BATCH_SIZE = 2048


@dataclass
class Dense0DeltaSeries:
    scheme: str
    label: str
    width: float
    training_steps: Tuple[float, ...]
    preactivation_deltas: np.ndarray
    postactivation_deltas: np.ndarray
    save_loss_frequency: Any


def collect_files_with_ending(directory: Path, ending: str) -> List[Path]:
    matches: List[Path] = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ending):
                matches.append(Path(root) / filename)
    return matches


def extract_after_char(s: str, start: str, stop: str) -> str:
    return s[s.rfind(start) + 1 : s.rfind(stop)]


def load_training_log(log_path: Path) -> Dict[str, object]:
    with log_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_evaluation_inputs(simulation_info: Dict[str, Any]) -> np.ndarray:
    training_info = simulation_info.get("training", {})
    training_data = training_info.get("training_data", {})
    if not isinstance(training_info, dict) or not isinstance(training_data, dict):
        raise ValueError("simulation_config.yaml must contain training.training_data.")

    task_path = training_data.get("task")
    if task_path is None:
        raise ValueError("simulation_config.yaml is missing training.training_data.task.")

    test_split = float(training_data.get("test_split", 0.2))
    # Datasets without a predefined split are split with seed 0 during training as well.
    _, _, x_test, _, _ = load_dataset(task_path, test_split, rng=np.random.default_rng(seed=0))

    x_test = np.asarray(x_test, dtype=np.float32)
    if x_test.ndim > 2:
        x_test = x_test.reshape(x_test.shape[0], -1)
    if x_test.ndim != 2:
        raise ValueError(f"Expected 2D evaluation inputs, got shape {x_test.shape}.")
    return x_test


def _numpy_activation(name: str, values: np.ndarray) -> np.ndarray:
    activation = name.lower()
    if activation == "identity":
        return values
    if activation == "relu":
        return np.maximum(values, 0.0)
    if activation == "tanh":
        return np.tanh(values)
    if activation == "sigmoid":
        clipped = np.clip(values, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    if activation == "gelu":
        return 0.5 * values * (
            1.0
            + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * values**3))
        )
    raise ValueError(f"Unsupported Dense_0 activation {name!r}.")


def _width_key(label: str) -> Sequence[int]:
    direct_split: List[int] = []
    try:
        direct_split = [int(part) for part in label.split("x")]
    except ValueError:
        direct_split = []
    if direct_split:
        return direct_split

    fallback = [int(match) for match in re.findall(r"\d+", label)]
    return fallback if fallback else [10**9]


def _hidden_width_label(label: str) -> str:
    widths = _width_key(label)
    if widths and widths[0] != 10**9:
        return str(widths[0])
    return label


def _extract_layer_activations(log_path: Path) -> np.ndarray:
    log_data = load_training_log(log_path)
    history = log_data.get("final_metrics", {}).get("history")
    if not history:
        history = log_data.get("history", {})
    activations = history.get("layer_activations")
    if activations is None:
        raise ValueError(f"No layer_activations history found in {log_path}")

    activations_array = np.asarray(activations, dtype=float)
    if activations_array.ndim != 2:
        raise ValueError(
            f"Expected layer_activations to be a 2D array, got shape {activations_array.shape}"
        )
    return activations_array


def _gather_activation_histories(log_dir: Path) -> Dict[str, np.ndarray]:
    log_files = sorted(collect_files_with_ending(log_dir, "training_log.json"))
    if not log_files:
        raise FileNotFoundError(f"No training logs found in {log_dir}")

    width_to_history: Dict[str, np.ndarray] = {}
    for log_path in log_files:
        width_label = extract_after_char(str(log_path), "-", "/")
        activations = _extract_layer_activations(log_path)
        width_to_history[width_label] = activations

    return dict(sorted(width_to_history.items(), key=lambda item: tuple(_width_key(item[0]))))


def _load_simulation_config_for_log(log_path: Path) -> Dict[str, Any]:
    simulation_config_path = log_path.parent / "simulation_config.yaml"
    if not simulation_config_path.exists():
        raise FileNotFoundError(f"Missing simulation_config.yaml next to {log_path}")
    return load_yaml_as_dict(simulation_config_path)


def _dense0_width_from_config(simulation_info: Dict[str, Any], label: str) -> float:
    network_info = simulation_info.get("network", {})
    nodes_per_layer = network_info.get("nodes_per_layer", {})
    if isinstance(nodes_per_layer, dict) and "Dense_0" in nodes_per_layer:
        return float(nodes_per_layer["Dense_0"])

    widths = _width_key(label)
    if widths:
        return float(widths[0])
    raise ValueError(f"Could not determine Dense_0 width for {label}.")


def _dense0_activation_name(simulation_info: Dict[str, Any]) -> str:
    network_info = simulation_info.get("network", {})
    activations = network_info.get("activations_per_layer", {})
    if not isinstance(activations, dict) or "Dense_0" not in activations:
        raise ValueError("Missing network.activations_per_layer.Dense_0 in simulation_config.yaml.")
    return str(activations["Dense_0"])


def _format_training_step(training_step: float) -> str:
    if float(training_step).is_integer():
        return str(int(training_step))
    return f"{training_step:g}"


def _dense0_delta_saved_indices(
    save_loss_frequency: Any,
    num_history_entries: int,
    num_weight_snapshots: int,
) -> List[Tuple[int, float]]:
    max_saved_index = min(
        DEFAULT_DENSE0_DELTA_COUNT,
        int(num_history_entries),
        int(num_weight_snapshots) - 1,
    )
    if max_saved_index < 1:
        raise ValueError(
            "Need at least one saved Dense_0 activation-delta interval in the logs."
        )

    if save_loss_frequency == "epoch":
        return [
            (saved_index, float(saved_index))
            for saved_index in range(1, max_saved_index + 1)
        ]

    save_every = float(save_loss_frequency)
    if save_every <= 0.0:
        raise ValueError(f"save_loss_frequency must be positive, got {save_loss_frequency}.")

    return [
        (saved_index, float(saved_index) * save_every)
        for saved_index in range(1, max_saved_index + 1)
    ]


FIGURE_DIR = Path("figures/analyze_acts")


def _plot_file_path(log_dir: Path, outfile: Path | None, suffix: str | None = None) -> Path:
    prefix = FIGURE_DIR
    if outfile:
        output = Path(outfile)
        if suffix:
            extension = output.suffix or ".png"
            output = output.with_name(f"{output.stem}_{suffix}{extension}")
        return prefix / output

    if suffix:
        return prefix / f"{log_dir.name}_{suffix}.png"
    return prefix / str(log_dir.name)


def _mean_abs_dense0_deltas(
    x_eval: np.ndarray,
    kernel_previous: np.ndarray,
    kernel_current: np.ndarray,
    activation_name: str,
    batch_size: int = DENSE0_DELTA_BATCH_SIZE,
) -> Tuple[float, float]:
    if kernel_previous.shape != kernel_current.shape:
        raise ValueError(
            "Dense_0 kernels for consecutive saved steps must have the same shape, "
            f"got {kernel_previous.shape} and {kernel_current.shape}."
        )
    if x_eval.shape[1] != kernel_previous.shape[0]:
        raise ValueError(
            "Evaluation input dimension does not match Dense_0 fan-in: "
            f"{x_eval.shape[1]} vs {kernel_previous.shape[0]}."
        )

    preactivation_abs_sum = 0.0
    postactivation_abs_sum = 0.0
    coordinate_count = 0

    kernel_previous = np.asarray(kernel_previous, dtype=np.float32)
    kernel_current = np.asarray(kernel_current, dtype=np.float32)

    for start in range(0, x_eval.shape[0], batch_size):
        stop = min(start + batch_size, x_eval.shape[0])
        x_batch = x_eval[start:stop]
        preactivation_previous = x_batch @ kernel_previous
        preactivation_current = x_batch @ kernel_current
        preactivation_delta = preactivation_current - preactivation_previous

        postactivation_previous = _numpy_activation(activation_name, preactivation_previous)
        postactivation_current = _numpy_activation(activation_name, preactivation_current)
        postactivation_delta = postactivation_current - postactivation_previous

        preactivation_abs_sum += float(
            np.sum(np.abs(preactivation_delta), dtype=np.float64)
        )
        postactivation_abs_sum += float(
            np.sum(np.abs(postactivation_delta), dtype=np.float64)
        )
        coordinate_count += int(preactivation_delta.size)

    if coordinate_count <= 0:
        raise ValueError("Cannot compute Dense_0 activation deltas from an empty evaluation set.")

    return (
        preactivation_abs_sum / coordinate_count,
        postactivation_abs_sum / coordinate_count,
    )


def _compute_dense0_delta_series(
    log_path: Path,
    x_eval: np.ndarray,
    simulation_info: Dict[str, Any],
) -> Dense0DeltaSeries:
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Dense_0 activation-delta plotting requires zarr. Run this script via "
            "`pixi run python visualizations/analyze_acts.py ...`."
        ) from exc

    weights_path = log_path.parent / "weights.zarr"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights.zarr next to {log_path}")

    history = _extract_layer_activations(log_path)
    root = zarr.open_group(weights_path, mode="r")
    try:
        dense0_kernel_history = root["Dense_0"]["kernel"]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"Missing Dense_0/kernel in {weights_path}") from exc

    training_info = simulation_info.get("training", {})
    selected_indices = _dense0_delta_saved_indices(
        training_info.get("save_loss_frequency"),
        int(history.shape[0]),
        int(dense0_kernel_history.shape[0]),
    )

    activation_name = _dense0_activation_name(simulation_info)
    training_steps: List[float] = []
    preactivation_deltas: List[float] = []
    postactivation_deltas: List[float] = []
    for saved_index, training_step in selected_indices:
        pre_delta, post_delta = _mean_abs_dense0_deltas(
            x_eval,
            np.asarray(dense0_kernel_history[saved_index - 1], dtype=np.float32),
            np.asarray(dense0_kernel_history[saved_index], dtype=np.float32),
            activation_name,
        )
        training_steps.append(training_step)
        preactivation_deltas.append(pre_delta)
        postactivation_deltas.append(post_delta)

    label = _label_from_log_path(log_path)
    return Dense0DeltaSeries(
        scheme=_scheme_from_path(log_path),
        label=label,
        width=_dense0_width_from_config(simulation_info, label),
        training_steps=tuple(training_steps),
        preactivation_deltas=np.asarray(preactivation_deltas, dtype=float),
        postactivation_deltas=np.asarray(postactivation_deltas, dtype=float),
        save_loss_frequency=training_info.get("save_loss_frequency"),
    )


def _gather_dense0_delta_series(log_dir: Path) -> List[Dense0DeltaSeries]:
    log_files = sorted(
        collect_files_with_ending(log_dir, "training_log.json"),
        key=lambda path: tuple(_width_key(_label_from_log_path(path))),
    )
    if not log_files:
        raise FileNotFoundError(f"No training logs found in {log_dir}")

    simulation_by_log = {
        log_path: _load_simulation_config_for_log(log_path)
        for log_path in log_files
    }
    task_paths = {
        str(
            simulation_info.get("training", {})
            .get("training_data", {})
            .get("task", "")
        )
        for simulation_info in simulation_by_log.values()
    }
    if len(task_paths) != 1:
        raise ValueError(
            "Dense_0 activation-delta plotting expects all runs to use the same task, "
            f"got {sorted(task_paths)}."
        )

    first_simulation_info = simulation_by_log[log_files[0]]
    x_eval = _load_evaluation_inputs(first_simulation_info)

    series_list: List[Dense0DeltaSeries] = []
    skipped: List[str] = []
    for log_path in log_files:
        try:
            series_list.append(
                _compute_dense0_delta_series(
                    log_path,
                    x_eval,
                    simulation_by_log[log_path],
                )
            )
        except FileNotFoundError as exc:
            skipped.append(str(exc))

    if skipped:
        print(
            "Warning: skipped Dense_0 activation-delta series without saved weights:\n"
            + "\n".join(f"  - {message}" for message in skipped)
        )
    if not series_list:
        raise ValueError(
            "No Dense_0 activation-delta series could be computed. "
            "Make sure weight_monitoring was enabled and weights.zarr exists."
        )

    return sorted(series_list, key=lambda series: tuple(_width_key(series.label)))


def _series_color_lookup(series_list: Sequence[Dense0DeltaSeries]) -> Dict[Tuple[str, str], Any]:
    scheme_groups = {
        "standard": [series for series in series_list if series.scheme == "standard"],
        "muP": [series for series in series_list if series.scheme == "muP"],
    }
    color_map = {
        "standard": _prepare_color_map(len(scheme_groups["standard"]), plt.cm.autumn),
        "muP": _prepare_color_map(len(scheme_groups["muP"]), plt.cm.winter),
    }
    color_indices = {"standard": 0, "muP": 0}
    colors: Dict[Tuple[str, str], Any] = {}
    for series in series_list:
        color = color_map[series.scheme][color_indices[series.scheme]]
        color_indices[series.scheme] += 1
        colors[(series.scheme, series.label)] = color
    return colors


def _plot_positive_point(ax: Any, width: float, value: float, color: Any, label: str | None) -> Any | None:
    if width <= 0.0 or value <= 0.0 or not np.isfinite(value):
        return None
    return ax.scatter([width], [value], color=[color], s=42, label=label, zorder=3)


def _connect_scheme_points(
    ax: Any,
    series_list: Sequence[Dense0DeltaSeries],
    value_getter: Any,
    color: Any,
) -> None:
    points = sorted(
        (
            (series.width, float(value_getter(series)))
            for series in series_list
            if series.width > 0.0
            and float(value_getter(series)) > 0.0
            and np.isfinite(float(value_getter(series)))
        ),
        key=lambda point: point[0],
    )
    if len(points) < 2:
        return
    widths = [point[0] for point in points]
    values = [point[1] for point in points]
    ax.plot(widths, values, color=color, alpha=0.35, linewidth=1.0, zorder=2)


def _plot_log2_scaling_fit(
    ax: Any,
    points: Sequence[Tuple[float, float]],
) -> None:
    positive_points = sorted(
        (
            (float(width), float(value))
            for width, value in points
            if width > 0.0 and value > 0.0 and np.isfinite(width) and np.isfinite(value)
        ),
        key=lambda point: point[0],
    )
    if len(positive_points) < 2:
        (line,) = ax.plot(
            [],
            [],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=r"$a$ unavailable",
        )
        ax.legend(handles=[line], loc="upper right", frameon=True, fontsize=15, handlelength=2.0)
        return

    widths = np.asarray([point[0] for point in positive_points], dtype=float)
    values = np.asarray([point[1] for point in positive_points], dtype=float)
    slope, intercept = np.polyfit(np.log2(widths), np.log2(values), deg=1)

    fit_widths = 2.0 ** np.linspace(np.log2(widths.min()), np.log2(widths.max()), num=200)
    fit_values = (2.0**intercept) * fit_widths**slope
    (line,) = ax.plot(
        fit_widths,
        fit_values,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=rf"$a={slope:.3g}$",
        zorder=4,
    )
    ax.legend(handles=[line], loc="upper right", frameon=True, fontsize=15, handlelength=2.0)


def plot_dense0_activation_deltas(
    log_dir: Path,
    outfile: Path | None,
    y_exp_limits: Tuple[float, float] | None = None,
) -> None:
    series_list = _gather_dense0_delta_series(log_dir)
    available_steps_by_series = [set(series.training_steps) for series in series_list]
    training_steps = [
        step
        for step in series_list[0].training_steps[:DEFAULT_DENSE0_DELTA_COUNT]
        if all(step in available_steps for available_steps in available_steps_by_series)
    ]
    num_deltas = len(training_steps)
    if num_deltas < 1:
        raise ValueError("No Dense_0 activation deltas are available to plot.")
    if num_deltas < min(DEFAULT_DENSE0_DELTA_COUNT, len(series_list[0].training_steps)):
        available = ", ".join(_format_training_step(step) for step in training_steps)
        print(
            "Warning: not all first saved Dense_0 activation-delta intervals are "
            f"available in every run. Plotting {available}."
        )

    colors = _series_color_lookup(series_list)
    legend_entries: Dict[str, List[Tuple[Any, str]]] = {"standard": [], "muP": []}
    fig, axes = plt.subplots(2, num_deltas, figsize=(5.5 * num_deltas, 8), squeeze=False)
    widths = sorted({series.width for series in series_list if series.width > 0.0})
    y_limits = None
    if y_exp_limits is not None:
        min_exp, max_exp = y_exp_limits
        if min_exp >= max_exp:
            raise ValueError(
                "Dense_0 activation-delta y-limit exponents must satisfy "
                f"min_exp < max_exp, got {min_exp} >= {max_exp}."
            )
        y_limits = (2.0**min_exp, 2.0**max_exp)

    step_index_by_series = {
        (series.scheme, series.label): {
            step: index for index, step in enumerate(series.training_steps)
        }
        for series in series_list
    }

    for delta_index, training_step in enumerate(training_steps):
        for row_index, quantity_name in enumerate(("preactivation", "postactivation")):
            ax = axes[row_index, delta_index]
            for scheme, line_color in (
                ("standard", plt.cm.autumn(0.65)),
                ("muP", plt.cm.winter(0.65)),
            ):
                scheme_series = [series for series in series_list if series.scheme == scheme]
                if not scheme_series:
                    continue
                _connect_scheme_points(
                    ax,
                    scheme_series,
                    lambda series, step=training_step, name=quantity_name: (
                        series.preactivation_deltas[
                            step_index_by_series[(series.scheme, series.label)][step]
                        ]
                        if name == "preactivation"
                        else series.postactivation_deltas[
                            step_index_by_series[(series.scheme, series.label)][step]
                        ]
                    ),
                    line_color,
                )

            fit_points: List[Tuple[float, float]] = []
            for series in series_list:
                series_delta_index = step_index_by_series[(series.scheme, series.label)][training_step]
                value = (
                    series.preactivation_deltas[series_delta_index]
                    if quantity_name == "preactivation"
                    else series.postactivation_deltas[series_delta_index]
                )
                color = colors[(series.scheme, series.label)]
                label = _hidden_width_label(series.label) if delta_index == 0 and row_index == 0 else None
                handle = _plot_positive_point(ax, series.width, float(value), color, label)
                if handle is not None and label is not None:
                    legend_entries[series.scheme].append((handle, label))
                fit_points.append((series.width, float(value)))

            _plot_log2_scaling_fit(ax, fit_points)
            if row_index == 0:
                ax.set_title(f"t={_format_training_step(training_step)}", fontsize=20)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            ax.grid(True, which="both", alpha=0.3)
            ax.tick_params(axis="both", labelsize=15)
            if widths:
                ax.set_xticks(widths)
                ax.set_xticklabels([str(int(width)) for width in widths], rotation=45, ha="right")
            if row_index == 1:
                ax.set_xlabel(r"$n$ $[\log_2]$", fontsize=20)
            if delta_index == 0:
                if quantity_name == "preactivation":
                    ax.set_ylabel(
                        r"$\left\langle \overline{\Delta h_t} "
                        r"\right\rangle_{\mathcal{D}_{te}}$",
                        fontsize=20,
                    )
                else:
                    ax.set_ylabel(
                        r"$\left\langle \overline{\Delta z_t} "
                        r"\right\rangle_{\mathcal{D}_{te}}$",
                        fontsize=20,
                    )

    combined_handles, combined_labels, header_indices = _build_combined_legend(
        legend_entries,
        {"standard": "width (n) - SP", "muP": r"width (n) - $\mu$P"},
    )
    if combined_handles:
        legend = fig.legend(
            combined_handles,
            combined_labels,
            loc="lower center",
            ncol=min(len(combined_labels), 5),
            frameon=True,
            fontsize=15,
        )
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        for index in header_indices:
            legend.get_texts()[index].set_fontweight("bold")

    fig.suptitle(
        r"Dashed black fit: $\log_2 y = a \cdot \log_2 n + b$"
        + "\n"
        + r"$y$ represents the pre-/postactivation and $a$ the corresponding "
        + r"scaling exponent fitted from data"
        + "\n"
        + r"$\rightarrow$ $y$ scales as: $y \sim n^{a}$ with the network width $n$",
        fontsize=20,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])

    file_path = _plot_file_path(log_dir, outfile, "dense0_activation_deltas")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Dense_0 activation-delta plot to {file_path}")


def plot_layer_activations(
    log_dir: Path,
    outfile: Path | None,
    dense0_delta_y_exp_limits: Tuple[float, float] | None = None,
) -> None:
    histories = _gather_activation_histories(log_dir)
    if not histories:
        raise ValueError(f"No activation histories available in {log_dir}")
    
    simluation_config_path = sorted(collect_files_with_ending(log_dir, "simulation_config.yaml"))[0]
    simulation_info = load_yaml_as_dict(simluation_config_path)
    dataset_info = simulation_info["training"]
    network_info = simulation_info["network"]
    save_loss_frequency = dataset_info.get("save_loss_frequency")

    epoch_counts = {width: history.shape[0] for width, history in histories.items()}
    min_epochs = min(epoch_counts.values())
    if min_epochs < 1:
        raise ValueError("Activation histories must contain at least one epoch.")

    num_layers_set = {history.shape[1] for history in histories.values()}
    if len(num_layers_set) != 1:
        raise ValueError(
            f"Inconsistent layer counts across widths: {sorted(num_layers_set)}"
        )
    num_layers = num_layers_set.pop()

    width_items = list(histories.items())
    parsed_widths: List[List[int]] = []
    for width_label, history in width_items:
        widths = [int(part) for part in re.findall(r"\d+", width_label)]
        if len(widths) < num_layers:
            raise ValueError(
                f"Width specification '{width_label}' has fewer entries than activation layers ({num_layers})."
            )
        parsed_widths.append(widths)

    hidden_count = num_layers
    if hidden_count <= 0:
        raise ValueError("Layer activation history does not include hidden layers to plot.")

    num_epochs = min(10, min_epochs)
    if min_epochs < 10:
        print(
            f"Warning: only {min_epochs} epochs available; plotting the first {num_epochs} epochs."
        )
    if min_epochs > 10:
        print(
            f"Note: {min_epochs} epochs available; plotting the first {num_epochs} epochs."
        )

    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    axes_flat = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0, 1, hidden_count))
    unique_width_values = sorted(
        {widths[idx] for widths in parsed_widths for idx in range(hidden_count)}
    )

    legend_handles: List[object] = []
    legend_labels: List[str] = []
    for epoch_idx in range(len(axes_flat)):
        ax = axes_flat[epoch_idx]
        if epoch_idx >= num_epochs:
            ax.axis("off")
            continue

        for layer_idx in range(hidden_count):
            layer_widths = [widths[0] for widths in parsed_widths]
            layer_values = [history[epoch_idx, layer_idx] for _, history in width_items]
            (line,) = ax.semilogx(
                layer_widths,
                layer_values,
                base=2,
                marker="o",
                color=colors[layer_idx],
                label=f"Hidden Layer {layer_idx + 1}",
            )
            if epoch_idx == 0 and layer_idx < len(colors):
                legend_handles.append(line)
                legend_labels.append(f"Dense Layer {layer_idx}")

        ax.set_title(f"Optimization Step {(epoch_idx + 1)*save_loss_frequency}", fontsize=15)
        if epoch_idx >= 5:
            ax.set_xlabel(r"Hidden Layer Width $[log_2]$", fontsize=15)
        if epoch_idx % 5 == 0:
            ax.set_ylabel("Logit Norms", fontsize=15)
        ax.set_xticks(layer_widths)
        ax.set_xticklabels([str(value) for value in layer_widths])
    
        ax.tick_params(axis="y", which="both", labelleft=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Width (n) " + r"$[log_{2}]$", fontsize=15)

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels), frameon=False, fontsize=15)

    fig.suptitle(f"Logit Norms for each Network Layer vs. Network Width in the Hidden Layers (Shown for Multiple Optimization Steps)", fontsize=15)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    file_path = _plot_file_path(log_dir, outfile)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {file_path}")

    plot_dense0_activation_deltas(log_dir, outfile, dense0_delta_y_exp_limits)


def main() -> None:
    global FIGURE_DIR
    parser = argparse.ArgumentParser(
        description=(
            "Visualize layer activation averages across widths (thesis Fig. 3.1: changes of the "
            "hidden-layer pre-/postactivations between saved training steps)."
        )
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Job directory containing scheme-*_widths-* run folders with training_log.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Directory in which the figures are saved (default: figures/analyze_acts).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file name of the generated plot (relative to --output-dir).",
    )
    parser.add_argument(
        "--dense0-delta-y-exp-limits",
        nargs=2,
        type=float,
        metavar=("MIN_EXP", "MAX_EXP"),
        help=(
            "Optional fixed y-axis limits for the Dense_0 activation-delta plot. "
            "The limits are set to 2**MIN_EXP and 2**MAX_EXP."
        ),
    )

    args = parser.parse_args()
    FIGURE_DIR = args.output_dir
    dense0_delta_y_exp_limits = (
        tuple(args.dense0_delta_y_exp_limits)
        if args.dense0_delta_y_exp_limits is not None
        else None
    )
    plot_layer_activations(
        args.log_dir,
        args.output,
        dense0_delta_y_exp_limits,
    )


if __name__ == "__main__":
    main()
