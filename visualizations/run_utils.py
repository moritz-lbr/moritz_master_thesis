"""Helpers for loading training runs and their weight-update metrics.

Extracted from plot_weight_update_similarity.py of the original working repository;
only the functions used by plot_update_norms.py and analyze_acts.py are kept.
A run is a scheme-<muP|standard>_widths-<n>x<p> folder written by
src/run_experiment.py; weight-update metrics are read from its weight_metrics/
folder (see training_analysis/compute_metrics.py).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_yaml_as_dict(file_path: Path) -> Dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        return {} if data is None else {"value": data}
    return data


def collect_files_with_ending(directory: Path, ending: str) -> List[Path]:
    matches: List[Path] = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ending):
                matches.append(Path(root) / filename)
    return matches


def load_training_log(log_path: Path) -> Dict[str, Any]:
    with log_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _scheme_from_path(log_path: Path) -> str:
    directory_name = log_path.parent.name
    return "standard" if "standard" in directory_name else "muP"


def _label_from_log_path(log_path: Path) -> str:
    return log_path.parent.name.rsplit("-", maxsplit=1)[-1]


def _prepare_color_map(count: int, cmap: Any) -> Iterable[Any]:
    steps = max(count, 1)
    return cmap(np.linspace(0, 1, steps))


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


def _sorted_entries(entries: List[Tuple[Any, str]]) -> List[Tuple[Any, str]]:
    return sorted(entries, key=lambda item: tuple(_width_key(item[1])))


def _build_combined_legend(
    legend_entries: Dict[str, List[Tuple[Any, str]]],
    legend_titles: Dict[str, str],
) -> Tuple[List[Any], List[str], List[int]]:
    handles: List[Any] = []
    labels: List[str] = []
    header_indices: List[int] = []

    for scheme in ("standard", "muP"):
        sorted_linear = _sorted_entries(legend_entries[scheme])
        if not sorted_linear:
            continue
        header = plt.Line2D([], [], linestyle="", marker="", linewidth=0)
        handles.append(header)
        labels.append(legend_titles[scheme])
        header_indices.append(len(labels) - 1)
        for handle, label in sorted_linear:
            handles.append(handle)
            labels.append(label)

    return handles, labels, header_indices


def _normalize_save_loss_frequency(save_loss_frequency: Any) -> float:
    if save_loss_frequency == "epoch":
        return 1.0
    return float(save_loss_frequency)


def _build_x_axis(
    losses: np.ndarray,
    dataset_info: Dict[str, Any],
    network_info: Dict[str, Any],
    compute_flag: bool,
) -> Tuple[np.ndarray, str, float]:
    save_loss_frequency = _normalize_save_loss_frequency(
        dataset_info.get("save_loss_frequency")
    )

    if compute_flag:
        parameters = float(network_info.get("total_params"))
        batch_size = float(dataset_info.get("batch_size"))
        x_axis = (
            np.arange(len(losses), dtype=float)
            * save_loss_frequency
            * batch_size
            * parameters
        )
        x_label = r"Training Compute $c_{i}$ [log]"
    else:
        x_axis = np.arange(len(losses), dtype=float) * save_loss_frequency
        x_label = r"Training Steps $t_{i}$ [log]"

    return x_axis, x_label, save_loss_frequency


def _load_final_metrics_block(log_path: Path) -> Dict[str, Any]:
    log_data = load_training_log(log_path)
    final_metrics = log_data.get("final_metrics")
    if final_metrics is None:
        final_metrics = log_data.get("avg_final_metrics")
    if not isinstance(final_metrics, dict):
        raise ValueError(f"No final metrics found in {log_path}")
    return final_metrics


def _load_training_metrics_block(weight_metrics_path: Path) -> Dict[str, Any]:
    weight_metrics = load_training_log(weight_metrics_path)
    training_metrics = weight_metrics.get("training_metrics")
    if training_metrics is None:
        training_metrics = weight_metrics.get("avg_training_metrics")
    if not isinstance(training_metrics, dict):
        raise ValueError(f"No training metrics found in {weight_metrics_path}")
    return training_metrics


def _zeros_or_array(values: Any, reference: np.ndarray) -> np.ndarray:
    if values is None:
        return np.zeros_like(reference, dtype=float)
    return np.asarray(values, dtype=float)


def _find_weight_metrics_path(run_dir: Path, layer: str) -> Path:
    for directory_name in ("weight_metrics", "avg_final_weight_metrics"):
        candidate = run_dir / directory_name / f"{layer}.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing weight metrics file for {layer} in {run_dir}")


def _has_uncertainty(std_values: np.ndarray) -> bool:
    return bool(np.any(np.nan_to_num(std_values, nan=0.0) != 0.0))


def _plot_with_uncertainty(
    ax: Any,
    x_axis: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    color: Any,
    label: str | None = None,
    shade: bool,
    min_lower: float | None = None,
) -> Any:
    line, = ax.plot(x_axis, mean, color=color, label=label)
    if shade:
        lower = mean - std
        upper = mean + std
        if min_lower is not None:
            lower = np.clip(lower, min_lower, None)
        ax.fill_between(x_axis, lower, upper, color=color, alpha=0.18, linewidth=0)
    return line


def _format_save_loss_frequency(save_loss_frequency: float) -> str:
    if float(save_loss_frequency).is_integer():
        return str(int(save_loss_frequency))
    return str(save_loss_frequency)


def _loss_plot_label(loss_type: str) -> str:
    return "Train Loss" if loss_type == "train" else "Test Loss"


@dataclass
class PlotSeries:
    scheme: str
    label: str
    x_axis: np.ndarray
    losses_mean: np.ndarray
    losses_std: np.ndarray
    similarities_mean: np.ndarray
    similarities_std: np.ndarray
    step_norms_mean: np.ndarray
    step_norms_std: np.ndarray
    task_name: str
    dataset_info: Dict[str, Any]
    network_info: Dict[str, Any]
    save_loss_frequency: float
    num_averaged_runs: int
    has_uncertainty: bool


def _collect_plot_series(
    job_dir: Path,
    log_path: Path,
    layer: str,
    compute_flag: bool,
    loss_type: str,
) -> PlotSeries:
    simulation_config_path = log_path.parent / "simulation_config.yaml"
    if not simulation_config_path.exists():
        raise FileNotFoundError(f"Missing simulation_config.yaml next to {log_path}")

    simulation_info = load_yaml_as_dict(simulation_config_path)
    dataset_info = simulation_info.get("training")
    network_info = simulation_info.get("network")
    if not isinstance(dataset_info, dict) or not isinstance(network_info, dict):
        raise ValueError(f"Incomplete simulation config in {simulation_config_path}")

    final_metrics = _load_final_metrics_block(log_path)
    history = final_metrics.get("history", {})
    if not isinstance(history, dict):
        raise ValueError(f"Missing history in {log_path}")

    loss_key = f"{loss_type}_loss"
    loss_std_key = f"{loss_key}_std"
    losses = history.get(loss_key)
    if not losses:
        raise ValueError(f"No {loss_key} history found in {log_path}")

    losses_mean = np.asarray(losses, dtype=float)
    losses_std = _zeros_or_array(history.get(loss_std_key), losses_mean)

    weight_metrics_path = _find_weight_metrics_path(log_path.parent, layer)
    training_metrics = _load_training_metrics_block(weight_metrics_path)

    step_norms = training_metrics.get("step_norms")
    similarities = training_metrics.get("similarities")
    if not step_norms:
        raise ValueError(f"No step norms history found in {weight_metrics_path}")
    if not similarities:
        raise ValueError(f"No similarities history found in {weight_metrics_path}")

    step_norms_mean = np.asarray(step_norms, dtype=float)
    step_norms_std = _zeros_or_array(training_metrics.get("step_norms_std"), step_norms_mean)
    similarities_mean = np.asarray(similarities, dtype=float)
    similarities_std = _zeros_or_array(
        training_metrics.get("similarities_std"), similarities_mean
    )

    x_axis, _, save_loss_frequency = _build_x_axis(
        losses_mean, dataset_info, network_info, compute_flag
    )
    task_path = dataset_info.get("training_data", {}).get("task")
    task_name = Path(task_path).name if task_path else job_dir.name
    averaging_info = simulation_info.get("averaging", {})
    num_averaged_runs = int(averaging_info.get("num_files_averaged", 1))

    return PlotSeries(
        scheme=_scheme_from_path(log_path),
        label=_label_from_log_path(log_path),
        x_axis=x_axis,
        losses_mean=losses_mean,
        losses_std=losses_std,
        similarities_mean=similarities_mean,
        similarities_std=similarities_std,
        step_norms_mean=step_norms_mean,
        step_norms_std=step_norms_std,
        task_name=task_name,
        dataset_info=dataset_info,
        network_info=network_info,
        save_loss_frequency=save_loss_frequency,
        num_averaged_runs=num_averaged_runs,
        has_uncertainty=any(
            (
                _has_uncertainty(losses_std),
                _has_uncertainty(similarities_std),
                _has_uncertainty(step_norms_std),
            )
        ),
    )
