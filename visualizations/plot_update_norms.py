"""Plot weight-update norms for Dense_0 and Dense_1."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_weight_update_similarity import (
    PlotSeries,
    _build_combined_legend,
    _build_x_axis,
    _collect_plot_series,
    _format_save_loss_frequency,
    _has_uncertainty,
    _label_from_log_path,
    _loss_plot_label,
    _prepare_color_map,
    _plot_with_uncertainty,
    _width_key,
    collect_files_with_ending,
)


LAYERS = ("Dense_0", "Dense_1")
LOSS_TYPE = "test"
DEVIATION_EPSILON = 1e-12
DEVIATION_STD_WINDOW = 10
DEFAULT_SIGMA_START = 1.0
DEFAULT_SIGMA_STOP = 3.0
DEFAULT_NUM_SIGMAS = 10
SIGMA_THRESHOLDS = tuple(
    np.linspace(DEFAULT_SIGMA_START, DEFAULT_SIGMA_STOP, DEFAULT_NUM_SIGMAS)
)
SIGMA_THRESHOLD_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")


def _sigma_thresholds(
    sigma_start: float,
    sigma_stop: float,
    num_sigmas: int,
) -> Tuple[float, ...]:
    if num_sigmas < 1:
        raise ValueError(f"num_sigmas must be >= 1, got {num_sigmas}.")
    if sigma_stop < sigma_start:
        raise ValueError(
            f"sigma_stop must be >= sigma_start, got {sigma_stop} < {sigma_start}."
        )
    return tuple(float(value) for value in np.linspace(sigma_start, sigma_stop, num_sigmas))


def _threshold_key(threshold: float) -> str:
    return f"{threshold:g}_sigma"


def _threshold_marker(index: int) -> str:
    return SIGMA_THRESHOLD_MARKERS[index % len(SIGMA_THRESHOLD_MARKERS)]


def _threshold_color(index: int, count: int) -> Any:
    color_positions = np.linspace(0.35, 0.95, max(count, 1))
    return plt.cm.Oranges(color_positions[index])


def _thresholds_from_breakdown_data(breakdown_data: Dict[str, Any]) -> Tuple[float, ...]:
    return tuple(float(threshold) for threshold in breakdown_data.get("thresholds", SIGMA_THRESHOLDS))


def _top_ylabel(layer: str) -> str:
    layer_index = layer[-1]
    return rf"$\|u_{{t,\tau}}^{{{layer_index}}}\|^2 (mn)\ [\log]$"


def _bottom_ylabel(layer: str) -> str:
    layer_index = layer[-1]
    return rf"$\frac{{\|u_{{t,\tau}}^{{{layer_index}}}\|^2 (mn)}}{{c^{{{layer_index}}}(m)}}\ [\log]$"


def _deviation_ylabel(layer: str) -> str:
    layer_index = layer[-1]
    return (
        r"$\frac{\left|\log\left(\frac{\tilde{O}_N^{\,"
        + layer_index
        + r"}(t_i)+\epsilon}{\tilde{O}_{N_{\infty}}^{\,"
        + layer_index
        + r"}(t_i)+\epsilon}\right)\right|}{\sqrt{\sigma_N^2+\sigma_{N_{\infty}}^2}},\ \epsilon=10^{-12}$"
    )


def _dense0_width_ratio(network_info: Dict[str, Any]) -> float:
    nodes_per_layer = network_info.get("nodes_per_layer", {})
    base_widths = network_info.get("base_layer_width") or network_info.get(
        "base_layer_widths"
    )
    if not isinstance(nodes_per_layer, dict) or "Dense_0" not in nodes_per_layer:
        raise ValueError("Missing network.nodes_per_layer.Dense_0 in simulation config.")

    if isinstance(base_widths, dict):
        base_width = base_widths.get("Dense_0")
    else:
        base_width = base_widths
    if base_width is None:
        raise ValueError("Missing Dense_0 base layer width in simulation config.")

    ratio = float(nodes_per_layer["Dense_0"]) / float(base_width)
    if ratio <= 0:
        raise ValueError("Dense_0 width/base-width ratio must be positive.")
    return ratio


def _scaled_norms(series: PlotSeries, layer: str) -> Tuple[np.ndarray, np.ndarray]:
    if series.scheme != "muP":
        raise ValueError(f"Scaling constants are defined only for muP runs, got {series.scheme}.")
    ratio = _dense0_width_ratio(series.network_info)
    scale = 1.0 / ratio if layer == "Dense_0" else ratio
    return series.step_norms_mean * scale, series.step_norms_std * scale


def _deviation_norms(series: PlotSeries, layer: str) -> np.ndarray:
    if series.scheme == "muP":
        return _scaled_norms(series, layer)[0]
    return series.step_norms_mean


def _dense0_width(series: PlotSeries) -> float:
    nodes_per_layer = series.network_info.get("nodes_per_layer", {})
    if not isinstance(nodes_per_layer, dict) or "Dense_0" not in nodes_per_layer:
        raise ValueError(f"Missing Dense_0 width for run {series.label}.")
    return float(nodes_per_layer["Dense_0"])


def _num_parameters(series: PlotSeries) -> float:
    parameters = series.network_info.get("total_params")
    if parameters is None:
        raise ValueError(f"Missing total_params for run {series.label}.")
    return float(parameters)


def _largest_width_series(series_list: List[PlotSeries]) -> PlotSeries:
    if not series_list:
        raise ValueError("Cannot select the largest run from an empty series list.")
    return max(
        series_list,
        key=lambda series: (_dense0_width(series), tuple(_width_key(series.label))),
    )


def _validate_window_width(name: str, window: int, *, minimum: int) -> None:
    if window < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {window}.")


def _slice_series(
    series: PlotSeries,
    start_idx: int | None,
    stop_idx: int | None,
) -> PlotSeries:
    if start_idx is None and stop_idx is None:
        return series

    time_slice = slice(start_idx, stop_idx)
    sliced = replace(
        series,
        x_axis=series.x_axis[time_slice],
        losses_mean=series.losses_mean[time_slice],
        losses_std=series.losses_std[time_slice],
        similarities_mean=series.similarities_mean[time_slice],
        similarities_std=series.similarities_std[time_slice],
        step_norms_mean=series.step_norms_mean[time_slice],
        step_norms_std=series.step_norms_std[time_slice],
    )
    if len(sliced.step_norms_mean) == 0:
        raise ValueError(
            f"Slice {start_idx}:{stop_idx} leaves no update-norm values for {series.label}."
        )
    return sliced


def _sliding_window_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return values.copy()
    if window > len(values):
        raise ValueError(
            f"Cannot average {len(values)} observable values with window {window}."
        )

    averaged = np.full(len(values) - window + 1, np.nan, dtype=float)
    for start in range(len(averaged)):
        window_values = values[start : start + window]
        window_values = window_values[np.isfinite(window_values)]
        if window_values.size:
            averaged[start] = float(np.mean(window_values))
    return averaged


def _sliding_window_x_axis(
    x_axis: np.ndarray,
    values_len: int,
    window: int,
) -> np.ndarray:
    x_axis = x_axis[:values_len]
    if window == 1:
        return x_axis.copy()
    if window > len(x_axis):
        raise ValueError(f"Cannot align {len(x_axis)} x-axis values with window {window}.")

    output_len = len(x_axis) - window + 1
    center_offset = window // 2
    return x_axis[center_offset : center_offset + output_len]


def _average_observables(series: PlotSeries, delta_t_avg: int) -> PlotSeries:
    if delta_t_avg == 1:
        return series

    step_x_axis = _sliding_window_x_axis(
        series.x_axis,
        len(series.step_norms_mean),
        delta_t_avg,
    )
    return replace(
        series,
        x_axis=step_x_axis,
        step_norms_mean=_sliding_window_mean(series.step_norms_mean, delta_t_avg),
        step_norms_std=_sliding_window_mean(series.step_norms_std, delta_t_avg),
        similarities_mean=_sliding_window_mean(series.similarities_mean, delta_t_avg),
        similarities_std=_sliding_window_mean(series.similarities_std, delta_t_avg),
    )


def _rolling_empirical_std(values: np.ndarray, window: int = DEVIATION_STD_WINDOW) -> np.ndarray:
    std_values = np.full(len(values), np.nan, dtype=float)
    for index in range(len(values)):
        window_values = values[max(0, index - window + 1) : index + 1]
        window_values = window_values[np.isfinite(window_values)]
        if len(window_values) >= 2:
            std_values[index] = float(np.std(window_values, ddof=1))
    return std_values


def _deviation_from_reference(
    series: PlotSeries,
    reference: PlotSeries,
    layer: str,
    delta_t_std: int,
) -> Tuple[np.ndarray, np.ndarray]:
    min_len = min(
        len(series.step_norms_mean),
        len(reference.step_norms_mean),
        len(series.x_axis),
    )
    x_axis = series.x_axis[:min_len]
    series_norms = _deviation_norms(series, layer)[:min_len]
    reference_norms = _deviation_norms(reference, layer)[:min_len]

    with np.errstate(divide="ignore", invalid="ignore"):
        log_series_norms = np.log(series_norms + DEVIATION_EPSILON)
        log_reference_norms = np.log(reference_norms + DEVIATION_EPSILON)
        deviation = np.abs(log_series_norms - log_reference_norms)
        combined_sigma = np.sqrt(
            _rolling_empirical_std(log_series_norms, delta_t_std) ** 2
            + _rolling_empirical_std(log_reference_norms, delta_t_std) ** 2
        )
        normalized_deviation = np.divide(
            deviation,
            combined_sigma,
            out=np.full(min_len, np.nan, dtype=float),
            where=combined_sigma > 0.0,
        )
    normalized_deviation[~np.isfinite(normalized_deviation)] = np.nan
    return x_axis, normalized_deviation


def _first_threshold_crossing(
    x_axis: np.ndarray,
    normalized_deviation: np.ndarray,
    threshold: float,
) -> float | None:
    valid = (
        np.isfinite(x_axis)
        & np.isfinite(normalized_deviation)
        & (x_axis > 0.0)
        & (normalized_deviation > threshold)
    )
    crossing_indices = np.flatnonzero(valid)
    if crossing_indices.size == 0:
        return None
    return float(x_axis[crossing_indices[0]])


def _breakdown_times(
    x_axis: np.ndarray,
    normalized_deviation: np.ndarray,
    thresholds: Tuple[float, ...],
) -> Dict[str, float | None]:
    return {
        _threshold_key(threshold): _first_threshold_crossing(
            x_axis,
            normalized_deviation,
            threshold,
        )
        for threshold in thresholds
    }


def _breakdown_record(
    series: PlotSeries,
    color: Any,
    x_axis: np.ndarray,
    normalized_deviation: np.ndarray,
    thresholds: Tuple[float, ...],
) -> Dict[str, Any]:
    color_array = np.asarray(color, dtype=float).ravel()
    return {
        "label": series.label,
        "scheme": series.scheme,
        "network_size": int(_dense0_width(series)),
        "parameters": int(_num_parameters(series)),
        "color": [float(value) for value in color_array],
        "break_down_times": _breakdown_times(x_axis, normalized_deviation, thresholds),
    }


def _fit_breakdown_scaling(
    records: List[Dict[str, Any]],
    threshold: float,
) -> Dict[str, float | int] | None:
    key = _threshold_key(threshold)
    xs = []
    ys = []
    for record in records:
        break_down_time = record["break_down_times"].get(key)
        parameters = record.get("parameters", record.get("network_size"))
        if break_down_time is None or parameters is None:
            continue
        if break_down_time <= 0.0 or parameters <= 0.0:
            continue
        xs.append(float(parameters))
        ys.append(float(break_down_time))

    if len(xs) < 2:
        return None

    log_x = np.log2(np.asarray(xs, dtype=float))
    log_y = np.log2(np.asarray(ys, dtype=float))
    scaling_exponent, offset = np.polyfit(log_x, log_y, deg=1)
    return {
        "offset": float(offset),
        "scaling_exponent": float(scaling_exponent),
        "num_points": len(xs),
        "parameter_min": float(min(xs)),
        "parameter_max": float(max(xs)),
    }


def _compute_breakdown_fits(breakdown_data: Dict[str, Any]) -> Dict[str, Any]:
    fits: Dict[str, Any] = {}
    thresholds = _thresholds_from_breakdown_data(breakdown_data)
    for layer in LAYERS:
        records = breakdown_data["layers"].get(layer, [])
        fits[layer] = {
            _threshold_key(threshold): _fit_breakdown_scaling(records, threshold)
            for threshold in thresholds
        }
    return fits


def _write_breakdown_times(
    job_dir: Path,
    breakdown_data: Dict[str, Any],
) -> Path:
    file_path = job_dir / "break_down_times_update_norms.json"
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(breakdown_data, file, indent=2)
    return file_path


def _plot_breakdown_times(
    breakdown_data: Dict[str, Any],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    thresholds = _thresholds_from_breakdown_data(breakdown_data)
    fits = breakdown_data.get("fits") or _compute_breakdown_fits(breakdown_data)

    for ax, layer in zip(axes, LAYERS):
        records = breakdown_data["layers"].get(layer, [])
        model_parameters = sorted(
            {record.get("parameters", record["network_size"]) for record in records}
        )
        fit_lines = [r"$\log_2 t = \alpha \log_2 P + \beta$"]
        layer_fit_values = []

        for threshold_index, threshold in enumerate(thresholds):
            key = _threshold_key(threshold)
            marker = _threshold_marker(threshold_index)
            threshold_color = _threshold_color(threshold_index, len(thresholds))
            for record in records:
                break_down_time = record["break_down_times"].get(key)
                if break_down_time is None or break_down_time <= 0.0:
                    continue
                ax.scatter(
                    record.get("parameters", record["network_size"]),
                    break_down_time,
                    marker=marker,
                    s=55,
                    color=record["color"],
                    edgecolors="black",
                    linewidths=0.35,
                )

            fit = fits.get(layer, {}).get(key)
            if fit is None:
                continue

            layer_fit_values.append(fit)
            x_fit = 2 ** np.linspace(
                np.log2(fit["parameter_min"]),
                np.log2(fit["parameter_max"]),
                num=200,
            )
            y_fit = (2 ** fit["offset"]) * (x_fit ** fit["scaling_exponent"])
            ax.plot(
                x_fit,
                y_fit,
                color=threshold_color,
                linewidth=1.7,
                alpha=0.95,
            )

        if layer_fit_values:
            mean_alpha = float(
                np.mean([fit["scaling_exponent"] for fit in layer_fit_values])
            )
            mean_beta = float(np.mean([fit["offset"] for fit in layer_fit_values]))
            fit_lines.append(rf"$\langle\alpha\rangle={mean_alpha:.3g}$")
            fit_lines.append(rf"$\langle\beta\rangle={mean_beta:.3g}$")
            fit_lines.append(f"fits averaged = {len(layer_fit_values)}")
        else:
            fit_lines.append("fit unavailable")

        ax.set_title(layer, fontsize=15)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("Number of parameters " + r"$(P)$", fontsize=13)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)
        if model_parameters:
            ax.set_xticks(model_parameters)
            ax.set_xticklabels(
                [str(int(parameters)) for parameters in model_parameters],
                rotation=45,
                ha="right",
            )
        ax.text(
            0.03,
            0.97,
            "\n".join(fit_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.35"),
        )

    axes[0].set_ylabel("Breakdown time", fontsize=13)
    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="-",
            marker=_threshold_marker(index),
            color=_threshold_color(index, len(thresholds)),
            markerfacecolor="white",
            markeredgecolor="black",
            label=rf"$>{threshold:g}\sigma$",
        )
        for index, threshold in enumerate(thresholds)
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.80, 0.5),
        ncol=2,
        frameon=True,
        fontsize=8,
    )
    fig.suptitle("Update-Norm Breakdown Times by Parameter Count", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 0.78, 0.93])

    file_path = output_dir / "break_down_times_update_norms.png"
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
    return file_path


def _plot_layer_series(
    ax_top: Any,
    ax_bottom: Any | None,
    series: PlotSeries,
    layer: str,
    color: Any,
    *,
    label: str | None = None,
) -> Any:
    x_axis = series.x_axis[: len(series.step_norms_mean)]
    line = _plot_with_uncertainty(
        ax_top,
        x_axis,
        series.step_norms_mean,
        series.step_norms_std,
        color=color,
        label=label,
        shade=series.has_uncertainty,
        min_lower=np.finfo(float).tiny,
    )

    if ax_bottom is not None and series.scheme == "muP":
        scaled_mean, scaled_std = _scaled_norms(series, layer)
        _plot_with_uncertainty(
            ax_bottom,
            x_axis,
            scaled_mean,
            scaled_std,
            color=color,
            shade=series.has_uncertainty and _has_uncertainty(scaled_std),
            min_lower=np.finfo(float).tiny,
        )
    return line


def _plot_deviation_from_largest_run(
    ax: Any,
    series: PlotSeries,
    reference: PlotSeries,
    layer: str,
    delta_t_std: int,
    color: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    x_axis, deviation = _deviation_from_reference(
        series,
        reference,
        layer,
        delta_t_std,
    )
    ax.plot(x_axis, deviation, color=color)
    return x_axis, deviation


def _plot_loss_figure(
    series_list: List[PlotSeries],
    output_dir: Path,
    compute_flag: bool,
    delta_t_avg: int,
    delta_t_std: int,
    x_label: str,
    title: str,
) -> Path:
    scheme_groups = {
        "standard": [series for series in series_list if series.scheme == "standard"],
        "muP": [series for series in series_list if series.scheme == "muP"],
    }
    color_map = {
        "standard": _prepare_color_map(len(scheme_groups["standard"]), plt.cm.autumn),
        "muP": _prepare_color_map(len(scheme_groups["muP"]), plt.cm.winter),
    }
    color_indices = {"standard": 0, "muP": 0}
    legend_entries: Dict[str, List[Tuple[Any, str]]] = {"standard": [], "muP": []}

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for series in series_list:
        color = color_map[series.scheme][color_indices[series.scheme]]
        color_indices[series.scheme] += 1
        line = _plot_with_uncertainty(
            ax,
            series.x_axis[: len(series.losses_mean)],
            series.losses_mean,
            series.losses_std,
            color=color,
            label=series.label,
            shade=_has_uncertainty(series.losses_std),
            min_lower=np.finfo(float).tiny,
        )
        legend_entries[series.scheme].append((line, series.label))

    ax.set_xscale("log", base=10)
    ax.set_yscale("log")
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(f"{_loss_plot_label(LOSS_TYPE)} [log]", fontsize=18)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=18)

    combined_handles, combined_labels, header_indices = _build_combined_legend(
        legend_entries,
        {"standard": "Standard parametrization", "muP": "muP parametrization"},
    )
    if combined_handles:
        legend = fig.legend(
            combined_handles,
            combined_labels,
            loc="center left",
            bbox_to_anchor=(0.76, 0.5),
            ncol=1,
            frameon=True,
            handlelength=1.5,
            handletextpad=0.6,
            fontsize=13,
        )
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        for index in header_indices:
            legend.get_texts()[index].set_fontweight("bold")

    fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=[0, 0.02, 0.74, 0.9])

    file_stem = f"{LOSS_TYPE}_loss_c" if compute_flag else f"{LOSS_TYPE}_loss"
    file_path = output_dir / f"{file_stem}_dt_avg_{delta_t_avg}_dt_std_{delta_t_std}.png"
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
    return file_path


def plot_update_norms(
    job_dir: Path,
    compute_flag: bool,
    delta_t_avg: int,
    delta_t_std: int,
    start_idx: int | None = None,
    stop_idx: int | None = None,
    sigma_start: float = DEFAULT_SIGMA_START,
    sigma_stop: float = DEFAULT_SIGMA_STOP,
    num_sigmas: int = DEFAULT_NUM_SIGMAS,
    skip_breakdown: bool = False,
) -> None:
    _validate_window_width("delta_t_avg", delta_t_avg, minimum=1)
    _validate_window_width("delta_t_std", delta_t_std, minimum=2)
    thresholds = _sigma_thresholds(sigma_start, sigma_stop, num_sigmas)

    log_paths = collect_files_with_ending(job_dir, "training_log.json")
    if not log_paths:
        raise FileNotFoundError(f"No training logs found in {job_dir}")

    log_paths = sorted(log_paths, key=lambda path: tuple(_width_key(_label_from_log_path(path))))
    series_pairs = [
        tuple(_collect_plot_series(job_dir, log_path, layer, compute_flag, LOSS_TYPE) for layer in LAYERS)
        for log_path in log_paths
    ]
    series_pairs = [
        tuple(_slice_series(series, start_idx, stop_idx) for series in pair)
        for pair in series_pairs
    ]
    loss_series_list = [pair[0] for pair in series_pairs]
    series_pairs = [
        tuple(_average_observables(series, delta_t_avg) for series in pair)
        for pair in series_pairs
    ]

    task_names = {series.task_name for pair in series_pairs for series in pair}
    if len(task_names) != 1:
        raise ValueError("All runs in the provided job directory must correspond to the same task.")

    first_series = series_pairs[0][0]
    dataset_info = first_series.dataset_info
    task_name = first_series.task_name
    x_label = _build_x_axis(
        first_series.losses_mean,
        first_series.dataset_info,
        first_series.network_info,
        compute_flag,
    )[1]
    is_average_job = job_dir.name.startswith("avg")

    scheme_groups = {
        "standard": [pair[0] for pair in series_pairs if pair[0].scheme == "standard"],
        "muP": [pair[0] for pair in series_pairs if pair[0].scheme == "muP"],
    }
    color_map = {
        "standard": _prepare_color_map(len(scheme_groups["standard"]), plt.cm.autumn),
        "muP": _prepare_color_map(len(scheme_groups["muP"]), plt.cm.winter),
    }
    show_scaled_update_norms = bool(scheme_groups["muP"])
    color_indices = {"standard": 0, "muP": 0}
    legend_entries: Dict[str, List[Tuple[Any, str]]] = {"standard": [], "muP": []}
    largest_series_by_layer = {
        "Dense_0": _largest_width_series([pair[0] for pair in series_pairs]),
        "Dense_1": _largest_width_series([pair[1] for pair in series_pairs]),
    }
    breakdown_data: Dict[str, Any] | None = None
    if not skip_breakdown:
        breakdown_data = {
            "thresholds": [float(threshold) for threshold in thresholds],
            "x_axis": "compute" if compute_flag else "training_steps",
            "settings": {
                "delta_t_avg": delta_t_avg,
                "delta_t_std": delta_t_std,
                "start_idx": start_idx,
                "stop_idx": stop_idx,
                "sigma_start": sigma_start,
                "sigma_stop": sigma_stop,
                "num_sigmas": num_sigmas,
            },
            "layers": {layer: [] for layer in LAYERS},
        }

    if show_scaled_update_norms:
        fig_main, axes_main = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        main_top_axes = (axes_main[0, 0], axes_main[0, 1])
        main_bottom_axes = (axes_main[1, 0], axes_main[1, 1])
    else:
        fig_main, main_top_axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
        main_bottom_axes = (None, None)
    fig_deviation, axes_deviation = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    axes_by_layer = {
        "Dense_0": (main_top_axes[0], main_bottom_axes[0], axes_deviation[0]),
        "Dense_1": (main_top_axes[1], main_bottom_axes[1], axes_deviation[1]),
    }

    for dense0_series, dense1_series in series_pairs:
        color = color_map[dense0_series.scheme][color_indices[dense0_series.scheme]]
        color_indices[dense0_series.scheme] += 1

        line = _plot_layer_series(
            *axes_by_layer["Dense_0"][:2],
            dense0_series,
            "Dense_0",
            color,
            label=dense0_series.label,
        )
        _plot_layer_series(*axes_by_layer["Dense_1"][:2], dense1_series, "Dense_1", color)
        dense0_deviation = _plot_deviation_from_largest_run(
            axes_by_layer["Dense_0"][2],
            dense0_series,
            largest_series_by_layer["Dense_0"],
            "Dense_0",
            delta_t_std,
            color,
        )
        dense1_deviation = _plot_deviation_from_largest_run(
            axes_by_layer["Dense_1"][2],
            dense1_series,
            largest_series_by_layer["Dense_1"],
            "Dense_1",
            delta_t_std,
            color,
        )
        if breakdown_data is not None:
            breakdown_data["layers"]["Dense_0"].append(
                _breakdown_record(dense0_series, color, *dense0_deviation, thresholds)
            )
            breakdown_data["layers"]["Dense_1"].append(
                _breakdown_record(dense1_series, color, *dense1_deviation, thresholds)
            )
        legend_entries[dense0_series.scheme].append((line, dense0_series.label))

    for layer in LAYERS:
        ax_top, ax_bottom, ax_deviation = axes_by_layer[layer]
        layer_label = rf"$l={layer[-1]}$"
        ax_top.set_title(layer_label, fontsize=20)
        ax_deviation.set_title(layer_label, fontsize=18)
        ax_top.set_ylabel(_top_ylabel(layer), fontsize=18)
        if ax_bottom is None:
            ax_top.set_xlabel(x_label, fontsize=18)
        else:
            ax_bottom.set_ylabel(_bottom_ylabel(layer), fontsize=20)
            ax_bottom.set_xlabel(x_label, fontsize=18)
        ax_deviation.set_ylabel(_deviation_ylabel(layer), fontsize=16)
        ax_deviation.set_xlabel(x_label, fontsize=18)
        ax_deviation.text(
            0.97,
            0.95,
            rf"$n_{{\infty}}$ proxy: {largest_series_by_layer[layer].label}",
            transform=ax_deviation.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        main_axes = (ax_top,) if ax_bottom is None else (ax_top, ax_bottom)
        for ax in main_axes:
            ax.set_xscale("log", base=10)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.tick_params(axis="both", labelsize=18)
        ax_deviation.set_xscale("log", base=10)
        ax_deviation.grid(True, which="both", alpha=0.3)
        ax_deviation.tick_params(axis="both", labelsize=18)

    combined_handles, combined_labels, header_indices = _build_combined_legend(
        legend_entries,
        {"standard": "Standard parametrization", "muP": "muP parametrization"},
    )
    if combined_handles:
        legend = fig_main.legend(
            combined_handles,
            combined_labels,
            loc="center left",
            bbox_to_anchor=(0.84, 0.5),
            ncol=1,
            frameon=True,
            handlelength=1.5,
            handletextpad=0.6,
            fontsize=13,
        )
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        for index in header_indices:
            legend.get_texts()[index].set_fontweight("bold")

    title = (
        f"$\\tau = {_format_save_loss_frequency(first_series.save_loss_frequency)}$ "
        + r"SGD Update Steps are Conducted Between Subsequent Data Points"
        + r" $t_{i+1}$ and $t_i$"
    )
    if is_average_job:
        title += "\nPre-averaged job; shaded regions show +/- " + r"$\sigma$"
    fig_main.suptitle(title, fontsize=20)
    fig_main.tight_layout(
        rect=[0, 0.02, 0.82, 0.9 if show_scaled_update_norms else 0.86]
    )

    deviation_title = (
        r"Normalized Update-Norm Deviations for $l=0$ and $l=1$"
        + f"\n$\\tau = {_format_save_loss_frequency(first_series.save_loss_frequency)}$ SGD Update Steps are Conducted Between Subsequent Data Points"
        + r" $t_{i+1}$ and $t_i$"
    )
    fig_deviation.suptitle(deviation_title, fontsize=18)
    fig_deviation.tight_layout(rect=[0, 0.02, 1, 0.92])

    output_dir = Path(f"figures_update_norm_analysis/{task_name}/{job_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = "update_norms_c" if compute_flag else "update_norms"
    file_path = output_dir / f"{file_stem}_dt_avg_{delta_t_avg}_dt_std_{delta_t_std}.png"
    deviation_file_path = (
        output_dir / f"{file_stem}_deviation_dt_avg_{delta_t_avg}_dt_std_{delta_t_std}.png"
    )
    fig_main.savefig(file_path, bbox_inches="tight")
    fig_deviation.savefig(deviation_file_path, bbox_inches="tight")
    plt.close(fig_main)
    plt.close(fig_deviation)
    print(f"Saved plot to {file_path}")
    print(f"Saved deviation plot to {deviation_file_path}")
    loss_plot_path = _plot_loss_figure(
        loss_series_list,
        output_dir,
        compute_flag,
        delta_t_avg,
        delta_t_std,
        x_label,
        title,
    )
    print(f"Saved loss plot to {loss_plot_path}")
    if breakdown_data is not None:
        breakdown_data["fits"] = _compute_breakdown_fits(breakdown_data)
        breakdown_file_path = _write_breakdown_times(job_dir, breakdown_data)
        with breakdown_file_path.open("r", encoding="utf-8") as file:
            saved_breakdown_data = json.load(file)
        breakdown_plot_path = _plot_breakdown_times(saved_breakdown_data, output_dir)
        print(f"Saved breakdown plot to {breakdown_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Dense_0 and Dense_1 weight-update norms for a job."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Path to a single job directory containing training_log.json files.",
    )
    parser.add_argument(
        "--compute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot compute on the x-axis instead of training steps.",
    )
    parser.add_argument(
        "--delta-t-avg",
        "--delta_t_avg",
        dest="delta_t_avg",
        type=int,
        default=1,
        help=(
            "Width, in saved observable data points, of the symmetric sliding "
            "window used to average update norms and cosine similarities. "
            "Use 1 to disable observable averaging."
        ),
    )
    parser.add_argument(
        "--delta-t-std",
        "--delta_t_std",
        dest="delta_t_std",
        type=int,
        default=DEVIATION_STD_WINDOW,
        help=(
            "Width, in saved observable data points, of the rolling window used "
            "to estimate log-space deviation uncertainty. Must be at least 2."
        ),
    )
    parser.add_argument(
        "--start-idx",
        "--start_idx",
        dest="start_idx",
        type=int,
        default=None,
        help="Optional start index for slicing all evaluated time series.",
    )
    parser.add_argument(
        "--stop-idx",
        "--stop_idx",
        dest="stop_idx",
        type=int,
        default=None,
        help="Optional stop index for slicing all evaluated time series.",
    )
    parser.add_argument(
        "--sigma-start",
        "--simga-start",
        dest="sigma_start",
        type=float,
        default=DEFAULT_SIGMA_START,
        help="Start of the sigma-threshold range used for breakdown times.",
    )
    parser.add_argument(
        "--sigma-stop",
        dest="sigma_stop",
        type=float,
        default=DEFAULT_SIGMA_STOP,
        help="End of the sigma-threshold range used for breakdown times.",
    )
    parser.add_argument(
        "--num-sigmas",
        dest="num_sigmas",
        type=int,
        default=DEFAULT_NUM_SIGMAS,
        help="Number of equidistant sigma thresholds to evaluate.",
    )
    parser.add_argument(
        "--skip-breakdown",
        action="store_true",
        help="Save only the update-norm and loss figures and skip breakdown-time outputs.",
    )
    args = parser.parse_args()
    plot_update_norms(
        args.log_dir,
        args.compute,
        args.delta_t_avg,
        args.delta_t_std,
        args.start_idx,
        args.stop_idx,
        args.sigma_start,
        args.sigma_stop,
        args.num_sigmas,
        args.skip_breakdown,
    )


if __name__ == "__main__":
    main()
