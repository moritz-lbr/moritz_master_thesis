#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

_CACHE_DIR = Path(".cache/matplotlib")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR.resolve()))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

import plot_superposition as ps


LAYER = "Dense_0"
KERNEL_KEY = "Dense_0/kernel"
AXIS_LABEL_FONTSIZE = 15


def dense0_kernel(zarr_path: Path) -> tuple[Path, dict[str, Any], tuple[int, ...]]:
    array_dir = zarr_path / KERNEL_KEY
    if not (array_dir / "zarr.json").exists():
        available = [
            (layer, key)
            for layer, key, _, _, _ in ps.dense_kernel_entries(zarr_path)
            if layer == LAYER
        ]
        if not available:
            raise FileNotFoundError(f"No {KERNEL_KEY} array found in {zarr_path}")
        _, key = available[0]
        array_dir = zarr_path / key
    meta = ps.load_json(array_dir / "zarr.json")
    shape = tuple(int(value) for value in meta["shape"])
    if len(shape) != 3:
        raise ValueError(f"Expected Dense_0/kernel snapshots with shape (t, d, n), got {shape}")
    return array_dir, meta, shape


def dense1_kernel(zarr_path: Path) -> tuple[Path, dict[str, Any], tuple[int, ...]]:
    array_dir = zarr_path / "Dense_1" / "kernel"
    if not (array_dir / "zarr.json").exists():
        raise FileNotFoundError(f"No Dense_1/kernel array found in {zarr_path}")
    meta = ps.load_json(array_dir / "zarr.json")
    shape = tuple(int(value) for value in meta["shape"])
    if len(shape) != 3:
        raise ValueError(f"Expected Dense_1/kernel snapshots with shape (t, n, out), got {shape}")
    return array_dir, meta, shape


def job_output_dir(base: Path, run_dir: Path) -> Path:
    parts: list[Path] = [base]
    task = ps.task_name(run_dir)
    if task:
        parts.append(Path(task))
    parts.append(Path(run_dir.parent.name))
    return Path(*parts)


def cache_dir(base: Path, run_dir: Path) -> Path:
    return job_output_dir(base, run_dir) / f"analysis_cache_{LAYER}"


def input_space_output_dir(base: Path, run_dir: Path) -> Path:
    return job_output_dir(base, run_dir) / run_dir.name / LAYER / "input_space_frames"


def spectral_reconstruction_output_dir(base: Path, run_dir: Path) -> Path:
    return job_output_dir(base, run_dir) / f"spectral_reconstruction_{LAYER}"


def selected_analysis_times(length: int, frequency: int, min_step: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    times = {int(time_index): int(step) for time_index, step in ps.selected_times(length, frequency, min_step)}
    times[0] = 0
    final_index = int(length) - 1
    times[final_index] = final_index * int(frequency)
    return sorted(times.items())


def cache_matches_times(record: dict[str, Any], times: Sequence[tuple[int, int]]) -> bool:
    cached = list(
        zip(
            np.asarray(record["time_indices"], dtype=np.int64).tolist(),
            np.asarray(record["steps"], dtype=np.int64).tolist(),
        )
    )
    expected = [(int(time_index), int(step)) for time_index, step in times]
    return cached == expected


def set_nonnegative_loglike_yscale(ax: plt.Axes, arrays: Sequence[Any]) -> None:
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
        ax.set_yscale("log")
        return
    positive = values[values > 0]
    if np.all(values >= 0) and positive.size:
        min_positive = float(np.min(positive))
        max_positive = float(np.max(positive))
        zero_position = min_positive / 10.0
        for line in ax.get_lines():
            ydata = np.asarray(line.get_ydata(), dtype=float)
            if ydata.size:
                line.set_ydata(np.where(ydata == 0.0, zero_position, ydata))
        ax.set_yscale("log")
        decade_min = int(np.floor(np.log10(min_positive)))
        decade_max = int(np.ceil(np.log10(max_positive)))
        positive_ticks = [
            10.0**exponent
            for exponent in range(decade_min, decade_max + 1)
            if min_positive <= 10.0**exponent <= max_positive
        ]
        ticks = [zero_position] + positive_ticks
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(
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
        ax.set_ylim(zero_position / np.sqrt(10.0), max_positive * np.sqrt(10.0))


def weight_rows_by_input_unit(weights: np.ndarray) -> np.ndarray:
    width = int(weights.shape[-1])
    return weights.reshape(-1, width).astype(np.float32, copy=False)


def input_overlap_matrix(rows: np.ndarray) -> np.ndarray:
    width = max(int(rows.shape[1]), 1)
    return (rows @ rows.T / float(width)).astype(np.float32, copy=False)


def reduced_overlap_matrix(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    input_dim, width = rows.shape
    scale = 1.0 / float(max(width, 1))
    diagonal = np.sum(np.square(rows, dtype=np.float64), axis=1) * scale
    rows64 = rows.astype(np.float64, copy=False)
    if width <= input_dim:
        return rows64.T @ rows64 * scale, diagonal
    return rows64 @ rows64.T * scale, diagonal


def overlap_summary(rows: np.ndarray) -> tuple[float, np.ndarray]:
    reduced, diagonal = reduced_overlap_matrix(rows)
    eigenvalues = np.linalg.eigvalsh(reduced)
    eigenvalues = np.sort(np.maximum(eigenvalues.astype(float), 0.0))[::-1]
    frob_sq = float(np.sum(np.square(eigenvalues, dtype=np.float64)))
    diagonal_sq = float(np.sum(np.square(diagonal, dtype=np.float64)))
    offdiag_sq = max(frob_sq - diagonal_sq, 0.0)
    return float(np.sqrt(offdiag_sq)), eigenvalues


def load_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "run": str(data["run"].item()),
            "steps": np.asarray(data["steps"], dtype=np.int64),
            "time_indices": np.asarray(data["time_indices"], dtype=np.int64),
            "frob_norms": np.asarray(data["frob_norms"], dtype=float),
            "eigenvalues": np.asarray(data["eigenvalues"], dtype=float),
            "input_dim": int(data["input_dim"].item()),
            "width": int(data["width"].item()),
            "batch_size": float(data["batch_size"].item()),
            "total_params": float(data["total_params"].item()),
            "job_dir": path.parent.parent,
            "cache_path": path,
        }


def save_cache(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        run=np.asarray(record["run"]),
        steps=np.asarray(record["steps"], dtype=np.int64),
        time_indices=np.asarray(record["time_indices"], dtype=np.int64),
        frob_norms=np.asarray(record["frob_norms"], dtype=np.float64),
        eigenvalues=np.asarray(record["eigenvalues"], dtype=np.float64),
        input_dim=np.asarray(int(record["input_dim"]), dtype=np.int64),
        width=np.asarray(int(record["width"]), dtype=np.int64),
        batch_size=np.asarray(float(record["batch_size"]), dtype=np.float64),
        total_params=np.asarray(float(record["total_params"]), dtype=np.float64),
    )


def compute_or_load_run_analysis(
    zarr_path: Path,
    output_base: Path,
    min_step: int,
    *,
    force: bool,
) -> dict[str, Any]:
    run_dir = zarr_path.parent
    array_dir, meta, shape = dense0_kernel(zarr_path)
    step_stride = ps.training_step_stride(run_dir)
    times = selected_analysis_times(shape[0], step_stride, min_step)
    if not times:
        raise ValueError(f"No selected Dense_0 snapshots for {run_dir}")

    run_cache = cache_dir(output_base, run_dir) / f"{run_dir.name}_{LAYER}_overlap.npz"
    if run_cache.exists() and not force:
        cached = load_cache(run_cache)
        if cache_matches_times(cached, times):
            return cached
        print(f"Recomputing stale Dense_0 overlap analysis cache with updated step grid: {run_cache}")

    config = ps.load_yaml(run_dir / "simulation_config.yaml")
    network = config.get("network", {})
    training = config.get("training", {})
    steps: list[int] = []
    time_indices: list[int] = []
    frob_norms: list[float] = []
    spectra: list[np.ndarray] = []
    for time_index, step in times:
        rows = weight_rows_by_input_unit(ps.read_snapshot(array_dir, meta, time_index))
        frob_norm, eigenvalues = overlap_summary(rows)
        steps.append(int(step))
        time_indices.append(int(time_index))
        frob_norms.append(float(frob_norm))
        spectra.append(eigenvalues)

    record = {
        "run": run_dir.name,
        "steps": np.asarray(steps, dtype=np.int64),
        "time_indices": np.asarray(time_indices, dtype=np.int64),
        "frob_norms": np.asarray(frob_norms, dtype=float),
        "eigenvalues": np.vstack(spectra),
        "input_dim": int(np.prod(shape[1:-1])),
        "width": int(shape[-1]),
        "batch_size": float(training.get("batch_size", 1.0)),
        "total_params": float(network.get("total_params", 1.0)),
        "job_dir": job_output_dir(output_base, run_dir),
        "cache_path": run_cache,
    }
    save_cache(run_cache, record)
    print(f"Saved cached Dense_0 overlap analysis to {run_cache}")
    return record


def write_cache_manifest(records: Sequence[dict[str, Any]], output_base: Path, input_dir: Path) -> None:
    if not records:
        return
    manifest_dir = Path(records[0]["cache_path"]).parent
    payload = {
        "input_dir": str(input_dir),
        "layer": LAYER,
        "matrix": "C_ij = (1/n) sum_alpha W_{i,alpha} W_{j,alpha}",
        "runs": [
            {
                "run": record["run"],
                "input_dim": int(record["input_dim"]),
                "width": int(record["width"]),
                "steps": [int(step) for step in record["steps"]],
                "cache_file": Path(record["cache_path"]).name,
            }
            for record in ps.sorted_series_entries(list(records))
        ],
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def symmetric_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    span = max(float(np.max(np.abs(finite))), 1e-12)
    return -span, span


def symmetric_offdiag_limits(matrix: np.ndarray) -> tuple[float, float]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        return symmetric_limits(values)
    mask = ~np.eye(values.shape[0], dtype=bool)
    offdiag = values[mask]
    finite = offdiag[np.isfinite(offdiag)]
    if finite.size == 0:
        return symmetric_limits(values)
    span = max(float(np.max(np.abs(finite))), 1e-12)
    return -span, span


def display_channel_label(layout: ps.ImageLayout, channel: int) -> str:
    if layout.channels == 3:
        return f"CH{int(channel) + 1}"
    return layout.channel_names[channel]


def plot_input_space_frame(
    matrix: np.ndarray,
    reference_units: np.ndarray,
    layout: ps.ImageLayout,
    output_path: Path,
    *,
    title: str,
    channel_mode: str,
    dpi: int,
    colorbar_factor: float = 1.0,
) -> None:
    expected_units = layout.height * layout.width * layout.channels
    if matrix.shape != (expected_units, expected_units):
        raise ValueError(f"Cannot plot {matrix.shape}; expected {expected_units}x{expected_units}")
    if float(colorbar_factor) <= 0.0:
        raise ValueError("colorbar_factor must be positive")

    mode = "split" if layout.channels == 1 else channel_mode
    row_count = layout.channels if mode == "split" else 1
    col_count = len(reference_units)
    columns = matrix[:, reference_units]
    cube = columns.reshape(layout.height, layout.width, layout.channels, col_count)
    colors = ps.reference_highlight_colors(reference_units, layout)
    vmin, vmax = symmetric_offdiag_limits(matrix)
    vmin *= float(colorbar_factor)
    vmax *= float(colorbar_factor)
    label_kwargs = {"fontsize": 15, "fontweight": "bold"}

    grid_col_count = 1 + col_count + 1
    width_ratios = [float(max(col_count, 1))] + [1.0] * col_count + [0.06]
    fig_width = max(10.8, 2.05 * (col_count + max(col_count, 1)) + 1.2)
    fig_height = max(4.2, 2.0 * row_count + 1.0)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(row_count, grid_col_count, width_ratios=width_ratios)

    matrix_ax = fig.add_subplot(grid[:, 0])
    matrix_ax.imshow(
        matrix,
        origin="upper",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    for ref_index, ref_unit in enumerate(reference_units):
        color = colors[ref_index]
        matrix_ax.axvspan(int(ref_unit) - 0.5, int(ref_unit) + 0.5, color=color, alpha=0.16, linewidth=0)
        matrix_ax.axvline(int(ref_unit), color=color, linewidth=1.1, alpha=0.9)
    matrix_ax.set_title(r"$C_{ij}=\frac{1}{n}\sum_{\alpha=1}^{n} W_{i\alpha}W_{j\alpha}$", **label_kwargs)
    matrix_ax.set_xlabel(r"input coordinate $i$", **label_kwargs)
    matrix_ax.set_ylabel(r"input coordinate $j$", **label_kwargs)

    image = None
    for ref_index, ref_unit in enumerate(reference_units):
        ref_row, ref_col, ref_channel = ps.unit_image_coordinates(int(ref_unit), layout)
        for row_index in range(row_count):
            channel = row_index if mode == "split" else ref_channel
            ax = fig.add_subplot(grid[row_index, ref_index + 1])
            image = ax.imshow(
                cube[:, :, channel, ref_index],
                origin="upper",
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            if mode == "split" or channel == ref_channel:
                ax.add_patch(
                    Rectangle(
                        (ref_col - 0.5, ref_row - 0.5),
                        1.0,
                        1.0,
                        facecolor=colors[ref_index],
                        edgecolor="black",
                        linewidth=0.6,
                        alpha=0.95,
                    )
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                channel_label = display_channel_label(layout, ref_channel)
                ax.set_title(
                    rf"$C_{{i,{int(ref_unit)}}}$"
                    "\n"
                    f"r={ref_row}, c={ref_col}, ch={channel_label}",
                    fontsize=9,
                    fontweight="bold",
                    color=colors[ref_index],
                )
            if ref_index == 0:
                channel_label = display_channel_label(layout, channel)
                ylabel = channel_label if mode == "split" else f"ref {channel_label}"
                ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")

    if image is not None:
        cax = fig.add_subplot(grid[:, -1])
        fig.colorbar(image, cax=cax)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_input_space_frames(
    zarr_path: Path,
    output_base: Path,
    record: dict[str, Any],
    *,
    reference_count: int,
    channel_mode: str,
    dpi: int,
    force: bool,
) -> int:
    run_dir = zarr_path.parent
    array_dir, meta, shape = dense0_kernel(zarr_path)
    input_dim = int(np.prod(shape[1:-1]))
    layout = ps.infer_image_layout(run_dir, input_dim)
    if layout is None:
        print(f"Skipping input-space frames for {run_dir.name}: no image layout for {input_dim} units")
        return 0

    reference_units = ps.selected_reference_units(input_dim, reference_count, layout=layout)
    out_dir = input_space_output_dir(output_base, run_dir)
    saved = 0
    for frame_index, (time_index, step) in enumerate(zip(record["time_indices"], record["steps"])):
        output_path = out_dir / f"input_space_{frame_index:03d}_step_{int(step)}.png"
        if output_path.exists() and not force:
            continue
        rows = weight_rows_by_input_unit(ps.read_snapshot(array_dir, meta, int(time_index)))
        matrix = input_overlap_matrix(rows)
        plot_input_space_frame(
            matrix,
            reference_units,
            layout,
            output_path,
            title=f"Training Step: {int(step)}",
            channel_mode=channel_mode,
            dpi=dpi,
        )
        saved += 1
    print(f"Saved {saved} input-space frames for {run_dir.name} to {out_dir}")
    return saved


def factor_label(factor: float) -> str:
    return f"{float(factor):g}".replace("-", "m").replace(".", "p")


def input_space_factor_suffix(factor: float) -> str:
    if np.isclose(float(factor), 1.0):
        return ""
    return f"_colorbar_factor_{factor_label(float(factor))}"


def save_single_input_space_frame(
    zarr_path: Path,
    output_base: Path,
    *,
    time_index: int,
    step: int,
    filename_stem: str,
    reference_count: int,
    channel_mode: str,
    dpi: int,
    force: bool,
    colorbar_factors: Sequence[float],
    test_suffix: bool = False,
) -> int:
    run_dir = zarr_path.parent
    array_dir, meta, shape = dense0_kernel(zarr_path)
    input_dim = int(np.prod(shape[1:-1]))
    layout = ps.infer_image_layout(run_dir, input_dim)
    if layout is None:
        print(f"Skipping input-space frame for {run_dir.name}: no image layout for {input_dim} units")
        return 0

    reference_units = ps.selected_reference_units(input_dim, reference_count, layout=layout)
    rows = weight_rows_by_input_unit(ps.read_snapshot(array_dir, meta, int(time_index)))
    matrix = input_overlap_matrix(rows)
    out_dir = input_space_output_dir(output_base, run_dir)
    saved = 0
    for factor in colorbar_factors:
        suffix = input_space_factor_suffix(float(factor))
        if test_suffix and suffix:
            suffix = f"{suffix}_test"
        output_path = out_dir / f"{filename_stem}{suffix}.png"
        if output_path.exists() and not force:
            continue
        plot_input_space_frame(
            matrix,
            reference_units,
            layout,
            output_path,
            title=f"Training Step: {int(step)}",
            channel_mode=channel_mode,
            dpi=dpi,
            colorbar_factor=float(factor),
        )
        saved += 1
    return saved


def save_best_input_space_frames(
    zarr_path: Path,
    output_base: Path,
    *,
    reference_count: int,
    channel_mode: str,
    dpi: int,
    force: bool,
    colorbar_factors: Sequence[float],
) -> int:
    run_dir = zarr_path.parent
    _, _, shape = dense0_kernel(zarr_path)
    snapshot = ps.best_test_loss_snapshot(
        run_dir,
        int(shape[0]),
        ps.training_step_stride(run_dir),
    )
    if snapshot is None:
        print(f"Skipping best-loss input-space frame for {run_dir.name}: no finite test loss")
        return 0
    step = int(snapshot["best_step"])
    saved = save_single_input_space_frame(
        zarr_path,
        output_base,
        time_index=int(snapshot["best_time_index"]),
        step=step,
        filename_stem=f"input_space_best_test_loss_step_{step}",
        reference_count=reference_count,
        channel_mode=channel_mode,
        dpi=dpi,
        force=force,
        colorbar_factors=colorbar_factors,
    )
    print(
        f"Saved {saved} best-loss input-space frame variants for "
        f"{run_dir.name} at step {step}"
    )
    return saved


def cache_records_to_series(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    series = []
    for record in records:
        series.append(
            {
                "job_dir": record["job_dir"],
                "layer": LAYER,
                "run": record["run"],
                "steps": np.asarray(record["steps"], dtype=np.int64),
                "frob_norms": np.asarray(record["frob_norms"], dtype=float),
                "input_dim": int(record["input_dim"]),
                "output_dim": int(record["width"]),
                "batch_size": float(record["batch_size"]),
                "total_params": float(record["total_params"]),
            }
        )
    return series


def save_frobenius_plot(records: Sequence[dict[str, Any]], dpi: int) -> None:
    if not records:
        return
    job_dir = Path(records[0]["job_dir"])
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.8), constrained_layout=True)
    raw_ax, initial_ax, log_initial_ax = axes
    series = cache_records_to_series(records)
    colors, legend_entries = ps.series_colors(series)
    initial_y_arrays = []
    for entry in ps.sorted_series_entries(series):
        scheme = ps.scheme_from_run(entry["run"])
        label = ps.label_from_run(entry["run"])
        color = colors[entry["run"]]
        values = np.asarray(entry["frob_norms"], dtype=float)
        line, = raw_ax.plot(
            entry["steps"],
            values,
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            color=color,
        )
        baseline = float(values[0]) if values.size else 0.0
        initial_values = values - baseline
        initial_ax.plot(
            entry["steps"],
            initial_values,
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            color=color,
        )
        log_initial_ax.plot(
            entry["steps"],
            initial_values,
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            color=color,
        )
        initial_y_arrays.append(initial_values)
        legend_entries[scheme].append((line, label))

    all_steps = [entry["steps"] for entry in series]
    ps.set_nonnegative_loglike_xscale(raw_ax, all_steps)
    ps.set_nonnegative_loglike_xscale(initial_ax, all_steps)
    ps.set_nonnegative_loglike_xscale(log_initial_ax, all_steps)
    raw_ax.set_title(f"Off-diagonal superposition Frobenius norm - {LAYER}")
    raw_ax.set_xlabel("Training Step")
    raw_ax.set_ylabel(r"$\|C-\mathrm{diag}(C)\|_F$")
    raw_ax.grid(True, which="both", alpha=0.3)
    ps.apply_combined_legend(raw_ax, legend_entries)

    initial_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    initial_ax.set_title("Initial-subtracted off-diagonal Frobenius norm")
    initial_ax.set_xlabel("Training Step")
    initial_ax.set_ylabel(r"$\|C-\mathrm{diag}(C)\|_F-\|C_0-\mathrm{diag}(C_0)\|_F$")
    initial_ax.grid(True, which="both", alpha=0.3)

    log_initial_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    set_nonnegative_loglike_yscale(log_initial_ax, initial_y_arrays)
    log_initial_ax.set_title("Initial-subtracted off-diagonal Frobenius norm (log y)")
    log_initial_ax.set_xlabel("Training Step")
    log_initial_ax.set_ylabel(r"$\|C-\mathrm{diag}(C)\|_F-\|C_0-\mathrm{diag}(C_0)\|_F$")
    log_initial_ax.grid(True, which="both", alpha=0.3)

    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / f"frob_norms_{LAYER}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Frobenius norm plot to {path}")


def save_eigenvalue_cache_json(records: Sequence[dict[str, Any]]) -> None:
    if not records:
        return
    spectrum_dir = Path(records[0]["job_dir"]) / f"eigenvalue_spectra_{LAYER}"
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    payload = []
    for record in ps.sorted_series_entries(cache_records_to_series(records)):
        source = next(item for item in records if item["run"] == record["run"])
        payload.append(
            {
                "run": source["run"],
                "layer": LAYER,
                "matrix": "C_ij = (1/n) sum_alpha W_{i,alpha} W_{j,alpha}",
                "input_dim": int(source["input_dim"]),
                "width": int(source["width"]),
                "steps": [int(step) for step in source["steps"]],
                "eigenvalues_by_step": [
                    {
                        "step": int(step),
                        "eigenvalues": np.asarray(values, dtype=float).tolist(),
                    }
                    for step, values in zip(source["steps"], source["eigenvalues"])
                ],
            }
        )
    with (spectrum_dir / f"eigenvalue_spectra_{LAYER}.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file)


def spectrum_trace(eigenvalues: np.ndarray) -> float:
    values = np.asarray(eigenvalues, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.sum(finite, dtype=np.float64))


def save_trace_vs_step_plot(
    records: Sequence[dict[str, Any]],
    spectra_by_run: dict[str, dict[int, np.ndarray]],
    spectrum_dir: Path,
    dpi: int,
) -> None:
    if not records:
        return
    series = cache_records_to_series(records)
    colors, legend_entries = ps.series_colors(series)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.4), constrained_layout=True)
    trace_ax, gap_ax = axes
    step_arrays: list[np.ndarray] = []
    trace_by_run: dict[str, dict[int, float]] = {}
    for entry in ps.sorted_series_entries(series):
        run = entry["run"]
        run_spectra = spectra_by_run.get(run, {})
        steps = np.asarray(sorted(run_spectra), dtype=float)
        if steps.size == 0:
            continue
        traces = np.asarray(
            [spectrum_trace(run_spectra[int(step)]) for step in steps],
            dtype=float,
        )
        trace_by_run[run] = {
            int(step): float(trace)
            for step, trace in zip(steps.astype(int), traces)
        }
        line, = trace_ax.plot(
            steps,
            traces,
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            color=colors[run],
        )
        legend_entries[ps.scheme_from_run(run)].append((line, ps.label_from_run(run)))
        step_arrays.append(steps)

    source_entry = max(series, key=lambda item: (int(item["output_dim"]), int(item["input_dim"])))
    source_run = str(source_entry["run"])
    source_traces = trace_by_run.get(source_run, {})
    gap_step_arrays: list[np.ndarray] = []
    for entry in ps.sorted_series_entries(series):
        run = entry["run"]
        if run == source_run:
            continue
        run_traces = trace_by_run.get(run, {})
        common_steps = np.asarray(
            sorted(set(source_traces).intersection(run_traces)),
            dtype=float,
        )
        if common_steps.size == 0:
            continue
        denominator = float(min(int(entry["output_dim"]), int(entry["input_dim"])))
        if denominator <= 0.0:
            continue
        normalized_gap = np.asarray(
            [
                (source_traces[int(step)] - run_traces[int(step)]) / denominator
                for step in common_steps
            ],
            dtype=float,
        )
        gap_ax.plot(
            common_steps,
            normalized_gap,
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            color=colors[run],
        )
        gap_step_arrays.append(common_steps)

    ps.set_nonnegative_loglike_xscale(trace_ax, step_arrays)
    ps.set_nonnegative_loglike_xscale(gap_ax, gap_step_arrays)
    trace_ax.set_title(f"Trace of superposition matrix - {LAYER}")
    trace_ax.set_xlabel("Training Step", fontsize=AXIS_LABEL_FONTSIZE)
    trace_ax.set_ylabel(r"$\mathrm{Tr}(C)$", fontsize=AXIS_LABEL_FONTSIZE)
    trace_ax.grid(True, which="both", alpha=0.3)
    ps.apply_combined_legend(trace_ax, legend_entries)

    gap_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    gap_ax.set_title(
        rf"Trace gap to largest network ({ps.label_from_run(source_run)})"
    )
    gap_ax.set_xlabel("Training Step", fontsize=AXIS_LABEL_FONTSIZE)
    gap_ax.set_ylabel(
        r"$\left(\mathrm{Tr}(C_{n_{\max}})-\mathrm{Tr}(C_n)\right)/\min(n,d)$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    gap_ax.grid(True, which="both", alpha=0.3)

    path = spectrum_dir / f"trace_vs_step_{LAYER}.png"
    legacy_path = spectrum_dir / f"rank_vs_step_{LAYER}.png"
    fig.savefig(path, dpi=dpi)
    fig.savefig(legacy_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved trace-vs-step plot to {path}")


def save_eigenvalue_spectrum_plots(records: Sequence[dict[str, Any]], dpi: int) -> None:
    if not records:
        return
    job_dir = Path(records[0]["job_dir"])
    spectrum_dir = job_dir / f"eigenvalue_spectra_{LAYER}"
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    save_eigenvalue_cache_json(records)

    steps = sorted({int(step) for record in records for step in record["steps"]})
    series = cache_records_to_series(records)
    colors, _ = ps.series_colors(series)
    spectra_by_run = {
        record["run"]: {
            int(step): np.asarray(values, dtype=float)
            for step, values in zip(record["steps"], record["eigenvalues"])
        }
        for record in records
    }

    for step in steps:
        fig, raw_ax = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
        _, legend_entries = ps.series_colors(series)
        for entry in ps.sorted_series_entries(series):
            eigenvalues = spectra_by_run[entry["run"]].get(step)
            if eigenvalues is None:
                continue
            ranks = np.arange(1, eigenvalues.size + 1, dtype=float)
            scheme = ps.scheme_from_run(entry["run"])
            label = ps.label_from_run(entry["run"])
            line, = raw_ax.plot(
                ranks,
                eigenvalues,
                linewidth=1.2,
                color=colors[entry["run"]],
            )
            legend_entries[scheme].append((line, label))

        raw_ax.set_title("Raw nonzero eigenvalue spectrum")
        raw_ax.set_xlabel("Eigenvalue rank", fontsize=AXIS_LABEL_FONTSIZE)
        raw_ax.set_ylabel(r"Eigenvalue of $C$", fontsize=AXIS_LABEL_FONTSIZE)
        raw_ax.grid(True, which="both", alpha=0.3)
        ps.apply_combined_legend(raw_ax, legend_entries)
        fig.suptitle(f"Superposition nonzero eigenvalue spectrum - {LAYER} - step {step}")
        path = spectrum_dir / f"eigenvalues_step_{step}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    save_trace_vs_step_plot(records, spectra_by_run, spectrum_dir, dpi)
    print(f"Saved {len(steps)} eigenvalue spectrum plots to {spectrum_dir}")


def requested_mode_counts(mode_counts: Sequence[int]) -> tuple[int, ...]:
    modes = tuple(sorted({int(value) for value in mode_counts if int(value) > 0}))
    if not modes:
        raise ValueError("At least one positive spectral reconstruction mode count is required")
    return modes


def spectral_reconstruction_cache_path(base: Path, run_dir: Path) -> Path:
    return cache_dir(base, run_dir) / f"{run_dir.name}_{LAYER}_spectral_reconstruction.json"


def spectral_basis_cache_path(base: Path, run_dir: Path) -> Path:
    return cache_dir(base, run_dir) / f"{run_dir.name}_{LAYER}_final_spectral_basis.npz"


def load_spectral_reconstruction_cache(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        record = json.load(file)
    if not isinstance(record, dict):
        raise ValueError(f"Expected an object in {path}")
    record["steps"] = [int(step) for step in record.get("steps", [])]
    record["time_indices"] = [int(index) for index in record.get("time_indices", [])]
    record["mode_counts"] = [int(value) for value in record.get("mode_counts", [])]
    record["job_dir"] = path.parent.parent
    record["cache_path"] = path
    record["basis_cache_path"] = path.parent / str(
        record.get("basis_cache_file", path.name.replace("_spectral_reconstruction.json", "_final_spectral_basis.npz"))
    )
    return record


def save_spectral_reconstruction_cache(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    basis_cache_path = Path(record["basis_cache_path"])
    payload = {
        "run": record["run"],
        "steps": [int(step) for step in record["steps"]],
        "time_indices": [int(index) for index in record["time_indices"]],
        "original_loss": [float(value) for value in record["original_loss"]],
        "input_dim": int(record["input_dim"]),
        "width": int(record["width"]),
        "batch_size": float(record["batch_size"]),
        "total_params": float(record["total_params"]),
        "mode_counts": [int(value) for value in record["mode_counts"]],
        "losses": {
            str(int(mode)): [float(value) for value in values]
            for mode, values in record["losses"].items()
        },
        "diagnostics": record.get("diagnostics", {}),
        "basis_cache_file": basis_cache_path.name,
        "basis_step": int(record["basis_step"]),
        "matrix": "raw Dense_0 spectral reconstruction from W = U Sigma V^T",
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def spectral_cache_matches(
    record: dict[str, Any],
    times: Sequence[tuple[int, int]],
    mode_counts: Sequence[int],
) -> bool:
    expected = [(int(time_index), int(step)) for time_index, step in times]
    cached = list(zip(record.get("time_indices", []), record.get("steps", [])))
    return cached == expected and tuple(record.get("mode_counts", [])) == tuple(mode_counts)


def save_spectral_basis_cache(
    path: Path,
    *,
    step: int,
    time_index: int,
    mode_counts: Sequence[int],
    row_norms: np.ndarray,
    u: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        step=np.asarray(int(step), dtype=np.int64),
        time_index=np.asarray(int(time_index), dtype=np.int64),
        mode_counts=np.asarray([int(value) for value in mode_counts], dtype=np.int64),
        row_norms=np.asarray(row_norms, dtype=np.float32),
        u=np.asarray(u, dtype=np.float32),
        singular_values=np.asarray(singular_values, dtype=np.float32),
        vt=np.asarray(vt, dtype=np.float32),
    )


def load_spectral_basis_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "step": int(data["step"].item()),
            "time_index": int(data["time_index"].item()),
            "mode_counts": np.asarray(data["mode_counts"], dtype=np.int64),
            "row_norms": np.asarray(data["row_norms"], dtype=np.float32),
            "u": np.asarray(data["u"], dtype=np.float32),
            "singular_values": np.asarray(data["singular_values"], dtype=np.float32),
            "vt": np.asarray(data["vt"], dtype=np.float32),
        }


def basis_cache_is_usable(
    path: Path,
    *,
    step: int,
    max_modes: int,
) -> bool:
    if not path.exists():
        return False
    try:
        data = load_spectral_basis_cache(path)
    except Exception:
        return False
    return int(data["step"]) == int(step) and int(data["u"].shape[1]) >= int(max_modes)


def compute_final_spectral_basis_cache(
    zarr_path: Path,
    output_base: Path,
    *,
    final_time_index: int,
    final_step: int,
    mode_counts: Sequence[int],
    max_modes: int,
) -> Path:
    run_dir = zarr_path.parent
    basis_path = spectral_basis_cache_path(output_base, run_dir)
    if basis_cache_is_usable(basis_path, step=final_step, max_modes=max_modes):
        return basis_path
    dense0_dir, dense0_meta, _ = dense0_kernel(zarr_path)
    dense0 = ps.read_snapshot(dense0_dir, dense0_meta, int(final_time_index))
    row_norms, u, singular_values, vt, _ = ps.spectral_basis_for_dense0(
        dense0,
        max_modes,
        raw_analysis=True,
    )
    save_spectral_basis_cache(
        basis_path,
        step=int(final_step),
        time_index=int(final_time_index),
        mode_counts=mode_counts,
        row_norms=row_norms,
        u=u,
        singular_values=singular_values,
        vt=vt,
    )
    print(f"Saved cached final Dense_0 spectral basis to {basis_path}")
    return basis_path


def compute_or_load_spectral_reconstruction(
    zarr_path: Path,
    output_base: Path,
    min_step: int,
    mode_counts: Sequence[int],
    batch_size: int,
    data_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]],
    *,
    force: bool,
) -> dict[str, Any] | None:
    run_dir = zarr_path.parent
    try:
        dense0_dir, dense0_meta, dense0_shape = dense0_kernel(zarr_path)
        dense1_dir, dense1_meta, dense1_shape = dense1_kernel(zarr_path)
    except FileNotFoundError as exc:
        print(f"Skipping spectral reconstruction for {zarr_path}: {exc}")
        return None

    requested_modes = requested_mode_counts(mode_counts)
    times = selected_analysis_times(
        dense0_shape[0],
        ps.training_step_stride(run_dir),
        min_step,
    )
    times = [(time_index, step) for time_index, step in times if time_index < dense1_shape[0]]
    if not times:
        print(f"Skipping spectral reconstruction for {zarr_path}: no selected times")
        return None

    max_modes = min(max(requested_modes), dense0_shape[1], dense0_shape[2])
    cache_path = spectral_reconstruction_cache_path(output_base, run_dir)
    if cache_path.exists() and not force:
        cached = load_spectral_reconstruction_cache(cache_path)
        if spectral_cache_matches(cached, times, requested_modes):
            basis_path = compute_final_spectral_basis_cache(
                zarr_path,
                output_base,
                final_time_index=int(cached["time_indices"][-1]),
                final_step=int(cached["steps"][-1]),
                mode_counts=requested_modes,
                max_modes=max_modes,
            )
            cached["basis_cache_path"] = basis_path
            return cached
        print(f"Recomputing stale Dense_0 spectral reconstruction cache: {cache_path}")

    config = ps.load_yaml(run_dir / "simulation_config.yaml")
    network = config.get("network", {})
    training = config.get("training", {})
    activations = network.get("activations_per_layer", {})
    activation0 = ps.numpy_activation(str(activations.get("Dense_0", "relu")))
    activation1 = ps.numpy_activation(str(activations.get("Dense_1", "identity")))
    loss_type = str(training.get("loss_type", "mse"))
    x_test, y_test = ps.load_reconstruction_eval_data(run_dir, data_cache)

    record: dict[str, Any] = {
        "job_dir": job_output_dir(output_base, run_dir),
        "run": run_dir.name,
        "steps": [],
        "time_indices": [],
        "original_loss": [],
        "input_dim": int(dense0_shape[1]),
        "width": int(dense0_shape[2]),
        "batch_size": float(training.get("batch_size", 1.0)),
        "total_params": float(network.get("total_params", 1.0)),
        "mode_counts": list(requested_modes),
        "losses": {str(k): [] for k in requested_modes},
        "diagnostics": {str(k): [] for k in requested_modes},
        "cache_path": cache_path,
        "basis_cache_path": spectral_basis_cache_path(output_base, run_dir),
    }

    last_basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    last_step = 0
    last_time_index = 0
    for time_index, step in times:
        dense0 = ps.read_snapshot(dense0_dir, dense0_meta, int(time_index))
        dense1 = ps.read_snapshot(dense1_dir, dense1_meta, int(time_index))
        row_norms, u, singular_values, vt, _ = ps.spectral_basis_for_dense0(
            dense0,
            max_modes,
            raw_analysis=True,
        )
        original_loss = ps.evaluate_dense_two_layer_loss(
            x_test,
            y_test,
            dense0,
            dense1,
            activation0=activation0,
            activation1=activation1,
            loss_type=loss_type,
            batch_size=batch_size,
        )
        record["steps"].append(int(step))
        record["time_indices"].append(int(time_index))
        record["original_loss"].append(float(original_loss))
        for k in requested_modes:
            loss, diagnostics = ps.evaluate_reconstructed_loss(
                x_test,
                y_test,
                dense1,
                row_norms=row_norms,
                u=u,
                singular_values=singular_values,
                vt=vt,
                modes=int(k),
                renormalize=False,
                activation0=activation0,
                activation1=activation1,
                loss_type=loss_type,
                batch_size=batch_size,
            )
            record["losses"][str(k)].append(float(loss))
            record["diagnostics"][str(k)].append(diagnostics)
        last_basis = (row_norms, u, singular_values, vt)
        last_step = int(step)
        last_time_index = int(time_index)

    if last_basis is None:
        return None
    row_norms, u, singular_values, vt = last_basis
    record["basis_step"] = int(last_step)
    save_spectral_basis_cache(
        Path(record["basis_cache_path"]),
        step=last_step,
        time_index=last_time_index,
        mode_counts=requested_modes,
        row_norms=row_norms,
        u=u,
        singular_values=singular_values,
        vt=vt,
    )
    save_spectral_reconstruction_cache(cache_path, record)
    print(f"Saved cached Dense_0 spectral reconstruction analysis to {cache_path}")
    return record


def save_spectral_reconstruction_loss_plot(records: Sequence[dict[str, Any]], dpi: int) -> None:
    records = [record for record in records if record is not None]
    if not records:
        return
    job_dir = Path(records[0]["job_dir"])
    out_dir = job_dir / f"spectral_reconstruction_{LAYER}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_counts = [int(value) for value in records[0]["mode_counts"]]
    ncols = min(2, max(1, len(mode_counts)))
    nrows = int(np.ceil(len(mode_counts) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.4 * ncols, 4.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    colors, legend_entries = ps.series_colors(list(records))
    legend_runs: set[str] = set()

    for panel_index, modes in enumerate(mode_counts):
        ax = axes_flat[panel_index]
        panel_values: list[np.ndarray] = []
        panel_steps: list[np.ndarray] = []
        for record in ps.sorted_series_entries(list(records)):
            steps = np.asarray(record["steps"], dtype=float)
            reconstructed = np.asarray(record["losses"][str(int(modes))], dtype=float)
            original = np.asarray(record["original_loss"], dtype=float)
            count = min(steps.size, reconstructed.size, original.size)
            if count < 1:
                continue
            steps = steps[:count]
            reconstructed = reconstructed[:count]
            original = original[:count]
            line, = ax.plot(
                steps,
                reconstructed,
                color=colors[record["run"]],
                linewidth=1.2,
                marker="o",
                markersize=2.5,
            )
            ax.plot(
                steps,
                original,
                color=colors[record["run"]],
                linewidth=0.9,
                linestyle="--",
                alpha=0.65,
            )
            panel_values.extend([reconstructed, original])
            panel_steps.append(steps)
            if record["run"] not in legend_runs:
                legend_entries[ps.scheme_from_run(record["run"])].append(
                    (line, ps.label_from_run(record["run"]))
                )
                legend_runs.add(record["run"])
        ps.set_nonnegative_loglike_xscale(ax, panel_steps)
        ps.set_padded_ylim(ax, panel_values)
        ax.set_title(f"Unnormalized, k={int(modes)}")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Test loss")
        ax.grid(True, which="both", alpha=0.3)

    for ax in axes_flat[len(mode_counts):]:
        ax.set_visible(False)

    fig.suptitle(
        f"{LAYER} raw-weight spectral reconstruction test loss\n"
        f"solid: reconstructed {LAYER}, dashed: original checkpoint"
    )
    handles, labels, header_indices = ps.combined_legend(legend_entries)
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            frameon=False,
        )
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        for index in header_indices:
            legend.get_texts()[index].set_fontweight("bold")
    path = out_dir / f"test_loss_spectral_reconstruction_{LAYER}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved spectral reconstruction test-loss plot to {path}")

    payload = []
    for record in ps.sorted_series_entries(list(records)):
        payload.append(
            {
                "run": record["run"],
                "steps": [int(step) for step in record["steps"]],
                "time_indices": [int(index) for index in record["time_indices"]],
                "original_loss": [float(value) for value in record["original_loss"]],
                "input_dim": int(record["input_dim"]),
                "width": int(record["width"]),
                "batch_size": float(record["batch_size"]),
                "total_params": float(record["total_params"]),
                "mode_counts": [int(value) for value in record["mode_counts"]],
                "losses": record["losses"],
                "diagnostics": record.get("diagnostics", {}),
                "basis_cache_file": Path(record["basis_cache_path"]).name,
            }
        )
    with (out_dir / "spectral_reconstruction_results.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_spectral_reconstruction_input_space_frames(
    zarr_path: Path,
    output_base: Path,
    record: dict[str, Any],
    *,
    dpi: int,
    force: bool,
) -> int:
    run_dir = zarr_path.parent
    basis_path = Path(record["basis_cache_path"])
    if not basis_path.exists():
        print(f"Skipping spectral reconstruction input-space frames for {run_dir.name}: missing {basis_path}")
        return 0
    basis = load_spectral_basis_cache(basis_path)
    row_norms = basis["row_norms"]
    layout = ps.infer_image_layout(run_dir, int(row_norms.size))
    if layout is None:
        print(
            f"Skipping spectral reconstruction input-space frames for {run_dir.name}: "
            f"could not infer image layout for {row_norms.size} units"
        )
        return 0

    reference_units = ps.selected_reference_units(int(row_norms.size), 3, layout=layout)
    out_dir = spectral_reconstruction_output_dir(output_base, run_dir) / "input_space_frames" / run_dir.name
    saved = 0
    for modes in record["mode_counts"]:
        output_path = out_dir / f"input_space_spectral_k_{int(modes):03d}_step_{int(basis['step'])}.png"
        if output_path.exists() and not force:
            continue
        reconstructed, _ = ps.spectral_reconstructed_dense0_kernel(
            basis["row_norms"],
            basis["u"],
            basis["singular_values"],
            basis["vt"],
            int(modes),
        )
        matrix = input_overlap_matrix(reconstructed)
        plot_input_space_frame(
            matrix,
            reference_units,
            layout,
            output_path,
            title=f"Training Step: {int(basis['step'])}",
            channel_mode="split",
            dpi=dpi,
        )
        saved += 1
    print(f"Saved {saved} spectral reconstruction input-space frames to {out_dir}")
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Dense_0 input-space superposition matrices "
            "C = W W^T / n from weights.zarr runs."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures_analyze_superposition"))
    parser.add_argument(
        "--min-step",
        type=int,
        default=100,
        help=(
            "Minimum nonzero training step for the logarithmic schedule. "
            "Step 0 and the final saved training step are always included."
        ),
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--force", action="store_true", help="Recompute cached analysis and overwrite plots.")
    parser.add_argument(
        "--skip-input-space",
        action="store_true",
        help="Skip the standard superposition input-space frame plots.",
    )
    parser.add_argument(
        "--skip-spectral-reconstruction",
        action="store_true",
        help="Skip Dense_0 raw-weight spectral reconstruction test-loss analysis.",
    )
    parser.add_argument(
        "--skip-spectral-input-space",
        action="store_true",
        help="Skip spectral reconstruction input-space frame plots.",
    )
    parser.add_argument(
        "--spectral-reconstruction-modes",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100],
        help="Mode counts used for Dense_0 spectral reconstruction.",
    )
    parser.add_argument(
        "--spectral-reconstruction-batch-size",
        type=int,
        default=1024,
        help="Evaluation batch size for spectral reconstruction test-loss computation.",
    )
    parser.add_argument(
        "--input-space-reference-count",
        type=int,
        default=3,
        help="Number of deterministic random input coordinates shown as image-space columns.",
    )
    parser.add_argument(
        "--input-space-channel-mode",
        choices=("split", "reference"),
        default="split",
    )
    parser.add_argument(
        "--best-input-space",
        action="store_true",
        help="Also save one Dense_0 input-space superposition frame at each run's best test-loss checkpoint.",
    )
    parser.add_argument(
        "--best-input-space-only",
        action="store_true",
        help="Only save best-test-loss input-space frames; skip the other plots and analyses after loading caches.",
    )
    parser.add_argument(
        "--best-input-space-colorbar-factors",
        type=float,
        nargs="+",
        default=[1.0],
        help="Positive factors multiplying the off-diagonal color limits for best-test-loss frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_space_reference_count < 1:
        raise ValueError("--input-space-reference-count must be positive")
    if args.spectral_reconstruction_batch_size < 1:
        raise ValueError("--spectral-reconstruction-batch-size must be positive")
    if any(float(value) <= 0.0 for value in args.best_input_space_colorbar_factors):
        raise ValueError("--best-input-space-colorbar-factors values must be positive")

    stores = ps.zarr_paths(args.input_dir.resolve())
    if not stores:
        raise FileNotFoundError(f"No weights.zarr directories found under {args.input_dir}")

    records = [
        compute_or_load_run_analysis(store, args.output_dir, args.min_step, force=args.force)
        for store in stores
    ]
    records = sorted(records, key=lambda item: tuple(ps.width_key(ps.label_from_run(item["run"]))))
    if args.best_input_space_only:
        best_count = 0
        for store in stores:
            best_count += save_best_input_space_frames(
                store,
                args.output_dir,
                reference_count=args.input_space_reference_count,
                channel_mode=args.input_space_channel_mode,
                dpi=args.dpi,
                force=args.force,
                colorbar_factors=args.best_input_space_colorbar_factors,
            )
        print(
            f"Completed best-test-loss input-space frame generation for "
            f"{len(stores)} runs; saved {best_count} frames."
        )
        return

    write_cache_manifest(records, args.output_dir, args.input_dir)
    save_frobenius_plot(records, args.dpi)
    save_eigenvalue_spectrum_plots(records, args.dpi)

    frame_count = 0
    if not args.skip_input_space:
        by_run = {record["run"]: record for record in records}
        for store in stores:
            record = by_run[store.parent.name]
            frame_count += save_input_space_frames(
                store,
                args.output_dir,
                record,
                reference_count=args.input_space_reference_count,
                channel_mode=args.input_space_channel_mode,
                dpi=args.dpi,
                force=args.force,
            )
    print(f"Completed Dense_0 superposition analysis for {len(records)} runs; saved {frame_count} new frames.")

    if args.best_input_space:
        best_count = 0
        for store in stores:
            best_count += save_best_input_space_frames(
                store,
                args.output_dir,
                reference_count=args.input_space_reference_count,
                channel_mode=args.input_space_channel_mode,
                dpi=args.dpi,
                force=args.force,
                colorbar_factors=args.best_input_space_colorbar_factors,
            )
        print(f"Saved {best_count} best-test-loss input-space frame variants.")

    if not args.skip_spectral_reconstruction:
        data_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]] = {}
        spectral_records = []
        for store in stores:
            record = compute_or_load_spectral_reconstruction(
                store,
                args.output_dir,
                args.min_step,
                args.spectral_reconstruction_modes,
                args.spectral_reconstruction_batch_size,
                data_cache,
                force=args.force,
            )
            if record is not None:
                spectral_records.append(record)
        save_spectral_reconstruction_loss_plot(spectral_records, args.dpi)
        spectral_frame_count = 0
        if not args.skip_spectral_input_space:
            by_run = {record["run"]: record for record in spectral_records}
            for store in stores:
                record = by_run.get(store.parent.name)
                if record is None:
                    continue
                spectral_frame_count += save_spectral_reconstruction_input_space_frames(
                    store,
                    args.output_dir,
                    record,
                    dpi=args.dpi,
                    force=args.force,
                )
        print(
            f"Completed Dense_0 spectral reconstruction analysis for "
            f"{len(spectral_records)} runs; saved {spectral_frame_count} new spectral frames."
        )


if __name__ == "__main__":
    main()
