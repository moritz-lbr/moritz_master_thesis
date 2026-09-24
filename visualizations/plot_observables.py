"""Plot test loss and update-norm observables across one or more jobs (Figs. 5.1-5.3).

This is a small replacement for the update-norm plotting workflow when only the
loss and update-norm figures are needed.  Runs are grouped by hidden width.  If
multiple jobs contain the same width, the mean is plotted and the across-run
standard deviation is shown as a shaded band.  A single run is plotted raw.

    # Fig. 5.1 (muP) / Fig. 5.3 top (SP): raw and rescaled layer-wise update norms
    python visualizations/plot_observables.py update-norms --log-dirs <job-dirs> --output <png>
    # Fig. 5.2 (muP) / Fig. 5.3 bottom (SP): test loss averaged over seeds, with zoom
    python visualizations/plot_observables.py test-loss --log-dirs <job-dirs> --output <png>

Update norms require ``training_analysis/compute_metrics.py`` to have been run first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle


LAYERS = ("Dense_0", "Dense_1")
FINAL_PARAMS_MARKER = b'"final_params"'
DEFAULT_TITLE_TEMPLATE = (
    r"$\tau = {tau}$ SGD Update Steps are Conducted Between Subsequent Data Points "
    r"$t_{{i+1}}$ and $t_i$"
)


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    width: int
    scheme: str
    label: str
    training: dict[str, Any]
    network: dict[str, Any]
    test_loss: np.ndarray


@dataclass(frozen=True)
class AggregatedSeries:
    x_axis: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    num_runs: int


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_final_metrics(path: Path) -> dict[str, Any]:
    """Load the metrics block without reading large final parameter arrays."""
    buffer = bytearray()
    prefix: str | None = None

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            buffer.extend(chunk)
            marker_index = buffer.find(FINAL_PARAMS_MARKER)
            if marker_index != -1:
                prefix = bytes(buffer[:marker_index]).decode("utf-8")
                break

    if prefix is None:
        data = _load_json(path)
    else:
        data = json.loads(prefix.rsplit(",", maxsplit=1)[0].rstrip() + "\n}")

    for key in ("final_metrics", "avg_final_metrics"):
        metrics = data.get(key)
        if isinstance(metrics, dict):
            return metrics
    raise ValueError(f"No final metrics block found in {path}")


def _load_training_metrics(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    for key in ("training_metrics", "avg_training_metrics"):
        metrics = data.get(key)
        if isinstance(metrics, dict):
            return metrics
    raise ValueError(f"No training metrics block found in {path}")


def _width_from_run_dir(run_dir: Path) -> int:
    match = re.search(r"widths-(\d+)x(\d+)", run_dir.name)
    if not match:
        raise ValueError(f"Cannot parse width from {run_dir}")
    return int(match.group(1))


def _scheme_from_run_dir(run_dir: Path) -> str:
    if "standard" in run_dir.name:
        return "SP"
    if "muP" in run_dir.name:
        return "muP"
    return "unknown"


def _find_run_dirs(log_dirs: Iterable[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    for log_dir in log_dirs:
        if not log_dir.exists():
            raise FileNotFoundError(log_dir)
        for log_path in sorted(log_dir.rglob("training_log.json")):
            if "widths-" in log_path.parent.name:
                run_dirs.append(log_path.parent)
    if not run_dirs:
        raise FileNotFoundError("No training_log.json files found under the log dirs.")
    return sorted(run_dirs, key=lambda path: (_width_from_run_dir(path), str(path)))


def _load_run(run_dir: Path) -> RunInfo:
    config_path = run_dir / "simulation_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = _load_yaml(config_path)
    training = config.get("training")
    network = config.get("network")
    if not isinstance(training, dict) or not isinstance(network, dict):
        raise ValueError(f"Missing training/network config in {config_path}")

    final_metrics = _load_final_metrics(run_dir / "training_log.json")
    history = final_metrics.get("history")
    if not isinstance(history, dict) or not history.get("test_loss"):
        raise ValueError(f"Missing test_loss history in {run_dir / 'training_log.json'}")

    width = _width_from_run_dir(run_dir)
    return RunInfo(
        run_dir=run_dir,
        width=width,
        scheme=_scheme_from_run_dir(run_dir),
        label=str(width),
        training=training,
        network=network,
        test_loss=np.asarray(history["test_loss"], dtype=float),
    )


def _save_loss_frequency(training: dict[str, Any]) -> float:
    value = training.get("save_loss_frequency")
    if value == "epoch":
        return 1.0
    return float(value)


def _x_axis(
    run: RunInfo,
    length: int,
    x_axis_kind: str,
    *,
    first_index: float = 1.0,
) -> np.ndarray:
    save_freq = _save_loss_frequency(run.training)
    x_axis = (np.arange(length, dtype=float) + first_index) * save_freq
    if x_axis_kind == "steps":
        return x_axis

    batch_size = float(run.training["batch_size"])
    total_params = float(run.network["total_params"])
    return x_axis * batch_size * total_params


def _weight_metrics_path(run_dir: Path, layer: str) -> Path:
    for directory in ("weight_metrics", "avg_final_weight_metrics"):
        path = run_dir / directory / f"{layer}.json"
        if path.exists():
            return path
    raise FileNotFoundError(f"No weight metrics for {layer} in {run_dir}")


def _step_norms(run: RunInfo, layer: str) -> np.ndarray:
    metrics = _load_training_metrics(_weight_metrics_path(run.run_dir, layer))
    values = metrics.get("step_norms")
    if not values:
        raise ValueError(f"Missing step_norms for {layer} in {run.run_dir}")
    return np.asarray(values, dtype=float)


def _width_ratio(run: RunInfo) -> float:
    nodes = run.network.get("nodes_per_layer")
    base_widths = run.network.get("base_layer_width") or run.network.get(
        "base_layer_widths"
    )
    if not isinstance(nodes, dict) or "Dense_0" not in nodes:
        raise ValueError(f"Missing Dense_0 width in {run.run_dir}")

    if isinstance(base_widths, dict):
        base_width = base_widths.get("Dense_0")
    else:
        base_width = base_widths
    if base_width is None:
        raise ValueError(f"Missing base width in {run.run_dir}")

    return float(nodes["Dense_0"]) / float(base_width)


def _scaled_update_norms(run: RunInfo, layer: str, values: np.ndarray) -> np.ndarray:
    ratio = _width_ratio(run)
    scale = 1.0 / ratio if layer == "Dense_0" else ratio
    return values * scale


def _filter_runs(
    runs: list[RunInfo],
    excluded_widths: set[int],
    exclude_largest: bool,
) -> list[RunInfo]:
    widths = {run.width for run in runs}
    if exclude_largest and widths:
        excluded_widths = set(excluded_widths)
        excluded_widths.add(max(widths))
    return [run for run in runs if run.width not in excluded_widths]


def _group_by_width(runs: list[RunInfo]) -> dict[int, list[RunInfo]]:
    grouped: dict[int, list[RunInfo]] = {}
    for run in runs:
        grouped.setdefault(run.width, []).append(run)
    return dict(sorted(grouped.items()))


def _aggregate_runs(
    grouped_runs: dict[int, list[RunInfo]],
    values_for_run: Any,
    x_axis_kind: str,
    *,
    first_index: float = 1.0,
) -> dict[int, AggregatedSeries]:
    aggregated: dict[int, AggregatedSeries] = {}
    for width, width_runs in grouped_runs.items():
        values = [np.asarray(values_for_run(run), dtype=float) for run in width_runs]
        min_len = min(len(value) for value in values)
        x_axes = [
            _x_axis(run, min_len, x_axis_kind, first_index=first_index)
            for run in width_runs
        ]
        x_ref = x_axes[0]
        for run, x_axis in zip(width_runs[1:], x_axes[1:]):
            if not np.allclose(x_ref, x_axis, rtol=1e-12, atol=0.0):
                raise ValueError(
                    f"x-axis mismatch for width {width}: {width_runs[0].run_dir} "
                    f"and {run.run_dir}"
                )

        stacked = np.vstack([value[:min_len] for value in values])
        if len(width_runs) == 1:
            std = np.zeros_like(stacked[0])
        else:
            std = np.nanstd(stacked, axis=0, ddof=1)
        aggregated[width] = AggregatedSeries(
            x_axis=x_ref,
            mean=np.nanmean(stacked, axis=0),
            std=std,
            num_runs=len(width_runs),
        )
    return aggregated


def _infer_parametrization(runs: list[RunInfo], fallback: str | None = None) -> str:
    schemes = {run.scheme for run in runs if run.scheme != "unknown"}
    if len(schemes) == 1:
        return next(iter(schemes))
    if fallback:
        return fallback
    if not schemes:
        return "run"
    raise ValueError(f"Mixed parametrizations found: {sorted(schemes)}")


def _colors(parametrization: str, count: int) -> np.ndarray:
    cmap = plt.cm.winter if parametrization == "muP" else plt.cm.autumn
    return cmap(np.linspace(0, 1, max(count, 1)))


def _plot_series(
    ax: Any,
    series: AggregatedSeries,
    color: Any,
    label: str | None = None,
) -> Any:
    (line,) = ax.plot(series.x_axis, series.mean, color=color, linewidth=1.4, label=label)
    if series.num_runs > 1:
        lower = np.maximum(series.mean - series.std, np.finfo(float).tiny)
        ax.fill_between(
            series.x_axis,
            lower,
            series.mean + series.std,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    return line


def _format_tau(run: RunInfo) -> str:
    tau = _save_loss_frequency(run.training)
    if float(tau).is_integer():
        return str(int(tau))
    return f"{tau:g}"


def _legend_title(parametrization: str, explicit_title: str | None) -> str:
    if explicit_title:
        return explicit_title
    return f"width (n) - {parametrization}"


def _loss_defaults(parametrization: str, x_axis_kind: str) -> dict[str, tuple[float, float]]:
    if x_axis_kind == "compute":
        return {
            "main_xlim": (5e10, 1.8e15),
            "zoom_xlim": (5e12, 1.6e15),
            "main_ylim": (1.28, 2.15),
            "zoom_ylim": (1.28, 1.51),
        }
    if parametrization == "muP":
        return {
            "main_xlim": (80.0, 2.5e4),
            "zoom_xlim": (4e3, 2.2e4),
            "main_ylim": (1.28, 2.15),
            "zoom_ylim": (1.28, 1.51),
        }
    return {
        "main_xlim": (80.0, 2.5e4),
        "zoom_xlim": (4e3, 2.2e4),
        "main_ylim": (1.30, 1.91),
        "zoom_ylim": (1.31, 1.44),
    }


def _auto_loss_limits(aggregated: dict[int, AggregatedSeries]) -> dict[str, tuple[float, float]]:
    """Derive main/zoom axis limits from the data (zoom = last 35% of the log time axis)."""
    x_all = np.concatenate([series.x_axis for series in aggregated.values()])
    y_all = np.concatenate([series.mean for series in aggregated.values()])
    valid = np.isfinite(x_all) & np.isfinite(y_all) & (x_all > 0) & (y_all > 0)
    x_all, y_all = x_all[valid], y_all[valid]
    x_min, x_max = float(x_all.min()), float(x_all.max())
    log_min, log_max = np.log10(x_min), np.log10(x_max)
    zoom_start = 10 ** (log_max - 0.35 * (log_max - log_min))
    zoom_y = y_all[x_all >= zoom_start]
    if zoom_y.size == 0:
        zoom_y = y_all
    return {
        "main_xlim": (x_min / 1.1, x_max * 1.1),
        "zoom_xlim": (zoom_start, x_max * 1.05),
        "main_ylim": (float(y_all.min()) / 1.02, float(y_all.max()) * 1.02),
        "zoom_ylim": (float(zoom_y.min()) / 1.005, float(zoom_y.max()) * 1.005),
    }


def _axes_for_loss(legend_position: str) -> tuple[Any, Any, Any]:
    fig = plt.figure(figsize=(13.57, 4.84), dpi=100)
    if legend_position == "right":
        return fig, fig.add_axes([0.12, 0.16, 0.305, 0.53]), fig.add_axes(
            [0.53, 0.16, 0.305, 0.53]
        )
    return fig, fig.add_axes([0.27, 0.16, 0.305, 0.53]), fig.add_axes(
        [0.68, 0.16, 0.305, 0.53]
    )


def plot_test_loss(args: argparse.Namespace) -> None:
    runs = [_load_run(path) for path in _find_run_dirs(args.log_dirs)]
    runs = _filter_runs(runs, set(args.exclude_width), args.exclude_largest)
    if not runs:
        raise ValueError("No runs left after width filtering.")

    parametrization = _infer_parametrization(runs, args.parametrization)
    grouped = _group_by_width(runs)
    aggregated = _aggregate_runs(grouped, lambda run: run.test_loss, args.x_axis)
    widths = sorted(aggregated)
    colors = _colors(parametrization, len(widths))
    if args.limits == "thesis":
        defaults = _loss_defaults(parametrization, args.x_axis)
    else:
        defaults = _auto_loss_limits(aggregated)

    main_xlim = tuple(args.main_xlim or defaults["main_xlim"])
    zoom_xlim = tuple(args.zoom_xlim or defaults["zoom_xlim"])
    main_ylim = tuple(args.main_ylim or defaults["main_ylim"])
    zoom_ylim = tuple(args.zoom_ylim or defaults["zoom_ylim"])

    fig, ax, ax_zoom = _axes_for_loss(args.legend_position)
    handles = []
    labels = []
    for width, color in zip(widths, colors):
        line = _plot_series(ax, aggregated[width], color)
        _plot_series(ax_zoom, aggregated[width], color)
        handles.append(line)
        labels.append(str(width))

    for current_ax in (ax, ax_zoom):
        current_ax.set_xscale("log")
        current_ax.set_yscale("log")
        current_ax.grid(True, which="both", alpha=0.35)
        current_ax.set_ylabel("Test Loss [log]")
        current_ax.set_xlabel(
            "Training Time [log]"
            if args.x_axis == "compute"
            else r"Training Steps $t_i$ [log]"
        )

    ax.set_xlim(*main_xlim)
    ax.set_ylim(*main_ylim)
    ax_zoom.set_xlim(*zoom_xlim)
    ax_zoom.set_ylim(*zoom_ylim)
    ax_zoom.set_title("zoom")
    ax.add_patch(
        Rectangle(
            (zoom_xlim[0], zoom_ylim[0]),
            zoom_xlim[1] - zoom_xlim[0],
            zoom_ylim[1] - zoom_ylim[0],
            fill=False,
            edgecolor="black",
            linestyle="--",
            linewidth=1.5,
            zorder=20,
        )
    )

    legend_anchor = (0.89, 0.48) if args.legend_position == "right" else (0.015, 0.48)
    legend = fig.legend(
        handles,
        labels,
        title=_legend_title(parametrization, args.legend_title),
        loc="center left",
        bbox_to_anchor=legend_anchor,
        frameon=True,
        handlelength=1.4,
        borderaxespad=0.0,
    )
    legend.get_title().set_fontweight("bold")

    title = args.title or DEFAULT_TITLE_TEMPLATE.format(tau=_format_tau(runs[0]))
    fig.suptitle(title, fontsize=21, y=0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=100)
    plt.close(fig)
    _print_summary(args.output, aggregated)


def _top_update_ylabel(layer: str) -> str:
    layer_index = layer[-1]
    return rf"$\|u_{{t,\tau}}^{{{layer_index}}}\|^2(mn)\ [\log]$"


def _bottom_update_ylabel(layer: str) -> str:
    layer_index = layer[-1]
    return (
        rf"$\frac{{\|u_{{t,\tau}}^{{{layer_index}}}\|^2(mn)}}"
        rf"{{c^{{{layer_index}}}(m)}}\ [\log]$"
    )


def _update_norm_values(layer: str, scaled: bool) -> Any:
    def values(run: RunInfo) -> np.ndarray:
        raw_values = _step_norms(run, layer)
        if scaled:
            return _scaled_update_norms(run, layer, raw_values)
        return raw_values

    return values


def plot_update_norms(args: argparse.Namespace) -> None:
    runs = [_load_run(path) for path in _find_run_dirs(args.log_dirs)]
    runs = _filter_runs(runs, set(args.exclude_width), args.exclude_largest)
    if not runs:
        raise ValueError("No runs left after width filtering.")

    parametrization = _infer_parametrization(runs, args.parametrization)
    grouped = _group_by_width(runs)
    top = {
        layer: _aggregate_runs(
            grouped,
            _update_norm_values(layer, scaled=False),
            args.x_axis,
            first_index=0.0,
        )
        for layer in LAYERS
    }
    show_scaled = args.show_scaled and parametrization == "muP"
    bottom = {
        layer: _aggregate_runs(
            grouped,
            _update_norm_values(layer, scaled=True),
            args.x_axis,
            first_index=0.0,
        )
        for layer in LAYERS
    } if show_scaled else {}

    widths = sorted(top[LAYERS[0]])
    colors = _colors(parametrization, len(widths))
    if show_scaled:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        top_axes = (axes[0, 0], axes[0, 1])
        bottom_axes = (axes[1, 0], axes[1, 1])
        tight_rect = [0, 0.02, 0.82, 0.9]
    else:
        fig, top_axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
        bottom_axes = (None, None)
        tight_rect = [0, 0.02, 0.82, 0.86]

    handles = []
    labels = []
    for width, color in zip(widths, colors):
        line = _plot_series(top_axes[0], top["Dense_0"][width], color)
        _plot_series(top_axes[1], top["Dense_1"][width], color)
        if show_scaled:
            _plot_series(bottom_axes[0], bottom["Dense_0"][width], color)
            _plot_series(bottom_axes[1], bottom["Dense_1"][width], color)
        handles.append(line)
        labels.append(str(width))

    for column, layer in enumerate(LAYERS):
        top_axes[column].set_title(rf"$l={column}$", fontsize=20)
        top_axes[column].set_ylabel(_top_update_ylabel(layer), fontsize=18)
        if bottom_axes[column] is None:
            top_axes[column].set_xlabel(_x_label(args.x_axis), fontsize=18)
        else:
            bottom_axes[column].set_ylabel(_bottom_update_ylabel(layer), fontsize=20)
            bottom_axes[column].set_xlabel(_x_label(args.x_axis), fontsize=18)

    for ax in [*top_axes, *(axis for axis in bottom_axes if axis is not None)]:
        ax.set_xscale("log", base=10)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(axis="both", labelsize=18)

    legend = fig.legend(
        handles,
        labels,
        title=_legend_title(parametrization, args.legend_title),
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        ncol=1,
        frameon=True,
        handlelength=1.5,
        handletextpad=0.6,
        fontsize=13,
    )
    legend._legend_box.align = "left"  # type: ignore[attr-defined]
    legend.get_title().set_fontweight("bold")

    title = args.title or DEFAULT_TITLE_TEMPLATE.format(tau=_format_tau(runs[0]))
    fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=tight_rect)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    _print_summary(args.output, top[LAYERS[0]])


def _x_label(x_axis_kind: str) -> str:
    return "Training Time [log]" if x_axis_kind == "compute" else r"Training Steps $t_i$ [log]"


def _print_summary(output: Path, aggregated: dict[int, AggregatedSeries]) -> None:
    run_counts = ", ".join(
        f"{width}:{series.num_runs}" for width, series in sorted(aggregated.items())
    )
    widths = ", ".join(str(width) for width in sorted(aggregated))
    print(f"Saved plot to {output}")
    print(f"Widths: {widths}")
    print(f"Runs per width: {run_counts}")


def _parse_pair(values: list[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("Expected exactly two values.")
    return float(values[0]), float(values[1])


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more job directories containing scheme-*widths-* runs.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument(
        "--parametrization",
        choices=("muP", "SP"),
        default=None,
        help="Legend parametrization label. Inferred from run directory names by default.",
    )
    parser.add_argument(
        "--legend-title",
        default=None,
        help="Override the legend title. Default: 'width (n) - <parametrization>'.",
    )
    parser.add_argument(
        "--exclude-width",
        type=int,
        action="append",
        default=[],
        help="Width to exclude. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--exclude-largest",
        action="store_true",
        help="Exclude the largest width found in the selected runs.",
    )
    parser.add_argument("--title", default=None, help="Override the figure title.")
    parser.add_argument(
        "--x-axis",
        choices=("steps", "compute"),
        default="steps",
        help="Use raw training steps or training-time/compute units on the x-axis.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot test loss and update-norm observables grouped by width."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    loss_parser = subparsers.add_parser("test-loss", help="Plot test loss with zoom.")
    _add_common_args(loss_parser)
    loss_parser.add_argument(
        "--legend-position",
        choices=("left", "right"),
        default="left",
        help="Place the legend to the left or right of the two panels.",
    )
    loss_parser.add_argument(
        "--limits",
        choices=("auto", "thesis"),
        default="auto",
        help=(
            "Default axis limits: 'auto' derives them from the data, 'thesis' uses the "
            "fixed limits of the CIFAR-10 figures in the thesis (Figs. 5.2, 5.3). "
            "Explicit --main-xlim/--zoom-xlim/... always take precedence."
        ),
    )
    loss_parser.add_argument("--main-xlim", type=float, nargs=2, default=None)
    loss_parser.add_argument("--zoom-xlim", type=float, nargs=2, default=None)
    loss_parser.add_argument("--main-ylim", type=float, nargs=2, default=None)
    loss_parser.add_argument("--zoom-ylim", type=float, nargs=2, default=None)
    loss_parser.set_defaults(func=plot_test_loss)

    update_parser = subparsers.add_parser(
        "update-norms",
        help="Plot Dense_0/Dense_1 update norms, with muP-scaled panels by default.",
    )
    _add_common_args(update_parser)
    update_parser.add_argument(
        "--show-scaled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show scaled update-norm panels for muP runs.",
    )
    update_parser.set_defaults(func=plot_update_norms)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("main_xlim", "zoom_xlim", "main_ylim", "zoom_ylim"):
        if hasattr(args, name):
            setattr(args, name, _parse_pair(getattr(args, name)))
    args.func(args)


if __name__ == "__main__":
    main()
