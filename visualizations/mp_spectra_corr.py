#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

import analyze_superposition as sup
import plot_superposition as ps


LAYER = sup.LAYER
AXIS_LABEL_FONTSIZE = sup.AXIS_LABEL_FONTSIZE
RECONSTRUCTION_RATIOS = np.asarray(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0],
    dtype=float,
)
TIME_RECONSTRUCTION_RATIOS = np.asarray([0.5, 1.0, 2.0, 5.0], dtype=float)
DETACHED_RATIO_DEFAULT = 1.01
DETACHED_Z_DEFAULT = DETACHED_RATIO_DEFAULT
DETACHED_RULE_NAME = "leading_prefix_real_over_null_mean"


def detached_rule_version(z_threshold: float = DETACHED_RATIO_DEFAULT) -> str:
    return f"{DETACHED_RULE_NAME}_threshold{float(z_threshold):g}_v1"


def stable_run_seed(run: str) -> int:
    digest = hashlib.blake2b(run.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def mp_cache_path(
    output_base: Path,
    run_dir: Path,
    *,
    replicates: int,
    seed: int,
) -> Path:
    return sup.cache_dir(output_base, run_dir) / (
        f"{run_dir.name}_{LAYER}_mp_null_R{int(replicates)}_seed{int(seed)}.npz"
    )


def best_reconstruction_cache_path(
    output_base: Path,
    run_dir: Path,
    *,
    replicates: int,
    seed: int,
    batch_size: int,
) -> Path:
    return sup.cache_dir(output_base, run_dir) / (
        f"{run_dir.name}_{LAYER}_best_mp_detached_reconstruction_"
        f"R{int(replicates)}_seed{int(seed)}_batch{int(batch_size)}.json"
    )


def time_reconstruction_cache_path(
    output_base: Path,
    run_dir: Path,
    *,
    replicates: int,
    seed: int,
    batch_size: int,
) -> Path:
    return sup.cache_dir(output_base, run_dir) / (
        f"{run_dir.name}_{LAYER}_time_mp_detached_reconstruction_"
        f"R{int(replicates)}_seed{int(seed)}_batch{int(batch_size)}.json"
    )


def mp_cache_matches(
    record: dict[str, Any],
    real_record: dict[str, Any],
    *,
    replicates: int,
    seed: int,
) -> bool:
    return (
        int(record["replicates"]) == int(replicates)
        and int(record["seed"]) == int(seed)
        and int(record["input_dim"]) == int(real_record["input_dim"])
        and int(record["width"]) == int(real_record["width"])
        and np.array_equal(np.asarray(record["steps"], dtype=np.int64), real_record["steps"])
        and np.array_equal(np.asarray(record["time_indices"], dtype=np.int64), real_record["time_indices"])
    )


def load_mp_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        detached_rule = (
            str(data["detached_rule"].item())
            if "detached_rule" in data.files
            else ""
        )
        detached_z = (
            float(data["detached_z"].item())
            if "detached_z" in data.files
            else float("nan")
        )
        return {
            "run": str(data["run"].item()),
            "steps": np.asarray(data["steps"], dtype=np.int64),
            "time_indices": np.asarray(data["time_indices"], dtype=np.int64),
            "replicates": int(data["replicates"].item()),
            "seed": int(data["seed"].item()),
            "input_dim": int(data["input_dim"].item()),
            "width": int(data["width"].item()),
            "null_spectra": np.asarray(data["null_spectra"], dtype=float),
            "null_mean": np.asarray(data["null_mean"], dtype=float),
            "null_min": np.asarray(data["null_min"], dtype=float),
            "null_max": np.asarray(data["null_max"], dtype=float),
            "edge": np.asarray(data["edge"], dtype=float),
            "analytic_mp_edge": np.asarray(data["analytic_mp_edge"], dtype=float),
            "detached_counts": np.asarray(data["detached_counts"], dtype=np.int64),
            "detached_rule": detached_rule,
            "detached_z": detached_z,
            "cache_path": path,
            "job_dir": path.parent.parent,
        }


def save_mp_cache(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        run=np.asarray(record["run"]),
        steps=np.asarray(record["steps"], dtype=np.int64),
        time_indices=np.asarray(record["time_indices"], dtype=np.int64),
        replicates=np.asarray(int(record["replicates"]), dtype=np.int64),
        seed=np.asarray(int(record["seed"]), dtype=np.int64),
        input_dim=np.asarray(int(record["input_dim"]), dtype=np.int64),
        width=np.asarray(int(record["width"]), dtype=np.int64),
        null_spectra=np.asarray(record["null_spectra"], dtype=np.float64),
        null_mean=np.asarray(record["null_mean"], dtype=np.float64),
        null_min=np.asarray(record["null_min"], dtype=np.float64),
        null_max=np.asarray(record["null_max"], dtype=np.float64),
        edge=np.asarray(record["edge"], dtype=np.float64),
        analytic_mp_edge=np.asarray(record["analytic_mp_edge"], dtype=np.float64),
        detached_counts=np.asarray(record["detached_counts"], dtype=np.int64),
        detached_rule=np.asarray(
            str(record.get("detached_rule", detached_rule_version(record.get("detached_z", DETACHED_Z_DEFAULT))))
        ),
        detached_z=np.asarray(float(record.get("detached_z", DETACHED_Z_DEFAULT)), dtype=np.float64),
    )


def independently_permute_input_rows(rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    keys = rng.random(rows.shape, dtype=np.float32)
    order = np.argsort(keys, axis=1)
    return np.take_along_axis(rows, order, axis=1)


def nonzero_overlap_eigenvalues(rows: np.ndarray) -> np.ndarray:
    reduced, _ = sup.reduced_overlap_matrix(rows)
    eigenvalues = np.linalg.eigvalsh(reduced)
    return np.maximum(eigenvalues.astype(float), 0.0)[::-1]


def analytic_mp_edge(eigenvalues: np.ndarray, input_dim: int, width: int) -> float:
    if input_dim <= 0 or width <= 0:
        return float("nan")
    sigma_sq = float(np.sum(eigenvalues, dtype=np.float64)) / float(input_dim)
    return sigma_sq * (1.0 + np.sqrt(float(input_dim) / float(width))) ** 2


def detached_mask_from_null_spectrum(
    eigenvalues: np.ndarray,
    null_spectrum: np.ndarray,
    *,
    z_threshold: float,
) -> np.ndarray:
    real = np.asarray(eigenvalues, dtype=float).ravel()
    null = np.asarray(null_spectrum, dtype=float)
    if null.ndim == 2:
        null_mean = np.mean(null, axis=0)
    else:
        null_mean = null.ravel()
    limit = min(real.size, null_mean.size)
    mask = np.zeros(real.size, dtype=bool)
    if limit == 0:
        return mask

    score = np.full(limit, -np.inf, dtype=float)
    numerator = real[:limit]
    denominator = null_mean[:limit]
    finite = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0.0)
    )
    score[finite] = numerator[finite] / denominator[finite]
    singular_positive = (
        np.isfinite(numerator)
        & (numerator > 0.0)
        & np.isfinite(denominator)
        & (denominator == 0.0)
    )
    score[singular_positive] = np.inf
    satisfied = score > float(z_threshold)
    violation_indices = np.flatnonzero(~satisfied)
    prefix_count = int(violation_indices[0]) if violation_indices.size else int(limit)
    mask[:prefix_count] = True
    return mask


def detached_count_from_null_spectrum(
    eigenvalues: np.ndarray,
    null_spectrum: np.ndarray,
    *,
    z_threshold: float,
) -> int:
    return int(np.count_nonzero(detached_mask_from_null_spectrum(
        eigenvalues,
        null_spectrum,
        z_threshold=z_threshold,
    )))


def refresh_detached_counts(
    record: dict[str, Any],
    real_record: dict[str, Any],
    *,
    z_threshold: float,
) -> bool:
    real_spectra = np.asarray(real_record["eigenvalues"], dtype=float)
    null_spectra = np.asarray(record["null_spectra"], dtype=float)
    counts = np.asarray(
        [
            detached_count_from_null_spectrum(
                real_spectrum,
                null_spectrum,
                z_threshold=z_threshold,
            )
            for real_spectrum, null_spectrum in zip(real_spectra, null_spectra)
        ],
        dtype=np.int64,
    )
    changed = (
        str(record.get("detached_rule", "")) != detached_rule_version(z_threshold)
        or not np.isclose(float(record.get("detached_z", np.nan)), float(z_threshold))
        or not np.array_equal(np.asarray(record["detached_counts"], dtype=np.int64), counts)
    )
    record["detached_counts"] = counts
    record["detached_rule"] = detached_rule_version(z_threshold)
    record["detached_z"] = float(z_threshold)
    return changed


def compute_or_load_mp_null(
    zarr_path: Path,
    output_base: Path,
    real_record: dict[str, Any],
    *,
    replicates: int,
    seed: int,
    z_threshold: float,
    force: bool,
) -> dict[str, Any]:
    run_dir = zarr_path.parent
    cache_path = mp_cache_path(output_base, run_dir, replicates=replicates, seed=seed)
    if cache_path.exists() and not force:
        cached = load_mp_cache(cache_path)
        if mp_cache_matches(cached, real_record, replicates=replicates, seed=seed):
            if refresh_detached_counts(cached, real_record, z_threshold=z_threshold):
                save_mp_cache(cache_path, cached)
                print(f"Updated detached-count rule in MP correction cache: {cache_path}")
            return cached
        print(f"Recomputing stale MP correction cache: {cache_path}")

    array_dir, meta, _ = sup.dense0_kernel(zarr_path)
    real_spectra = np.asarray(real_record["eigenvalues"], dtype=float)
    steps = np.asarray(real_record["steps"], dtype=np.int64)
    time_indices = np.asarray(real_record["time_indices"], dtype=np.int64)
    input_dim = int(real_record["input_dim"])
    width = int(real_record["width"])
    rank_count = int(real_spectra.shape[1])

    null_spectra = np.empty((len(steps), int(replicates), rank_count), dtype=np.float64)
    edge = np.empty(len(steps), dtype=np.float64)
    analytic_edges = np.empty(len(steps), dtype=np.float64)
    detached_counts = np.empty(len(steps), dtype=np.int64)
    run_seed = stable_run_seed(str(real_record["run"]))

    for step_index, (time_index, step) in enumerate(zip(time_indices, steps)):
        rows = sup.weight_rows_by_input_unit(ps.read_snapshot(array_dir, meta, int(time_index)))
        for replicate in range(int(replicates)):
            seed_sequence = np.random.SeedSequence([int(seed), run_seed, int(step), int(replicate)])
            rng = np.random.default_rng(seed_sequence)
            permuted = independently_permute_input_rows(rows, rng)
            null_spectra[step_index, replicate] = nonzero_overlap_eigenvalues(permuted)

        edge[step_index] = float(np.max(null_spectra[step_index, :, 0]))
        analytic_edges[step_index] = analytic_mp_edge(real_spectra[step_index], input_dim, width)
        detached_counts[step_index] = detached_count_from_null_spectrum(
            real_spectra[step_index],
            null_spectra[step_index],
            z_threshold=z_threshold,
        )
        print(
            f"{run_dir.name}: step {int(step)} "
            f"edge={edge[step_index]:.6g}, detached={int(detached_counts[step_index])}"
        )

    record = {
        "run": str(real_record["run"]),
        "steps": steps,
        "time_indices": time_indices,
        "replicates": int(replicates),
        "seed": int(seed),
        "input_dim": input_dim,
        "width": width,
        "null_spectra": null_spectra,
        "null_mean": np.mean(null_spectra, axis=1),
        "null_min": np.min(null_spectra, axis=1),
        "null_max": np.max(null_spectra, axis=1),
        "edge": edge,
        "analytic_mp_edge": analytic_edges,
        "detached_counts": detached_counts,
        "detached_rule": detached_rule_version(z_threshold),
        "detached_z": float(z_threshold),
        "cache_path": cache_path,
        "job_dir": sup.job_output_dir(output_base, run_dir),
    }
    save_mp_cache(cache_path, record)
    print(f"Saved MP correction cache to {cache_path}")
    return record


def cache_records_to_series(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "job_dir": record["job_dir"],
            "run": record["run"],
            "layer": LAYER,
            "steps": record["steps"],
            "total_params": record["total_params"],
            "batch_size": record["batch_size"],
            "width": record["width"],
        }
        for record in records
    ]


def positive_floor(arrays: Sequence[Any]) -> float | None:
    positives = []
    for values in arrays:
        array = np.asarray(values, dtype=float).ravel()
        finite = array[np.isfinite(array) & (array > 0.0)]
        if finite.size:
            positives.append(finite)
    if not positives:
        return None
    return float(np.min(np.concatenate(positives)))


def save_manifest(
    spectrum_dir: Path,
    real_records: Sequence[dict[str, Any]],
    mp_records: Sequence[dict[str, Any]],
    *,
    z_threshold: float,
) -> None:
    payload = {
        "layer": LAYER,
        "matrix": "C = W W^T / n, Dense_0 input-space overlap",
        "null": "MP permutation null; independently permute each input row across width",
        "detached_rule": detached_rule_version(z_threshold),
        "detached_z": float(z_threshold),
        "runs": [],
    }
    by_run = {record["run"]: record for record in mp_records}
    for real in ps.sorted_series_entries(cache_records_to_series(real_records)):
        mp = by_run[real["run"]]
        payload["runs"].append(
            {
                "run": real["run"],
                "width": int(real["width"]),
                "steps": [int(value) for value in mp["steps"]],
                "replicates": int(mp["replicates"]),
                "seed": int(mp["seed"]),
                "cache_file": Path(mp["cache_path"]).name,
                "detached_counts": [int(value) for value in mp["detached_counts"]],
                "edge": [float(value) for value in mp["edge"]],
                "analytic_mp_edge": [float(value) for value in mp["analytic_mp_edge"]],
            }
        )
    with (spectrum_dir / "mp_correction_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def record_step_index(record: dict[str, Any], step: int) -> int | None:
    matches = np.flatnonzero(np.asarray(record["steps"], dtype=np.int64) == int(step))
    if matches.size == 0:
        return None
    return int(matches[0])


def modes_for_ratio(k_detached: int, ratio: float, rank_count: int) -> int:
    if int(k_detached) <= 0:
        return 0
    return min(int(rank_count), int(np.ceil(float(ratio) * int(k_detached))))


def compression_ratio(mode_count: int, input_dim: int, width: int) -> float:
    denominator = float(max(int(input_dim) * int(width), 1))
    return float(int(mode_count) * (int(input_dim) + int(width)) / denominator)


def detached_spectrum_at_best_step(
    *,
    rows: np.ndarray,
    best_step: int,
    real_record: dict[str, Any],
    mp_record: dict[str, Any],
    replicates: int,
    seed: int,
    z_threshold: float,
) -> tuple[np.ndarray, float, int, float]:
    real_index = record_step_index(real_record, best_step)
    if real_index is not None:
        eigenvalues = np.asarray(real_record["eigenvalues"][real_index], dtype=float)
    else:
        eigenvalues = nonzero_overlap_eigenvalues(rows)

    mp_index = record_step_index(mp_record, best_step)
    if mp_index is not None:
        edge = float(np.asarray(mp_record["edge"], dtype=float)[mp_index])
        analytic_edge = float(np.asarray(mp_record["analytic_mp_edge"], dtype=float)[mp_index])
        null_spectra = np.asarray(mp_record["null_spectra"], dtype=float)[mp_index]
    else:
        rank_count = int(eigenvalues.size)
        null_spectra = np.empty((int(replicates), rank_count), dtype=np.float64)
        run_seed = stable_run_seed(str(real_record["run"]))
        for replicate in range(int(replicates)):
            seed_sequence = np.random.SeedSequence([int(seed), run_seed, int(best_step), int(replicate)])
            rng = np.random.default_rng(seed_sequence)
            null_spectra[replicate] = nonzero_overlap_eigenvalues(
                independently_permute_input_rows(rows, rng)
            )
        edge = float(np.max(null_spectra[:, 0]))
        analytic_edge = analytic_mp_edge(
            eigenvalues,
            int(real_record["input_dim"]),
            int(real_record["width"]),
        )
    k_detached = detached_count_from_null_spectrum(
        eigenvalues,
        null_spectra,
        z_threshold=z_threshold,
    )
    return eigenvalues, edge, int(k_detached), analytic_edge


def load_best_reconstruction_cache(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None


def best_reconstruction_cache_matches(
    record: dict[str, Any],
    *,
    run: str,
    best_step: int,
    best_time_index: int,
    replicates: int,
    seed: int,
    batch_size: int,
    z_threshold: float,
) -> bool:
    cached_ratios = np.asarray(record.get("ratios", []), dtype=float)
    if cached_ratios.shape != RECONSTRUCTION_RATIOS.shape:
        return False
    return (
        str(record.get("run")) == str(run)
        and str(record.get("detached_rule", "")) == detached_rule_version(z_threshold)
        and np.isclose(float(record.get("detached_z", np.nan)), float(z_threshold))
        and int(record.get("best_step", -1)) == int(best_step)
        and int(record.get("best_time_index", -1)) == int(best_time_index)
        and int(record.get("replicates", -1)) == int(replicates)
        and int(record.get("seed", -1)) == int(seed)
        and int(record.get("batch_size", -1)) == int(batch_size)
        and np.allclose(cached_ratios, RECONSTRUCTION_RATIOS, rtol=0.0, atol=1e-12)
    )


def save_best_reconstruction_cache(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)


def compute_or_load_best_reconstruction(
    zarr_path: Path,
    output_base: Path,
    real_record: dict[str, Any],
    mp_record: dict[str, Any],
    *,
    replicates: int,
    seed: int,
    batch_size: int,
    data_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]],
    z_threshold: float,
    force: bool,
) -> dict[str, Any] | None:
    run_dir = zarr_path.parent
    dense0_dir, dense0_meta, dense0_shape = sup.dense0_kernel(zarr_path)
    dense1_dir, dense1_meta, dense1_shape = sup.dense1_kernel(zarr_path)
    step_stride = ps.training_step_stride(run_dir)
    snapshot = ps.best_test_loss_snapshot(run_dir, int(dense0_shape[0]), step_stride)
    if snapshot is None:
        print(f"Skipping best-checkpoint reconstruction for {run_dir.name}: no finite test loss")
        return None
    best_time_index = int(snapshot["best_time_index"])
    if best_time_index >= int(dense1_shape[0]):
        print(
            f"Skipping best-checkpoint reconstruction for {run_dir.name}: "
            f"best Dense_0 time index {best_time_index} is missing in Dense_1"
        )
        return None
    best_step = int(snapshot["best_step"])

    cache_path = best_reconstruction_cache_path(
        output_base,
        run_dir,
        replicates=replicates,
        seed=seed,
        batch_size=batch_size,
    )
    cached = None if force else load_best_reconstruction_cache(cache_path)
    if cached is not None and best_reconstruction_cache_matches(
        cached,
        run=run_dir.name,
        best_step=best_step,
        best_time_index=best_time_index,
        replicates=replicates,
        seed=seed,
        batch_size=batch_size,
        z_threshold=z_threshold,
    ):
        return cached

    config = ps.load_yaml(run_dir / "simulation_config.yaml")
    network = config.get("network", {})
    training = config.get("training", {})
    activations = network.get("activations_per_layer", {})
    activation0 = ps.numpy_activation(str(activations.get("Dense_0", "relu")))
    activation1 = ps.numpy_activation(str(activations.get("Dense_1", "identity")))
    loss_type = str(training.get("loss_type", "mse"))

    dense0_kernel = ps.read_snapshot(dense0_dir, dense0_meta, best_time_index)
    dense1_kernel = ps.read_snapshot(dense1_dir, dense1_meta, best_time_index)
    rows = sup.weight_rows_by_input_unit(dense0_kernel)
    eigenvalues, edge, k_detached, analytic_edge = detached_spectrum_at_best_step(
        rows=rows,
        best_step=best_step,
        real_record=real_record,
        mp_record=mp_record,
        replicates=replicates,
        seed=seed,
        z_threshold=z_threshold,
    )
    rank_count = int(eigenvalues.size)
    mode_counts = [
        modes_for_ratio(k_detached, float(ratio), rank_count)
        for ratio in RECONSTRUCTION_RATIOS
    ]
    required_modes = sorted({mode for mode in mode_counts if mode > 0})
    max_modes = max(required_modes) if required_modes else 1
    row_norms, u, singular_values, vt, _ = ps.spectral_basis_for_dense0(
        dense0_kernel,
        max_modes,
        raw_analysis=True,
    )

    x_test, y_test = ps.load_reconstruction_eval_data(run_dir, data_cache)
    full_loss = ps.evaluate_dense_two_layer_loss(
        x_test,
        y_test,
        dense0_kernel,
        dense1_kernel,
        activation0=activation0,
        activation1=activation1,
        loss_type=loss_type,
        batch_size=batch_size,
    )

    loss_by_mode: dict[int, float] = {}
    for mode_count in sorted(set(mode_counts)):
        loss, _ = ps.evaluate_reconstructed_loss(
            x_test,
            y_test,
            dense1_kernel,
            row_norms=row_norms,
            u=u,
            singular_values=singular_values,
            vt=vt,
            modes=int(mode_count),
            renormalize=False,
            activation0=activation0,
            activation1=activation1,
            loss_type=loss_type,
            batch_size=batch_size,
        )
        loss_by_mode[int(mode_count)] = float(loss)

    input_dim = int(real_record["input_dim"])
    width = int(real_record["width"])
    losses = [loss_by_mode[int(mode_count)] for mode_count in mode_counts]
    record = {
        "run": run_dir.name,
        "job_dir": str(sup.job_output_dir(output_base, run_dir)),
        "scheme": ps.scheme_from_run(run_dir.name),
        "label": ps.label_from_run(run_dir.name),
        "width": width,
        "input_dim": input_dim,
        "rank_count": rank_count,
        "best_step": best_step,
        "best_time_index": best_time_index,
        "best_history_index": int(snapshot["best_history_index"]),
        "logged_best_test_loss": float(snapshot["best_test_loss"]),
        "full_loss": float(full_loss),
        "edge": float(edge),
        "analytic_edge": float(analytic_edge),
        "k_detached": int(k_detached),
        "detached_rule": detached_rule_version(z_threshold),
        "detached_z": float(z_threshold),
        "replicates": int(replicates),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "ratios": [float(value) for value in RECONSTRUCTION_RATIOS],
        "mode_counts": [int(value) for value in mode_counts],
        "compression_ratios": [
            compression_ratio(int(mode_count), input_dim, width)
            for mode_count in mode_counts
        ],
        "losses": [float(value) for value in losses],
        "cache_file": str(cache_path),
    }
    save_best_reconstruction_cache(cache_path, record)
    print(
        f"{run_dir.name}: best step {best_step}, k_det={int(k_detached)}, "
        f"full_loss={float(full_loss):.6g}; saved {cache_path}"
    )
    return record


def sorted_reconstruction_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: tuple(ps.width_key(str(item["label"]))))


def load_cached_best_reconstruction_records(
    input_dir: Path,
    output_base: Path,
    real_records: Sequence[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    batch_size: int,
    z_threshold: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing: list[Path] = []
    stale: list[Path] = []
    input_root = input_dir.resolve()
    for real in ps.sorted_series_entries(real_records):
        cache_path = best_reconstruction_cache_path(
            output_base,
            input_root / str(real["run"]),
            replicates=replicates,
            seed=seed,
            batch_size=batch_size,
        )
        cached = load_best_reconstruction_cache(cache_path)
        if cached is None:
            missing.append(cache_path)
            continue
        cached_ratios = np.asarray(cached.get("ratios", []), dtype=float)
        matches = (
            str(cached.get("detached_rule", "")) == detached_rule_version(z_threshold)
            and np.isclose(float(cached.get("detached_z", np.nan)), float(z_threshold))
            and int(cached.get("replicates", -1)) == int(replicates)
            and int(cached.get("seed", -1)) == int(seed)
            and int(cached.get("batch_size", -1)) == int(batch_size)
            and cached_ratios.shape == RECONSTRUCTION_RATIOS.shape
            and np.allclose(cached_ratios, RECONSTRUCTION_RATIOS, rtol=0.0, atol=1e-12)
        )
        if not matches:
            stale.append(cache_path)
            continue
        records.append(cached)
    if missing or stale:
        problems = []
        if missing:
            problems.append("missing: " + ", ".join(str(path) for path in missing))
        if stale:
            problems.append("stale: " + ", ".join(str(path) for path in stale))
        raise RuntimeError(
            "Cannot regenerate best-fraction plot from cache; run the full script first. "
            + "; ".join(problems)
        )
    return records


def reconstruction_loss_at_ratio(record: dict[str, Any], ratio: float) -> tuple[int, float]:
    ratios = np.asarray(record["ratios"], dtype=float)
    matches = np.flatnonzero(np.isclose(ratios, float(ratio), rtol=0.0, atol=1e-12))
    if matches.size == 0:
        raise ValueError(f"Missing reconstruction ratio {ratio} for {record['run']}")
    index = int(matches[0])
    return int(record["mode_counts"][index]), float(record["losses"][index])


def set_positive_log_yscale(ax: plt.Axes, arrays: Sequence[Any]) -> None:
    floor = positive_floor(arrays)
    if floor is None:
        return
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor / np.sqrt(10.0))


def set_log_loss_ylim_near_baseline(
    ax: plt.Axes,
    *,
    baselines: Sequence[Any],
    values: Sequence[Any],
) -> None:
    baseline_floor = positive_floor(baselines)
    value_floor = positive_floor(values)
    if baseline_floor is None and value_floor is None:
        return
    ax.set_yscale("log")
    bottoms = [value for value in (baseline_floor, value_floor) if value is not None]
    bottom = min(bottoms) * 0.97
    if baseline_floor is not None:
        bottom = min(bottom, baseline_floor * 0.97)
    maxima = []
    for array_like in [*baselines, *values]:
        array = np.asarray(array_like, dtype=float).ravel()
        finite = array[np.isfinite(array) & (array > 0.0)]
        if finite.size:
            maxima.append(float(np.max(finite)))
    if maxima:
        ax.set_ylim(bottom=bottom, top=max(maxima) * 1.08)
    else:
        ax.set_ylim(bottom=bottom)


def apply_readable_ticks(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)


def set_panel_loss_ticks(ax: plt.Axes, full_loss: float) -> None:
    ymin, ymax = ax.get_ylim()
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin <= 0.0:
        return
    baseline = float(full_loss)
    step = 0.2
    start = np.floor(ymin / step) * step
    stop = np.ceil(ymax / step) * step
    ticks = [
        float(value)
        for value in np.arange(start, stop + 0.5 * step, step)
        if ymin <= value <= ymax and abs(float(value) - baseline) > 0.035
    ]
    if ymin <= baseline <= ymax:
        ticks.append(baseline)
    ticks = sorted({round(tick, 6) for tick in ticks})
    if not ticks:
        return

    def formatter(value: float, _pos: int) -> str:
        if np.isclose(value, baseline, rtol=0.0, atol=max(1e-5, 1e-4 * abs(baseline))):
            return f"{baseline:.2f}"
        return f"{value:.1f}"

    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(formatter))


def reconstruction_style_handles(*, include_full_model: bool = True) -> tuple[list[Any], list[str]]:
    handles = [plt.Line2D([], [], color="black", marker="o", linewidth=1.55, markersize=4.0)]
    labels = ["Reconstruction"]
    if include_full_model:
        handles.append(plt.Line2D([], [], color="black", linestyle="--", linewidth=1.25))
        labels.append("Full model")
    return handles, labels


def compact_width_label(label: Any) -> str:
    return str(label).split("x", maxsplit=1)[0]


def compact_width_legend_title(schemes: set[str]) -> str:
    scheme_label = "SP" if "standard" in schemes else "muP"
    return f"width ($n$) - {scheme_label}"


def add_reconstruction_legend(
    ax: plt.Axes,
    *,
    series_entries: Sequence[dict[str, Any]],
    colors: dict[str, Any],
    fontsize: int = 12,
    include_full_model: bool = True,
) -> None:
    legend_width_entries: list[tuple[Any, str]] = []
    schemes: set[str] = set()
    for entry in ps.sorted_series_entries(list(series_entries)):
        run = str(entry["run"])
        scheme = ps.scheme_from_run(run)
        handle = plt.Line2D([], [], color=colors[run], marker="o", linewidth=1.55, markersize=4.0)
        schemes.add(scheme)
        legend_width_entries.append((handle, compact_width_label(ps.label_from_run(run))))
    handles = [handle for handle, _ in legend_width_entries]
    labels = [label for _, label in legend_width_entries]
    style_handles, style_labels = reconstruction_style_handles(include_full_model=include_full_model)
    handles.extend(style_handles)
    labels.extend(style_labels)
    if handles:
        legend = ax.legend(
            handles,
            labels,
            title=compact_width_legend_title(schemes),
            loc="center",
            fontsize=fontsize,
            title_fontsize=fontsize,
            frameon=False,
        )
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        legend.get_title().set_fontweight("bold")


def save_best_detached_reconstruction_loss_plot(
    records: Sequence[dict[str, Any]],
    series: Sequence[dict[str, Any]],
    spectrum_dir: Path,
    dpi: int,
) -> None:
    entries = sorted_reconstruction_records(records)
    if not entries:
        return
    colors, _ = ps.series_colors(list(series))
    widths = np.asarray([int(record["width"]) for record in entries], dtype=float)
    full_losses = np.asarray([float(record["full_loss"]) for record in entries], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)
    panel_specs = [
        (0.5, r"$k=0.5\,k_{\mathrm{det}}$ (50% of $k_{\mathrm{det}}$)"),
        (1.0, r"$k=k_{\mathrm{det}}$ (100% of $k_{\mathrm{det}}$)"),
    ]
    for ax, (ratio, title) in zip(axes, panel_specs):
        losses = np.asarray([reconstruction_loss_at_ratio(record, ratio)[1] for record in entries], dtype=float)
        rec_line, = ax.plot(
            widths,
            losses,
            color="black",
            marker="o",
            markersize=4.0,
            linewidth=1.6,
            label="MP-detached reconstruction",
            zorder=2,
        )
        for width, loss, record in zip(widths, losses, entries):
            ax.scatter(
                [width],
                [loss],
                color=colors.get(record["run"], "0.25"),
                edgecolors="black",
                linewidths=0.25,
                s=42,
                zorder=3,
            )
        full_line, = ax.plot(
            widths,
            full_losses,
            color="black",
            linestyle="--",
            linewidth=1.35,
            label="Full model",
            zorder=1,
        )
        ax.set_title(title, fontsize=15)
        ax.set_xlabel(r"Width $n$", fontsize=15)
        ax.set_ylabel("Test loss", fontsize=15)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(widths))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", labelbottom=False)
        ax.set_xticklabels([str(int(width)) for width in widths], rotation=30, ha="right")
        set_log_loss_ylim_near_baseline(ax, baselines=[full_losses], values=[losses])
        apply_readable_ticks(ax)
        ax.grid(True, which="both", alpha=0.3)
        if ratio == 1.0:
            ax.legend([rec_line, full_line], ["MP-detached reconstruction", "Full model"], fontsize=12, frameon=False)

    fig.savefig(
        spectrum_dir / f"best_test_loss_mp_detached_reconstruction_{LAYER}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def compression_label(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value == 0.0:
        return "0"
    if value < 0.01:
        return f"{value:.1e}"
    if value < 0.1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def save_best_loss_vs_mode_fraction_plot(
    records: Sequence[dict[str, Any]],
    series: Sequence[dict[str, Any]],
    spectrum_dir: Path,
    dpi: int,
) -> None:
    entries = sorted_reconstruction_records(records)
    if not entries:
        return
    colors, _ = ps.series_colors(list(series))
    ncols = 3 if len(entries) > 4 else max(len(entries), 1)
    nrows = int(np.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.9 * ncols, 3.75 * nrows),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.07, top=0.91, wspace=0.22, hspace=0.55)
    axes_flat = axes.ravel()
    series_entries = ps.sorted_series_entries(list(series))
    used_axes: list[tuple[int, int, plt.Axes]] = []

    for flat_index, (ax, record) in enumerate(zip(axes_flat, entries)):
        row_index = flat_index // ncols
        col_index = flat_index % ncols
        used_axes.append((row_index, col_index, ax))
        ratios = np.asarray(record["ratios"], dtype=float)
        full_loss = float(record["full_loss"])
        losses = (np.asarray(record["losses"], dtype=float) - full_loss) / full_loss
        compression = np.asarray(record["compression_ratios"], dtype=float)
        color = colors.get(record["run"], "0.25")
        ax.plot(
            ratios,
            losses,
            color=color,
            marker="o",
            markersize=3.2,
            linewidth=1.55,
            label="Reconstruction",
        )
        ax.text(
            0.96,
            0.92,
            rf"$k_\mathrm{{det}}={int(record['k_detached'])}$",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=15,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
        )
        ax.set_xlim(0.05, 5.15)
        set_positive_log_yscale(ax, [losses])
        if col_index == 0:
            ax.set_ylabel(r"$(L-L_{\mathrm{Full}})/L_{\mathrm{Full}}$", fontsize=15)
        apply_readable_ticks(ax)
        ax.grid(True, which="both", alpha=0.3)

        top = ax.twiny()
        top.set_xlim(ax.get_xlim())
        tick_ratios = np.asarray([0.1, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        tick_labels = []
        for tick in tick_ratios:
            match = int(np.argmin(np.abs(ratios - tick)))
            tick_labels.append(compression_label(float(compression[match])))
        top.set_xticks(tick_ratios)
        top.set_xticklabels(tick_labels, fontsize=11)
        if row_index == 0:
            top.set_xlabel(r"$R(k)$", fontsize=15, labelpad=9)
        top.tick_params(axis="both", which="major", labelsize=11, pad=3)

    for ax in axes_flat[len(entries):]:
        ax.axis("off")
    if used_axes:
        bottom_by_col = {
            col: max((row, axis) for row, c, axis in used_axes if c == col)[1]
            for col in {col for _, col, _ in used_axes}
        }
        for _, col, ax in used_axes:
            if bottom_by_col.get(col) is ax:
                ax.set_xlabel(r"$k/k_{\mathrm{det}}$", fontsize=15)
            else:
                ax.tick_params(labelbottom=False)
    if len(entries) < len(axes_flat):
        legend_ax = axes_flat[-1]
        legend_ax.axis("off")
        add_reconstruction_legend(
            legend_ax,
            series_entries=series_entries,
            colors=colors,
            fontsize=12,
            include_full_model=False,
        )

    fig.savefig(
        spectrum_dir / f"best_test_loss_vs_detached_mode_fraction_{LAYER}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_best_loss_vs_mode_fraction_overlay(
    records: Sequence[dict[str, Any]],
    series: Sequence[dict[str, Any]],
    spectrum_dir: Path,
    dpi: int,
) -> None:
    entries = sorted_reconstruction_records(records)
    if not entries:
        return
    colors, _ = ps.series_colors(list(series))
    fig = plt.figure(figsize=(9.6, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 5.0))
    legend_ax = fig.add_subplot(grid[0, 0])
    legend_ax.axis("off")
    ax = fig.add_subplot(grid[0, 1])
    all_losses: list[np.ndarray] = []
    all_baselines: list[np.ndarray] = []
    legend_width_entries: list[tuple[Any, str]] = []
    schemes: set[str] = set()
    for record in entries:
        ratios = np.asarray(record["ratios"], dtype=float)
        losses = np.asarray(record["losses"], dtype=float)
        full_loss = float(record["full_loss"])
        color = colors.get(record["run"], "0.25")
        line, = ax.plot(
            ratios,
            losses,
            color=color,
            marker="o",
            markersize=3.2,
            linewidth=1.55,
        )
        ax.axhline(
            full_loss,
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )
        schemes.add(ps.scheme_from_run(str(record["run"])))
        legend_width_entries.append((line, str(record["label"]).split("x", maxsplit=1)[0]))
        all_losses.append(losses)
        all_baselines.append(np.asarray([full_loss], dtype=float))

    ax.set_xlabel(r"$k/k_{\mathrm{det}}$", fontsize=15)
    ax.set_ylabel("Best test loss", fontsize=15)
    ax.set_xlim(0.05, 5.15)
    set_log_loss_ylim_near_baseline(ax, baselines=all_baselines, values=all_losses)
    apply_readable_ticks(ax)
    ax.grid(True, which="both", alpha=0.3)
    handles = [handle for handle, _ in legend_width_entries]
    labels = [label for _, label in legend_width_entries]
    style_handles, style_labels = reconstruction_style_handles(include_full_model=True)
    handles.extend(style_handles)
    labels.extend(style_labels)
    if handles:
        scheme_label = "SP" if "standard" in schemes else "muP"
        legend = legend_ax.legend(
            handles,
            labels,
            title=f"width ($n$) - {scheme_label}",
            loc="center",
            fontsize=12,
            title_fontsize=12,
            frameon=False,
        )
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        legend.get_title().set_fontweight("bold")
    fig.savefig(
        spectrum_dir / f"best_test_loss_vs_detached_mode_fraction_all_widths_{LAYER}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def ratio_key(ratio: float) -> str:
    return f"{float(ratio):g}"


def time_reconstruction_cache_matches(
    record: dict[str, Any],
    real_record: dict[str, Any],
    mp_record: dict[str, Any],
    *,
    replicates: int,
    seed: int,
    batch_size: int,
    z_threshold: float,
) -> bool:
    cached_ratios = np.asarray(record.get("ratios", []), dtype=float)
    if cached_ratios.shape != TIME_RECONSTRUCTION_RATIOS.shape:
        return False
    return (
        str(record.get("run")) == str(real_record["run"])
        and str(record.get("detached_rule", "")) == detached_rule_version(z_threshold)
        and np.isclose(float(record.get("detached_z", np.nan)), float(z_threshold))
        and int(record.get("replicates", -1)) == int(replicates)
        and int(record.get("seed", -1)) == int(seed)
        and int(record.get("batch_size", -1)) == int(batch_size)
        and np.allclose(cached_ratios, TIME_RECONSTRUCTION_RATIOS, rtol=0.0, atol=1e-12)
        and np.array_equal(
            np.asarray(record.get("steps", []), dtype=np.int64),
            np.asarray(real_record["steps"], dtype=np.int64),
        )
        and np.array_equal(
            np.asarray(record.get("time_indices", []), dtype=np.int64),
            np.asarray(real_record["time_indices"], dtype=np.int64),
        )
        and np.array_equal(
            np.asarray(record.get("k_detached", []), dtype=np.int64),
            np.asarray(mp_record["detached_counts"], dtype=np.int64),
        )
    )


def compute_or_load_time_reconstruction(
    zarr_path: Path,
    output_base: Path,
    real_record: dict[str, Any],
    mp_record: dict[str, Any],
    *,
    replicates: int,
    seed: int,
    batch_size: int,
    data_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]],
    z_threshold: float,
    force: bool,
) -> dict[str, Any] | None:
    run_dir = zarr_path.parent
    cache_path = time_reconstruction_cache_path(
        output_base,
        run_dir,
        replicates=replicates,
        seed=seed,
        batch_size=batch_size,
    )
    cached = None if force else load_best_reconstruction_cache(cache_path)
    if cached is not None and time_reconstruction_cache_matches(
        cached,
        real_record,
        mp_record,
        replicates=replicates,
        seed=seed,
        batch_size=batch_size,
        z_threshold=z_threshold,
    ):
        return cached

    dense0_dir, dense0_meta, dense0_shape = sup.dense0_kernel(zarr_path)
    dense1_dir, dense1_meta, dense1_shape = sup.dense1_kernel(zarr_path)
    steps = np.asarray(real_record["steps"], dtype=np.int64)
    time_indices = np.asarray(real_record["time_indices"], dtype=np.int64)
    k_detached = np.asarray(mp_record["detached_counts"], dtype=np.int64)
    valid = time_indices < int(dense1_shape[0])
    if not np.all(valid):
        steps = steps[valid]
        time_indices = time_indices[valid]
        k_detached = k_detached[valid]
    if steps.size == 0:
        print(f"Skipping time-dependent reconstruction for {run_dir.name}: no matched snapshots")
        return None

    config = ps.load_yaml(run_dir / "simulation_config.yaml")
    network = config.get("network", {})
    training = config.get("training", {})
    activations = network.get("activations_per_layer", {})
    activation0 = ps.numpy_activation(str(activations.get("Dense_0", "relu")))
    activation1 = ps.numpy_activation(str(activations.get("Dense_1", "identity")))
    loss_type = str(training.get("loss_type", "mse"))
    x_test, y_test = ps.load_reconstruction_eval_data(run_dir, data_cache)

    rank_count = int(real_record["eigenvalues"].shape[1])
    losses_by_ratio = {ratio_key(ratio): [] for ratio in TIME_RECONSTRUCTION_RATIOS}
    modes_by_ratio = {ratio_key(ratio): [] for ratio in TIME_RECONSTRUCTION_RATIOS}
    full_losses = []

    for time_index, step, k_det in zip(time_indices, steps, k_detached):
        step_modes = [
            modes_for_ratio(int(k_det), float(ratio), rank_count)
            for ratio in TIME_RECONSTRUCTION_RATIOS
        ]
        max_modes = max([mode for mode in step_modes if mode > 0], default=1)
        dense0_kernel = ps.read_snapshot(dense0_dir, dense0_meta, int(time_index))
        dense1_kernel = ps.read_snapshot(dense1_dir, dense1_meta, int(time_index))
        row_norms, u, singular_values, vt, _ = ps.spectral_basis_for_dense0(
            dense0_kernel,
            max_modes,
            raw_analysis=True,
        )
        full_loss = ps.evaluate_dense_two_layer_loss(
            x_test,
            y_test,
            dense0_kernel,
            dense1_kernel,
            activation0=activation0,
            activation1=activation1,
            loss_type=loss_type,
            batch_size=batch_size,
        )
        full_losses.append(float(full_loss))

        loss_by_mode: dict[int, float] = {}
        for mode_count in sorted(set(step_modes)):
            loss, _ = ps.evaluate_reconstructed_loss(
                x_test,
                y_test,
                dense1_kernel,
                row_norms=row_norms,
                u=u,
                singular_values=singular_values,
                vt=vt,
                modes=int(mode_count),
                renormalize=False,
                activation0=activation0,
                activation1=activation1,
                loss_type=loss_type,
                batch_size=batch_size,
            )
            loss_by_mode[int(mode_count)] = float(loss)

        for ratio, mode_count in zip(TIME_RECONSTRUCTION_RATIOS, step_modes):
            key = ratio_key(float(ratio))
            modes_by_ratio[key].append(int(mode_count))
            losses_by_ratio[key].append(float(loss_by_mode[int(mode_count)]))
        print(
            f"{run_dir.name}: time step {int(step)}, "
            f"k_det={int(k_det)}, modes={step_modes}"
        )

    record = {
        "run": run_dir.name,
        "job_dir": str(sup.job_output_dir(output_base, run_dir)),
        "scheme": ps.scheme_from_run(run_dir.name),
        "label": ps.label_from_run(run_dir.name),
        "width": int(real_record["width"]),
        "input_dim": int(real_record["input_dim"]),
        "rank_count": rank_count,
        "steps": [int(step) for step in steps],
        "time_indices": [int(index) for index in time_indices],
        "k_detached": [int(value) for value in k_detached],
        "detached_rule": detached_rule_version(z_threshold),
        "detached_z": float(z_threshold),
        "ratios": [float(value) for value in TIME_RECONSTRUCTION_RATIOS],
        "mode_counts": modes_by_ratio,
        "losses": losses_by_ratio,
        "full_loss": [float(value) for value in full_losses],
        "replicates": int(replicates),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "cache_file": str(cache_path),
    }
    save_best_reconstruction_cache(cache_path, record)
    print(f"Saved time-dependent MP-detached reconstruction cache to {cache_path}")
    return record


def save_time_reconstruction_loss_plot(
    records: Sequence[dict[str, Any]],
    series: Sequence[dict[str, Any]],
    spectrum_dir: Path,
    dpi: int,
) -> None:
    entries = sorted_reconstruction_records(records)
    if not entries:
        return
    colors, legend_entries = ps.series_colors(list(series))
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()
    panel_specs = [
        (0.5, r"$k=0.5\,k_{\mathrm{det}}(t)$"),
        (1.0, r"$k=k_{\mathrm{det}}(t)$"),
        (2.0, r"$k=2\,k_{\mathrm{det}}(t)$"),
        (5.0, r"$k=5\,k_{\mathrm{det}}(t)$"),
    ]
    for panel_index, (ax, (ratio, title)) in enumerate(zip(axes_flat, panel_specs)):
        key = ratio_key(ratio)
        panel_steps = []
        panel_losses = []
        panel_baselines = []
        for entry_index, record in enumerate(entries):
            steps = np.asarray(record["steps"], dtype=float)
            losses = np.asarray(record["losses"][key], dtype=float)
            full_loss = np.asarray(record["full_loss"], dtype=float)
            color = colors.get(record["run"], "0.25")
            line, = ax.plot(
                steps,
                losses,
                color=color,
                marker="o",
                markersize=3.0,
                linewidth=1.25,
            )
            ax.plot(
                steps,
                full_loss,
                color=color,
                linestyle="--",
                linewidth=0.95,
                alpha=0.72,
            )
            if panel_index == 0:
                legend_entries[ps.scheme_from_run(record["run"])].append((line, str(record["label"])))
            panel_steps.append(steps)
            panel_losses.append(losses)
            panel_baselines.append(full_loss)

        ps.set_nonnegative_loglike_xscale(ax, panel_steps)
        set_log_loss_ylim_near_baseline(ax, baselines=panel_baselines, values=panel_losses)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Training Step", fontsize=15)
        ax.set_ylabel("Test loss", fontsize=15)
        apply_readable_ticks(ax)
        ax.grid(True, which="both", alpha=0.3)

    handles, labels, header_indices = ps.combined_legend(legend_entries)
    style_handles, style_labels = reconstruction_style_handles()
    handles.extend(style_handles)
    labels.extend(style_labels)
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=12,
            frameon=False,
        )
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        for index in header_indices:
            legend.get_texts()[index].set_fontweight("bold")
    fig.suptitle(
        f"{LAYER} MP-detached spectral reconstruction test loss\n"
        "solid: reconstructed Dense_0, dashed: original checkpoint",
        fontsize=15,
    )
    fig.savefig(
        spectrum_dir / f"test_loss_time_mp_detached_reconstruction_{LAYER}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_best_reconstruction_plots(
    records: Sequence[dict[str, Any]],
    real_records: Sequence[dict[str, Any]],
    spectrum_dir: Path,
    dpi: int,
) -> None:
    save_best_detached_reconstruction_loss_plot(records, real_records, spectrum_dir, dpi)
    save_best_loss_vs_mode_fraction_plot(records, real_records, spectrum_dir, dpi)
    save_best_loss_vs_mode_fraction_overlay(records, real_records, spectrum_dir, dpi)
    print(f"Saved best-checkpoint MP-detached reconstruction plots to {spectrum_dir}")


def spectrum_payload_at_step(
    real_record: dict[str, Any],
    mp_record: dict[str, Any],
    step: int,
    *,
    z_threshold: float,
) -> dict[str, Any] | None:
    real_index = record_step_index(real_record, int(step))
    mp_index = record_step_index(mp_record, int(step))
    if real_index is None or mp_index is None:
        return None
    eigenvalues = np.asarray(real_record["eigenvalues"][real_index], dtype=float)
    null_spectra = np.asarray(mp_record["null_spectra"], dtype=float)[mp_index]
    detached_mask = detached_mask_from_null_spectrum(
        eigenvalues,
        null_spectra,
        z_threshold=z_threshold,
    )
    return {
        "eigenvalues": eigenvalues,
        "null_min": np.asarray(mp_record["null_min"][mp_index], dtype=float),
        "null_max": np.asarray(mp_record["null_max"][mp_index], dtype=float),
        "null_mean": np.asarray(mp_record["null_mean"][mp_index], dtype=float),
        "edge": float(np.asarray(mp_record["edge"], dtype=float)[mp_index]),
        "analytic_edge": float(np.asarray(mp_record["analytic_mp_edge"], dtype=float)[mp_index]),
        "detached_count": int(np.count_nonzero(detached_mask)),
        "detached_mask": detached_mask,
    }


def spectrum_payload_from_checkpoint(
    store: Path,
    real_record: dict[str, Any],
    *,
    step: int,
    time_index: int,
    replicates: int,
    seed: int,
    z_threshold: float,
) -> dict[str, Any]:
    array_dir, meta, _ = sup.dense0_kernel(store)
    rows = sup.weight_rows_by_input_unit(ps.read_snapshot(array_dir, meta, int(time_index)))
    eigenvalues = nonzero_overlap_eigenvalues(rows)
    rank_count = int(eigenvalues.size)
    null_spectra = np.empty((int(replicates), rank_count), dtype=np.float64)
    run_seed = stable_run_seed(str(real_record["run"]))
    for replicate in range(int(replicates)):
        seed_sequence = np.random.SeedSequence([int(seed), run_seed, int(step), int(replicate)])
        rng = np.random.default_rng(seed_sequence)
        null_spectra[replicate] = nonzero_overlap_eigenvalues(
            independently_permute_input_rows(rows, rng)
        )
    detached_mask = detached_mask_from_null_spectrum(
        eigenvalues,
        null_spectra,
        z_threshold=z_threshold,
    )
    return {
        "eigenvalues": eigenvalues,
        "null_min": np.min(null_spectra, axis=0),
        "null_max": np.max(null_spectra, axis=0),
        "null_mean": np.mean(null_spectra, axis=0),
        "edge": float(np.max(null_spectra[:, 0])),
        "analytic_edge": analytic_mp_edge(
            eigenvalues,
            int(real_record["input_dim"]),
            int(real_record["width"]),
        ),
        "detached_count": int(np.count_nonzero(detached_mask)),
        "detached_mask": detached_mask,
    }


def best_checkpoint_summary_records(
    stores_by_run: dict[str, Path],
    real_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for real in ps.sorted_series_entries(real_records):
        store = stores_by_run[str(real["run"])]
        run_dir = store.parent
        _, _, dense0_shape = sup.dense0_kernel(store)
        snapshot = ps.best_test_loss_snapshot(
            run_dir,
            int(dense0_shape[0]),
            ps.training_step_stride(run_dir),
        )
        if snapshot is None:
            continue
        records.append(
            {
                "run": str(real["run"]),
                "job_dir": str(real["job_dir"]),
                "scheme": ps.scheme_from_run(str(real["run"])),
                "label": ps.label_from_run(str(real["run"])),
                "width": int(real["width"]),
                "input_dim": int(real["input_dim"]),
                "best_step": int(snapshot["best_step"]),
                "best_time_index": int(snapshot["best_time_index"]),
            }
        )
    return records


def plot_mp_spectrum_panel(
    ax: plt.Axes,
    payload: dict[str, Any],
    *,
    color: Any,
    label: str,
    log_y: bool,
    log_x: bool,
    show_analytic_edge: bool,
) -> Any:
    eigenvalues = np.asarray(payload["eigenvalues"], dtype=float)
    null_min = np.asarray(payload["null_min"], dtype=float)
    null_max = np.asarray(payload["null_max"], dtype=float)
    null_mean = np.asarray(payload["null_mean"], dtype=float)
    ranks = np.arange(1, eigenvalues.size + 1, dtype=float)
    detached = np.asarray(payload.get("detached_mask", []), dtype=bool)
    if detached.shape != eigenvalues.shape:
        detached = np.zeros_like(eigenvalues, dtype=bool)
    y_arrays = [eigenvalues, null_min, null_max, null_mean]

    floor = positive_floor(y_arrays)
    plot_min = np.maximum(null_min, floor if floor is not None else 0.0)
    ax.fill_between(
        ranks,
        plot_min,
        null_max,
        color="0.6",
        alpha=0.22,
        linewidth=0.0,
        zorder=1,
    )
    real_line, = ax.plot(
        ranks,
        eigenvalues,
        linewidth=1.65,
        color=color,
        zorder=3,
    )
    ax.plot(
        ranks,
        null_mean,
        color="black",
        linestyle="--",
        linewidth=1.45,
        alpha=0.9,
        zorder=5,
    )
    if np.any(detached):
        ax.scatter(
            ranks[detached],
            eigenvalues[detached],
            s=12,
            color=color,
            edgecolors="black",
            linewidths=0.2,
            zorder=4,
        )

    if log_x:
        ax.set_xscale("log")
    if log_y and floor is not None:
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor / np.sqrt(10.0))
    ax.set_title(label, fontsize=11, color=color, fontweight="bold")
    ax.text(
        0.04,
        0.06,
        rf"$k_\mathrm{{det}}={int(payload['detached_count'])}$",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=15,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 2.0},
    )
    ax.grid(True, which="both", alpha=0.3)
    return real_line


def save_best_test_loss_spectrum_plot(
    reconstruction_records: Sequence[dict[str, Any]],
    real_records: Sequence[dict[str, Any]],
    mp_records: Sequence[dict[str, Any]],
    spectrum_dir: Path,
    dpi: int,
    *,
    log_y: bool,
    log_x: bool,
    stores_by_run: dict[str, Path] | None = None,
    replicates: int = 5,
    seed: int = 0,
    z_threshold: float = DETACHED_Z_DEFAULT,
) -> None:
    entries = sorted_reconstruction_records(reconstruction_records)
    if not entries:
        return
    series = cache_records_to_series(real_records)
    colors, _ = ps.series_colors(series)
    real_by_run = {record["run"]: record for record in real_records}
    mp_by_run = {record["run"]: record for record in mp_records}

    ncols = 3 if len(entries) > 4 else max(len(entries), 1)
    nrows = int(np.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 3.55 * nrows),
        sharex="col",
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    legend_width_entries: list[tuple[Any, str]] = []
    schemes: set[str] = set()
    used_axes: list[tuple[int, int, plt.Axes]] = []
    skipped = []

    for flat_index, (ax, record) in enumerate(zip(axes_array, entries)):
        run = str(record["run"])
        best_step = int(record["best_step"])
        real_record = real_by_run.get(run)
        mp_record = mp_by_run.get(run)
        payload = None
        if real_record is not None and mp_record is not None:
            payload = spectrum_payload_at_step(
                real_record,
                mp_record,
                best_step,
                z_threshold=z_threshold,
            )
        if (
            payload is None
            and stores_by_run is not None
            and real_record is not None
            and run in stores_by_run
        ):
            payload = spectrum_payload_from_checkpoint(
                stores_by_run[run],
                real_record,
                step=best_step,
                time_index=int(record["best_time_index"]),
                replicates=replicates,
                seed=seed,
                z_threshold=z_threshold,
            )
        if payload is None:
            ax.axis("off")
            skipped.append(f"{run}@{best_step}")
            continue
        row_index = flat_index // ncols
        col_index = flat_index % ncols
        used_axes.append((row_index, col_index, ax))
        color = colors.get(run, "0.25")
        label = ps.label_from_run(run)
        real_line = plot_mp_spectrum_panel(
            ax,
            payload,
            color=color,
            label=label,
            log_y=log_y,
            log_x=log_x,
            show_analytic_edge=False,
        )
        ax.text(
            0.96,
            0.92,
            f"step={best_step}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=10,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
        )
        schemes.add(ps.scheme_from_run(run))
        legend_width_entries.append((real_line, compact_width_label(label)))

    for ax in axes_array[len(entries):]:
        ax.axis("off")
    if used_axes:
        bottom_by_col = {
            col: max((row, axis) for row, c, axis in used_axes if c == col)[1]
            for col in {col for _, col, _ in used_axes}
        }
        for _, col, ax in used_axes:
            if bottom_by_col.get(col) is ax:
                ax.set_xlabel(r"Eigenvalue rank $(\alpha)$", fontsize=15)
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(r"$\lambda_\alpha$", fontsize=15)

    if len(entries) < len(axes_array):
        legend_ax = axes_array[-1]
        legend_ax.axis("off")
        handles = [handle for handle, _ in legend_width_entries]
        labels = [label for _, label in legend_width_entries]
        handles.append(plt.Line2D([], [], color="black", linestyle="--", linewidth=1.45))
        labels.append("Decorrelated null spectrum")
        if handles:
            legend = legend_ax.legend(
                handles,
                labels,
                title=compact_width_legend_title(schemes),
                loc="center",
                fontsize=12,
                title_fontsize=12,
                frameon=False,
            )
            legend._legend_box.align = "left"  # type: ignore[attr-defined]
            legend.get_title().set_fontweight("bold")

    fig.suptitle("Best Test Loss Checkpoints", fontsize=15, fontweight="bold")
    fig.savefig(spectrum_dir / f"eigenvalues_best_test_loss_{LAYER}.png", dpi=dpi)
    plt.close(fig)
    if skipped:
        print("Skipped best-test-loss spectrum panels missing from caches: " + ", ".join(skipped))
    print(f"Saved best-test-loss MP-corrected eigenvalue spectrum plot to {spectrum_dir}")


def save_mp_spectrum_plots(
    real_records: Sequence[dict[str, Any]],
    mp_records: Sequence[dict[str, Any]],
    *,
    dpi: int,
    log_y: bool,
    log_x: bool,
    z_threshold: float,
) -> None:
    if not real_records:
        return
    log_y = True
    job_dir = Path(real_records[0]["job_dir"])
    spectrum_dir = job_dir / f"mp_cor_eigenvalue_spectra_{LAYER}"
    spectrum_dir.mkdir(parents=True, exist_ok=True)

    series = cache_records_to_series(real_records)
    colors, _ = ps.series_colors(series)
    entries = ps.sorted_series_entries(series)
    real_by_run = {
        record["run"]: {
            int(step): np.asarray(values, dtype=float)
            for step, values in zip(record["steps"], record["eigenvalues"])
        }
        for record in real_records
    }
    mp_by_run = {
        record["run"]: {
            int(step): {
                "min": np.asarray(min_values, dtype=float),
                "max": np.asarray(max_values, dtype=float),
                "mean": np.asarray(mean_values, dtype=float),
                "null_spectra": np.asarray(null_spectrum, dtype=float),
                "edge": float(edge),
                "analytic_edge": float(analytic_edge),
                "detached_count": int(detached_count),
            }
            for step, min_values, max_values, mean_values, null_spectrum, edge, analytic_edge, detached_count in zip(
                record["steps"],
                record["null_min"],
                record["null_max"],
                record["null_mean"],
                record["null_spectra"],
                record["edge"],
                record["analytic_mp_edge"],
                record["detached_counts"],
            )
        }
        for record in mp_records
    }

    steps = sorted({int(step) for record in real_records for step in record["steps"]})
    for step in steps:
        ncols = 3 if len(entries) > 4 else max(len(entries), 1)
        nrows = int(np.ceil(len(entries) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.7 * ncols, 3.55 * nrows),
            sharex="col",
            constrained_layout=True,
        )
        axes_array = np.atleast_1d(axes).ravel()
        legend_width_entries: list[tuple[Any, str]] = []
        schemes: set[str] = set()
        used_axes: list[tuple[int, int, plt.Axes]] = []
        for flat_index, (ax, entry) in enumerate(zip(axes_array, entries)):
            run = entry["run"]
            eigenvalues = real_by_run[run].get(step)
            mp = mp_by_run[run].get(step)
            if eigenvalues is None or mp is None:
                ax.axis("off")
                continue
            row_index = flat_index // ncols
            col_index = flat_index % ncols
            used_axes.append((row_index, col_index, ax))
            ranks = np.arange(1, eigenvalues.size + 1, dtype=float)
            color = colors[run]
            null_min = mp["min"]
            null_max = mp["max"]
            null_mean = mp["mean"]
            detached = detached_mask_from_null_spectrum(
                eigenvalues,
                mp["null_spectra"],
                z_threshold=z_threshold,
            )
            detached_count = int(np.count_nonzero(detached))
            y_arrays = [eigenvalues, null_min, null_max, null_mean]

            floor = positive_floor(y_arrays)
            plot_min = np.maximum(null_min, floor if floor is not None else 0.0)
            ax.fill_between(
                ranks,
                plot_min,
                null_max,
                color="0.6",
                alpha=0.22,
                linewidth=0.0,
                zorder=1,
            )
            real_line, = ax.plot(
                ranks,
                eigenvalues,
                linewidth=1.65,
                color=color,
                zorder=3,
            )
            ax.plot(
                ranks,
                null_mean,
                color="black",
                linestyle="--",
                linewidth=1.45,
                alpha=0.9,
                zorder=5,
            )
            label = ps.label_from_run(run)
            schemes.add(ps.scheme_from_run(run))
            legend_width_entries.append((real_line, compact_width_label(label)))
            if np.any(detached):
                ax.scatter(
                    ranks[detached],
                    eigenvalues[detached],
                    s=12,
                    color=color,
                    edgecolors="black",
                    linewidths=0.2,
                    zorder=4,
                )

            if log_x:
                ax.set_xscale("log")
            if log_y:
                floor = positive_floor(y_arrays)
                if floor is not None:
                    ax.set_yscale("log")
                    ax.set_ylim(bottom=floor / np.sqrt(10.0))
            ax.set_title(label, fontsize=11, color=color, fontweight="bold")
            ax.text(
                0.04,
                0.06,
                rf"$k_\mathrm{{det}}={detached_count}$",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=15,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 2.0},
            )
            ax.grid(True, which="both", alpha=0.3)

        for ax in axes_array[len(entries):]:
            ax.axis("off")
        bottom_by_col = {
            col: max((row, axis) for row, c, axis in used_axes if c == col)[1]
            for col in {col for _, col, _ in used_axes}
        }
        for row, col, ax in used_axes:
            if bottom_by_col.get(col) is ax:
                ax.set_xlabel(r"Eigenvalue rank $(\alpha)$", fontsize=15)
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(r"$\lambda_\alpha$", fontsize=15)

        if len(entries) < len(axes_array):
            legend_ax = axes_array[-1]
            legend_ax.axis("off")
            handles = [handle for handle, _ in legend_width_entries]
            labels = [label for _, label in legend_width_entries]
            handles.append(plt.Line2D([], [], color="black", linestyle="--", linewidth=1.45))
            labels.append("Decorrelated null spectrum")
            if handles:
                legend = legend_ax.legend(
                    handles,
                    labels,
                    title=compact_width_legend_title(schemes),
                    loc="center",
                    fontsize=12,
                    title_fontsize=12,
                    frameon=False,
                )
                legend._legend_box.align = "left"  # type: ignore[attr-defined]
                legend.get_title().set_fontweight("bold")
        fig.suptitle(f"Training Step: {step}", fontsize=15, fontweight="bold")
        fig.savefig(spectrum_dir / f"eigenvalues_step_{step}.png", dpi=dpi)
        plt.close(fig)

    save_detached_count_plot(real_records, mp_records, spectrum_dir, dpi)
    save_step0_sanity_overlay(real_records, mp_records, spectrum_dir, dpi, log_y=log_y, log_x=log_x)
    save_manifest(spectrum_dir, real_records, mp_records, z_threshold=z_threshold)
    print(f"Saved {len(steps)} MP-corrected eigenvalue spectrum plots to {spectrum_dir}")


def save_step0_sanity_overlay(
    real_records: Sequence[dict[str, Any]],
    mp_records: Sequence[dict[str, Any]],
    spectrum_dir: Path,
    dpi: int,
    *,
    log_y: bool,
    log_x: bool,
) -> None:
    series = cache_records_to_series(real_records)
    entries = ps.sorted_series_entries(series)
    if not entries:
        return
    colors, _ = ps.series_colors(series)
    real_by_run = {record["run"]: record for record in real_records}
    mp_by_run = {record["run"]: record for record in mp_records}

    entry = entries[-1]
    run = entry["run"]
    real_record = real_by_run[run]
    mp_record = mp_by_run[run]
    matches = np.flatnonzero(np.asarray(real_record["steps"], dtype=np.int64) == 0)
    if matches.size == 0:
        return
    step_index = int(matches[0])
    real = np.asarray(real_record["eigenvalues"][step_index], dtype=float)
    null_spectra = np.asarray(mp_record["null_spectra"][step_index], dtype=float)
    null_min = np.asarray(mp_record["null_min"][step_index], dtype=float)
    null_max = np.asarray(mp_record["null_max"][step_index], dtype=float)
    null_mean = np.asarray(mp_record["null_mean"][step_index], dtype=float)
    ranks = np.arange(1, real.size + 1, dtype=float)
    color = colors[run]

    fig, ax = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
    floor = positive_floor([real, null_spectra, null_min, null_max, null_mean])
    plot_min = np.maximum(null_min, floor if floor is not None else 0.0)
    ax.fill_between(ranks, plot_min, null_max, color="0.6", alpha=0.3, linewidth=0.0, label="permutation null band")
    for spectrum in null_spectra:
        ax.plot(ranks, spectrum, color="0.55", alpha=0.45, linewidth=0.7)
    ax.plot(ranks, null_mean, color="0.25", linewidth=0.8, alpha=0.8, label="permutation null mean")
    ax.plot(ranks, real, color=color, linewidth=1.25, label=f"real {ps.label_from_run(run)}")
    if log_x:
        ax.set_xscale("log")
    if log_y and floor is not None:
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor / np.sqrt(10.0))
    ax.set_title(f"Step-0 MP sanity check - {ps.label_from_run(run)}")
    ax.set_xlabel("Eigenvalue rank", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Eigenvalue of $C$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(spectrum_dir / f"sanity_step_0_real_vs_permuted_{run}.png", dpi=dpi)
    plt.close(fig)


def save_detached_count_plot(
    real_records: Sequence[dict[str, Any]],
    mp_records: Sequence[dict[str, Any]],
    spectrum_dir: Path,
    dpi: int,
) -> None:
    series = cache_records_to_series(real_records)
    colors, _ = ps.series_colors(series)
    mp_by_run = {record["run"]: record for record in mp_records}

    fig, ax = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
    _, legend_entries = ps.series_colors(series)
    step_arrays = []
    for entry in ps.sorted_series_entries(series):
        run = entry["run"]
        record = mp_by_run[run]
        steps = np.asarray(record["steps"], dtype=float)
        counts = np.asarray(record["detached_counts"], dtype=float)
        line, = ax.plot(
            steps,
            counts,
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            color=colors[run],
        )
        step_arrays.append(steps)
        legend_entries[ps.scheme_from_run(run)].append((line, ps.label_from_run(run)))
    ps.set_nonnegative_loglike_xscale(ax, step_arrays)
    sup.set_nonnegative_loglike_yscale(
        ax,
        [np.asarray(record["detached_counts"], dtype=float) for record in mp_records],
    )
    ax.set_title(f"MP-detached eigenvalue count - {LAYER}")
    ax.set_xlabel("Training Step", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$k_{\mathrm{det}}$ (rank-wise null crossing)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, which="both", alpha=0.3)
    ps.apply_combined_legend(ax, legend_entries)
    fig.savefig(spectrum_dir / f"detached_count_vs_step_{LAYER}.png", dpi=dpi)
    plt.close(fig)


def load_real_records(input_dir: Path, output_base: Path) -> list[dict[str, Any]]:
    stores = ps.zarr_paths(input_dir.resolve())
    records = []
    for store in stores:
        run_dir = store.parent
        cache_path = sup.cache_dir(output_base, run_dir) / f"{run_dir.name}_{LAYER}_overlap.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing Dense_0 overlap cache for {run_dir.name}: {cache_path}. "
                "Run analyze_superposition.py first."
            )
        records.append(sup.load_cache(cache_path))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MP permutation-null corrected Dense_0 eigenspectrum plots "
            "from analyze_superposition overlap caches."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures_analyze_superposition"))
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--force", action="store_true", help="Recompute cached MP permutation-null spectra.")
    parser.add_argument(
        "--mp-ratio-threshold",
        dest="mp_ratio_threshold",
        type=float,
        default=DETACHED_RATIO_DEFAULT,
        help=(
            "Detached-mode threshold for the leading-prefix rank-wise ratio "
            "lambda_k / mu_k, where mu_k is the mean eigenvalue at rank k "
            "from the decorrelated null spectra. Counting stops at the first "
            "rank that does not exceed this threshold."
        ),
    )
    parser.add_argument(
        "--mp-z-threshold",
        dest="mp_ratio_threshold",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--spectra-only",
        action="store_true",
        help="Regenerate MP-corrected eigenspectrum plots without recomputing reconstruction losses.",
    )
    parser.add_argument(
        "--best-fraction-only",
        action="store_true",
        help=(
            "Regenerate only best_test_loss_vs_detached_mode_fraction_Dense_0.png "
            "from cached best-checkpoint reconstruction results."
        ),
    )
    parser.add_argument(
        "--best-fraction-overlay-only",
        action="store_true",
        help=(
            "Regenerate only best_test_loss_vs_detached_mode_fraction_all_widths_Dense_0.png "
            "from cached best-checkpoint reconstruction results."
        ),
    )
    parser.add_argument(
        "--reconstruction-batch-size",
        type=int,
        default=1024,
        help="Batch size used when evaluating best-checkpoint spectral reconstructions.",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Deprecated and ignored; MP spectra and reconstruction plots always use log y.",
    )
    parser.add_argument(
        "--linear-x",
        action="store_true",
        help="Use a linear x-axis instead of the default log-rank x-axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicates <= 0:
        raise ValueError("--replicates must be positive")
    if args.reconstruction_batch_size <= 0:
        raise ValueError("--reconstruction-batch-size must be positive")
    if not np.isfinite(float(args.mp_ratio_threshold)):
        raise ValueError("--mp-ratio-threshold must be finite")
    real_records = load_real_records(args.input_dir, args.output_dir)
    if args.best_fraction_only or args.best_fraction_overlay_only:
        reconstruction_records = load_cached_best_reconstruction_records(
            args.input_dir,
            args.output_dir,
            real_records,
            replicates=args.replicates,
            seed=args.seed,
            batch_size=args.reconstruction_batch_size,
            z_threshold=args.mp_ratio_threshold,
        )
        if real_records:
            spectrum_dir = Path(real_records[0]["job_dir"]) / f"mp_cor_eigenvalue_spectra_{LAYER}"
            if args.best_fraction_only:
                save_best_loss_vs_mode_fraction_plot(
                    reconstruction_records,
                    real_records,
                    spectrum_dir,
                    args.dpi,
                )
                print(
                    "Saved cached best-fraction plot to "
                    f"{spectrum_dir / f'best_test_loss_vs_detached_mode_fraction_{LAYER}.png'}"
                )
            if args.best_fraction_overlay_only:
                save_best_loss_vs_mode_fraction_overlay(
                    reconstruction_records,
                    real_records,
                    spectrum_dir,
                    args.dpi,
                )
                print(
                    "Saved cached all-widths best-fraction plot to "
                    f"{spectrum_dir / f'best_test_loss_vs_detached_mode_fraction_all_widths_{LAYER}.png'}"
                )
        return
    mp_records = []
    stores_by_run = {store.parent.name: store for store in ps.zarr_paths(args.input_dir.resolve())}
    for record in ps.sorted_series_entries(real_records):
        store = stores_by_run[record["run"]]
        mp_records.append(
            compute_or_load_mp_null(
                store,
                args.output_dir,
                record,
                replicates=args.replicates,
                seed=args.seed,
                z_threshold=args.mp_ratio_threshold,
                force=args.force,
            )
        )
    save_mp_spectrum_plots(
        real_records,
        mp_records,
        dpi=args.dpi,
        log_y=True,
        log_x=not args.linear_x,
        z_threshold=args.mp_ratio_threshold,
    )
    if real_records:
        mp_by_run = {record["run"]: record for record in mp_records}
        spectrum_dir = Path(real_records[0]["job_dir"]) / f"mp_cor_eigenvalue_spectra_{LAYER}"
        if args.spectra_only:
            save_best_test_loss_spectrum_plot(
                best_checkpoint_summary_records(stores_by_run, real_records),
                real_records,
                mp_records,
                spectrum_dir,
                args.dpi,
                log_y=True,
                log_x=not args.linear_x,
                stores_by_run=stores_by_run,
                replicates=args.replicates,
                seed=args.seed,
                z_threshold=args.mp_ratio_threshold,
            )
            return
        data_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]] = {}
        reconstruction_records = []
        for real in ps.sorted_series_entries(real_records):
            record = compute_or_load_best_reconstruction(
                stores_by_run[real["run"]],
                args.output_dir,
                real,
                mp_by_run[real["run"]],
                replicates=args.replicates,
                seed=args.seed,
                batch_size=args.reconstruction_batch_size,
                data_cache=data_cache,
                z_threshold=args.mp_ratio_threshold,
                force=args.force,
            )
            if record is not None:
                reconstruction_records.append(record)
        save_best_reconstruction_plots(
            reconstruction_records,
            real_records,
            spectrum_dir,
            args.dpi,
        )
        save_best_test_loss_spectrum_plot(
            reconstruction_records,
            real_records,
            mp_records,
            spectrum_dir,
            args.dpi,
            log_y=True,
            log_x=not args.linear_x,
            stores_by_run=stores_by_run,
            replicates=args.replicates,
            seed=args.seed,
            z_threshold=args.mp_ratio_threshold,
        )
        time_reconstruction_records = []
        for real in ps.sorted_series_entries(real_records):
            record = compute_or_load_time_reconstruction(
                stores_by_run[real["run"]],
                args.output_dir,
                real,
                mp_by_run[real["run"]],
                replicates=args.replicates,
                seed=args.seed,
                batch_size=args.reconstruction_batch_size,
                data_cache=data_cache,
                z_threshold=args.mp_ratio_threshold,
                force=args.force,
            )
            if record is not None:
                time_reconstruction_records.append(record)
        save_time_reconstruction_loss_plot(
            time_reconstruction_records,
            real_records,
            spectrum_dir,
            args.dpi,
        )


if __name__ == "__main__":
    main()
