#!/usr/bin/env bash
# Create all data-driven figures of the thesis from trained job directories.
#
# Usage (from the repository root, inside the pixi environment):
#   pixi run figures --mup <job-dir> [<job-dir> ...] --sp <job-dir> [<job-dir> ...] \
#                    --out <figure-dir> [--thesis] [--force]
#
#   --mup / --sp  job directories (experiments/<exp>/logs/job-*) of the muP and SP runs.
#                 Several jobs per parametrization (independent seeds) are averaged in
#                 the loss / update-norm figures; the weight-based analyses (Figs. 3.1,
#                 5.7-5.12) use the first job given.
#   --out         output directory (default: figures/<name of first muP job>)
#   --thesis      use the fixed axis limits of the thesis figures and drop the n=5793
#                 runs, as in Figs. 5.1-5.3
#   --force       recompute cached analysis results
#
# Result:
#   <out>/fig*.png   one file (or one per parametrization) for every thesis figure
#   <out>/details/<mup|sp>/   all outputs of the individual analysis scripts
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

MUP_JOBS=()
SP_JOBS=()
OUT=""
THESIS=0
FORCE=()
mode=""
while (($# > 0)); do
  case "$1" in
    --mup) mode="mup" ;;
    --sp) mode="sp" ;;
    --out) OUT="$2"; shift ;;
    --thesis) THESIS=1 ;;
    --force) FORCE=(--force) ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)
      case "$mode" in
        mup) MUP_JOBS+=("$(realpath "$1")") ;;
        sp) SP_JOBS+=("$(realpath "$1")") ;;
        *) echo "Job directory '$1' must follow --mup or --sp" >&2; exit 1 ;;
      esac
      ;;
  esac
  shift
done

if ((${#MUP_JOBS[@]} == 0 && ${#SP_JOBS[@]} == 0)); then
  echo "Provide at least one job directory via --mup and/or --sp (see --help)." >&2
  exit 1
fi
if [[ -z "$OUT" ]]; then
  first_job="${MUP_JOBS[0]:-${SP_JOBS[0]}}"
  OUT="figures/$(basename "$first_job")"
fi
mkdir -p "$OUT/details"
OUT="$(realpath "$OUT")"

LOSS_ARGS=()
NORM_ARGS=()
if ((THESIS)); then
  LOSS_ARGS=(--limits thesis --exclude-width 5793)
  NORM_ARGS=(--exclude-width 5793)
fi

cd "$REPO_DIR"
V=visualizations

step() { printf '\n==> %s\n' "$*"; }

# Copies the first file matching a glob pattern to a target name (warns if missing).
collect() {
  local target="$1"; shift
  local match
  match="$(compgen -G "$1" | sort -V | head -n 1 || true)"
  if [[ -n "$match" ]]; then
    cp "$match" "$OUT/$target"
    echo "  $target  <-  ${match#"$OUT/"}"
  else
    echo "  WARNING: no file matches $1 (skipped $target)"
  fi
}

# Returns the run directory with the largest hidden width inside a job directory.
largest_run() {
  find "$1" -maxdepth 1 -type d -name 'scheme-*_widths-*' \
    | awk -F'widths-' '{split($2, w, "x"); print w[1] "\t" $0}' | sort -n | tail -n 1 | cut -f2
}

task_of() {
  "$PYTHON" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['training']['training_data']['task'].rstrip('/').split('/')[-1])" \
    "$(find "$1" -name simulation_config.yaml | sort | head -n 1)"
}

step "Layer-wise weight-update metrics (training_analysis/compute_metrics.py)"
"$PYTHON" training_analysis/compute_metrics.py --log-dir "${MUP_JOBS[@]}" "${SP_JOBS[@]}" --layers Dense_0 Dense_1 "${FORCE[@]}"

for scheme in mup sp; do
  if [[ "$scheme" == mup ]]; then jobs=("${MUP_JOBS[@]}"); else jobs=("${SP_JOBS[@]}"); fi
  ((${#jobs[@]} == 0)) && continue
  job="${jobs[0]}"
  job_name="$(basename "$job")"
  DETAILS="$OUT/details/$scheme"
  task="$(task_of "$job")"
  if [[ "$scheme" == mup ]]; then
    fig_norm=fig5_1; fig_loss=fig5_2; fig_sup=fig5_7; fig_eig=fig5_9
  else
    fig_norm=fig5_3; fig_loss=fig5_3; fig_sup=fig5_8; fig_eig=fig5_10
  fi

  step "[$scheme] Update norms and test loss (Figs. 5.1-5.3)"
  "$PYTHON" $V/plot_observables.py update-norms --log-dirs "${jobs[@]}" "${NORM_ARGS[@]}" \
    --output "$OUT/${fig_norm}_update_norms_${scheme}.png"
  "$PYTHON" $V/plot_observables.py test-loss --log-dirs "${jobs[@]}" "${LOSS_ARGS[@]}" \
    --output "$OUT/${fig_loss}_test_loss_${scheme}.png"
  "$PYTHON" $V/plot_update_norms.py --log-dir "$job" --output-dir "$DETAILS/update_norms"

  step "[$scheme] Hidden-layer activation changes (Fig. 3.1)"
  "$PYTHON" $V/analyze_acts.py --log-dir "$job" --output-dir "$DETAILS/analyze_acts"
  collect "fig3_1_activation_deltas_${scheme}.png" "$DETAILS/analyze_acts/${job_name}_dense0_activation_deltas.png"

  step "[$scheme] Superposition matrix (Figs. 5.7, 5.8)"
  "$PYTHON" $V/analyze_superposition.py --input-dir "$job" --output-dir "$DETAILS/superposition" \
    --best-input-space "${FORCE[@]}"
  run="$(basename "$(largest_run "$job")")"
  frames="$DETAILS/superposition/$task/$job_name/$run/Dense_0/input_space_frames"
  collect "${fig_sup}_superposition_${scheme}_step0.png" "$frames/input_space_000_step_0.png"
  collect "${fig_sup}_superposition_${scheme}_best_test_loss.png" "$frames/input_space_best_test_loss_step_*.png"

  step "[$scheme] Detached eigenvalues and PCA reconstruction (Figs. 5.9-5.12)"
  "$PYTHON" $V/mp_spectra_corr.py --input-dir "$job" --output-dir "$DETAILS/superposition" "${FORCE[@]}"
  spectra="$DETAILS/superposition/$task/$job_name/mp_cor_eigenvalue_spectra_Dense_0"
  collect "${fig_eig}_eigenvalues_${scheme}_step0.png" "$spectra/eigenvalues_step_0.png"
  collect "${fig_eig}_eigenvalues_${scheme}_best_test_loss.png" "$spectra/eigenvalues_best_test_loss_Dense_0.png"
  collect "fig5_11_relative_reconstruction_error_${scheme}.png" "$spectra/best_test_loss_vs_detached_mode_fraction_Dense_0.png"
  collect "fig5_12_reconstruction_best_test_loss_${scheme}.png" "$spectra/best_test_loss_vs_detached_mode_fraction_all_widths_Dense_0.png"
done

step "Done. Thesis figures in $OUT"
ls -1 "$OUT"
