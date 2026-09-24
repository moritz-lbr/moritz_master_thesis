#!/usr/bin/env bash
# Quickstart: download the data, train small muP and SP networks and create all figures.
#
# Usage (from the repository root):
#   pixi run quickstart              # CIFAR-10 and SVHN
#   pixi run quickstart cifar10      # only CIFAR-10
#
# For every dataset <ds> this
#   1. downloads the dataset to datasets/                     (datasets/download.py)
#   2. trains the widths n = 128, 256, 512 under muP and SP  (experiments/quickstart/<ds>_{mup,sp})
#      -> experiments/quickstart/<ds>_{mup,sp}/logs/job-quickstart/
#   3. runs the full analysis pipeline                        (scripts/make_figures.sh)
#      -> figures/quickstart/<ds>/
# On a CPU the whole quickstart takes roughly 30-40 minutes.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
PYTHON="${PYTHON:-python}"

DATASETS=("$@")
((${#DATASETS[@]} == 0)) && DATASETS=(cifar10 svhn)

"$PYTHON" datasets/download.py "${DATASETS[@]}"

for ds in "${DATASETS[@]}"; do
  for scheme in mup sp; do
    exp="experiments/quickstart/${ds}_${scheme}"
    printf '\n==> Training %s\n' "$exp"
    "$PYTHON" src/run_experiment.py --path "$exp" --output_dir "$exp/logs/job-quickstart"
  done
  bash scripts/make_figures.sh \
    --mup "experiments/quickstart/${ds}_mup/logs/job-quickstart" \
    --sp "experiments/quickstart/${ds}_sp/logs/job-quickstart" \
    --out "figures/quickstart/${ds}"
done

printf '\nQuickstart finished. Figures: %s\n' "$(printf 'figures/quickstart/%s ' "${DATASETS[@]}")"
