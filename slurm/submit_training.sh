#!/usr/bin/env bash
# Submit one SLURM array task per width config of an experiment.
#
# Usage (from anywhere):
#   bash slurm/submit_training.sh <EXPERIMENT_DIR> [additional sbatch options]
# Example:
#   bash slurm/submit_training.sh experiments/cifar10/mup
#   PIXI_ENV=default bash slurm/submit_training.sh experiments/quickstart/cifar10_mup --partition=cluster --gres=none
#
# The results are written to <EXPERIMENT_DIR>/logs/job-<SLURM_ARRAY_JOB_ID>/.
set -euo pipefail

CONFIG_DIR=${1:?"Please provide the experiment directory that contains the configs/ folder: bash slurm/submit_training.sh <EXPERIMENT_DIR>"}
shift
CONFIG_DIR="$(realpath "$CONFIG_DIR")"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Count configs (skip master)
N=$(ls -1 "$CONFIG_DIR/configs"/*.yaml 2>/dev/null | grep -v 'master_config.yaml' | wc -l | tr -d ' ')
if [ "$N" -eq 0 ]; then
  echo "No config files found in $CONFIG_DIR/configs (excluding master_config.yaml)."
  exit 1
fi

cd "$REPO_DIR"
sbatch --export=ALL,CONFIG_DIR="$CONFIG_DIR",PIXI_ENV="${PIXI_ENV:-gpu}" --array=1-"$N" "$@" slurm/train.sbatch
