#!/bin/bash
# Run scripts/make_figures.sh as a SLURM job (recommended for the full-size thesis runs,
# whose weight snapshots are too large for a login node).
#
# Usage (from the repository root):
#   sbatch slurm/make_figures.sbatch --mup <job-dirs> --sp <job-dirs> --out figures/thesis --thesis

#SBATCH --job-name=figures
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=cluster
#SBATCH --output=slurm-figures-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
pixi run figures "$@"
