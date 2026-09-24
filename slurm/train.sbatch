#!/bin/bash
# SLURM array job that trains one width config per array task.
# Submit it through slurm/submit_training.sh, which sets CONFIG_DIR and the array size.

################ SLURM HEADER ################
#SBATCH --job-name=train
#SBATCH --mail-type=ALL,ARRAY_TASKS

#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Adapt partition and GPU type to your cluster.
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

################ BASH SCRIPT #################

# --- Paths --------------------------------------------------------------------
REPO_DIR="${SLURM_SUBMIT_DIR}"             # submit_training.sh submits from the repository root
PROGRAM="src/run_experiment.py"
PIXI_ENV="${PIXI_ENV:-gpu}"                # pixi environment: gpu (CUDA 12), gpu13 or default (CPU)

# CONFIG_DIR is passed in from submit_training.sh and must contain a "configs/" folder.
: "${CONFIG_DIR:?CONFIG_DIR must be provided via submit_training.sh}"
CONFIG_ROOT="${CONFIG_DIR}/configs"

# --- Per-job output dir ------------------------------------------------------
OUTPUTDIR="${CONFIG_DIR}/logs/job-${SLURM_ARRAY_JOB_ID}"
SLURMOUTDIR="${OUTPUTDIR}/slurm_output"

mkdir -p "${SLURMOUTDIR}"
exec > "${SLURMOUTDIR}/task-${SLURM_ARRAY_TASK_ID}.out" 2>&1

# --- Select the config of this array task ------------------------------------
mapfile -t CONFIGS < <(
    ls -1 "${CONFIG_ROOT}/"*.yaml 2>/dev/null |
    grep -v 'master_config.yaml' |
    sort -V
    )

TOTAL=${#CONFIGS[@]}
IDX=$((SLURM_ARRAY_TASK_ID - 1))

if (( TOTAL == 0 )); then
  echo "No config files found in ${CONFIG_ROOT}."
  exit 1
fi
if (( IDX < 0 || IDX >= TOTAL )); then
  echo "Task index ${SLURM_ARRAY_TASK_ID} is out of range (have ${TOTAL} configs). Exiting gracefully."
  exit 0
fi

SELECTED_CONFIG="${CONFIGS[$IDX]}"
echo "This task will run config: ${SELECTED_CONFIG}"
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID NODE=$SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv || true

# --- Simulation Info ---------------------------------------------------------
echo "# Job info"
echo "@job_name          ${SLURM_JOB_NAME}"
echo "@job_id            ${SLURM_JOB_ID}"
echo "@array_job_id      ${SLURM_ARRAY_JOB_ID}"
echo "@array_task_id     ${SLURM_ARRAY_TASK_ID}"
echo "@start_date        $(date)"
echo "@host              ${SLURMD_NODENAME}"

# --- Run the training ----------------------------------------------------------
cd "${REPO_DIR}"
set +e
pixi run -e "${PIXI_ENV}" python "${PROGRAM}" --path "${SELECTED_CONFIG}" --output_dir "${OUTPUTDIR}"
simulation_returncode=$?
set -e

if [ ${simulation_returncode} == 0 ]; then
    echo "Simulation was successful"
else
    echo "Simulation exited with ${simulation_returncode}. Check error logs"
fi

exit ${simulation_returncode}
