#!/bin/bash
#SBATCH -A PAS2159
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logFiles/slurm_convert_%A_%a.out
#SBATCH --error=logFiles/slurm_convert_%A_%a.err
#####XYZSBATCH --array=0
#SBATCH --array=0-49

mkdir -p logFiles

source /users/PAS1043/osu7903/MLAND/run_setup.sh

# Target run number (passed as 1st CLI argument or read from env)
RUN_NUMBER=${1:-$run}

if [ -z "$RUN_NUMBER" ]; then
    echo "Error: No run number specified. Pass it as: sbatch run_convert.sh <RUN_NUMBER>"
    exit 1
fi

TOTAL_PARTS=50
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

#export ANITA_DATA=/fs/ess/PAS2159/shared_code/
export ANITA_DATA=/fs/ess/PAS2159/
export ANALYZED_RUNS_DIR=/fs/ess/PAS2608/anita_sim/single_energies/analyzed/
export SCRIPTS_DIR=/fs/ess/PAS2608/anita_sim/standard_sim/scripts
export ANITA_MC_DATA=/fs/scratch/PAS2159/HughesLab/ANITA_DATA/simulated/$energy
export UCORRELATOR_TIMEAVG_DIR=/fs/ess/PAS2608/anita_sim/spectrumAverages

echo "Running part ${TASK_ID} of ${TOTAL_PARTS} for Run ${RUN_NUMBER} (Job ID: ${SLURM_JOB_ID})"
echo "Output files will be placed in /fs/scratch/PAS3311/anita_data/anita_data_parquet/run${RUN_NUMBER}_part${TASK_ID:02d}of${TOTAL_PARTS:02d}_calibrated_maps.parquet"
python -u convert_anita_data_to_parquet_calibrated_array.py \
    --ANITA_ROOT_DATA /fs/ess/PAS2159/ \
    --run ${RUN_NUMBER} \
    --num_parts ${TOTAL_PARTS} \
    --part_index ${TASK_ID} \
    >& logFiles/run_convert_${RUN_NUMBER}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.log