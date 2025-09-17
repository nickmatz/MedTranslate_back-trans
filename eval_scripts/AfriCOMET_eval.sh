
#!/bin/bash
#PBS -N Champion_AfriCOMET_Eval
#PBS -P CSCI1674
#PBS -q gpu_1
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=02:00:00
#PBS -o /mnt/lustre/users/nmatzopoulos/nmt_project/logs/champion_africomet_eval.o
#PBS -e /mnt/lustre/users/nmatzopoulos/nmt_project/logs/champion_africomet_eval.e

# --- Environment Setup ---
cd /mnt/lustre/users/nmatzopoulos/nmt_project/scripts
export HF_HOME=/mnt/lustre/users/nmatzopoulos/nmt_project/hf_cache_final
export COMET_CACHE_DIR=/mnt/lustre/users/nmatzopoulos/nmt_project/comet_cache
module purge
module load chpc/python/anaconda/3-2024.10.1 chpc/cuda/12.0/12.0
source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate /home/nmatzopoulos/pytorch_cu12_env

# --- Define Paths ---
PREDICTION_DIR="/home/nmatzopoulos/lustre/nmt_project/outputs/comet"
TEST_DATA_DIR="/home/nmatzopoulos/lustre/nmt_project/pristine_eval_data"
FWD_SOURCE="${TEST_DATA_DIR}/blocker_test.en"
FWD_REFERENCE="${TEST_DATA_DIR}/blocker_test.xh"
BCK_SOURCE="${TEST_DATA_DIR}/blocker_test.xh"
BCK_REFERENCE="${TEST_DATA_DIR}/blocker_test.en"

echo "======================================================"
echo "###      EVALUATING CHAMPION MODELS WITH AFRICOMET     ###"
echo "======================================================"

# --- Function using comet-score CLI ---
run_africomet_eval() {
    local model_nickname=$1
    local source_file=$2
    local pred_file=$3
    local ref_file=$4
    
    echo -e "\n--- Scoring: ${model_nickname} ---"
    
    AFRICOMET_SCORE=$(comet-score \
        -s "$source_file" \
        -t "$pred_file" \
        -r "$ref_file" \
        --model masakhane/africomet-stl-1.1 \
        --batch_size 16 \
        --gpus 1 \
        2>/dev/null | grep -oP 'score:\s*\K[\d.]+')
    
    echo "Model: $(basename ${pred_file})"
    echo "System-level AfriCOMET Score: ${AFRICOMET_SCORE}"
}

# --- Run Evaluations ---
echo -e "\n### FORWARD-TRANSLATION (EN -> XH) ###"
run_africomet_eval "MeMaT-Fwd-Best" "$FWD_SOURCE" "${PREDICTION_DIR}/fwd-20-Wmemat.xh" "$FWD_REFERENCE"
run_africomet_eval "FLORES-Fwd-Best" "$FWD_SOURCE" "${PREDICTION_DIR}/fwd-61-Wflores.xh" "$FWD_REFERENCE"

echo -e "\n\n### BACK-TRANSLATION (XH -> EN) ###"
run_africomet_eval "FLORES-Bck-Best" "$BCK_SOURCE" "${PREDICTION_DIR}/bck-37-Wflores.en" "$BCK_REFERENCE"
run_africomet_eval "MeMaT-Bck-Best" "$BCK_SOURCE" "${PREDICTION_DIR}/bck-41-Wmemat.en" "$BCK_REFERENCE"

echo -e "\n\n--- All AfriCOMET evaluations complete. ---"