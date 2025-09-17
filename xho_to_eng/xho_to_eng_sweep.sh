#!/bin/bash

# Written by Nick Matzopoulos (mtznic006@myuct.ac.za)



LOG_DIR="${PWD}/logs"
STDOUT_LOG="${LOG_DIR}/xh_to_en_sweep.o"
STDERR_LOG="${LOG_DIR}/xh_to_en_sweep.e"

# Redirect all stdout and stderr from this point on to our log files in append mode (>>)
exec >> "${STDOUT_LOG}" 2>> "${STDERR_LOG}"

# --- Environment Setup ---
export HF_HOME=${PWD}/nllb-200 # Path to baseline model to fine-tune
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# --- STATE MANAGEMENT ---
STATE_FILE="sweep_state_xho_to_eng.txt"
if [ ! -f "$STATE_FILE" ]; then
    echo "0" > "$STATE_FILE"
    echo "Initialized state file for Xho to Eng sweep."
fi

# --- HYPERPARAMETER GRID ---
LEARNING_RATES=( "5e-6" "1e-5" "3e-5" "5e-5" "5e-4" "5e-3")
EFFECTIVE_BATCH_SIZES=( 32 64 )
NUM_EPOCHS=( 3 5 7)
WARMUP_RATIOS=( "0.0" "0.1" )
WEIGHT_DECAYS=( "0.0" "0.01" )
LABEL_SMOOTHINGS=( "0.1" )
BASE_BATCH_SIZE=4

EXPERIMENTS=()
for LR in "${LEARNING_RATES[@]}"; do
  for EBS in "${EFFECTIVE_BATCH_SIZES[@]}"; do
    for EPOCHS in "${NUM_EPOCHS[@]}"; do
      for WR in "${WARMUP_RATIOS[@]}"; do
        for WD in "${WEIGHT_DECAYS[@]}"; do
          for LS in "${LABEL_SMOOTHINGS[@]}"; do
            GA=$((EBS / BASE_BATCH_SIZE))
            EXPERIMENTS+=("${LR}:${BASE_BATCH_SIZE}:${GA}:${EPOCHS}:${WR}:${WD}:${LS}")
          done
        done
      done
    done
  done
done
TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
echo "Prepared a total of ${TOTAL_EXPERIMENTS} experiments"

# --- RESILIENT EXPERIMENT LOOP ---
START_INDEX=$(cat "$STATE_FILE")
echo "Resuming sweep from experiment index: ${START_INDEX}"

for (( i=${START_INDEX}; i<${TOTAL_EXPERIMENTS}; i++ )); do
    CONFIG_STRING=${EXPERIMENTS[$i]}
    IFS=':' read -r LR BS GA EPOCHS WR WD LS <<< "$CONFIG_STRING"

    RUN_NAME="run${i}_lr${LR}_ebs$((BS * GA))_epochs${EPOCHS}_wr${WR}_wd${WD}_ls${LS}"
    OUTPUT_DIR="finetuned_models/xh_to_en_sweep/${RUN_NAME}"

    echo ""
    echo "======================================================"
    echo "      STARTING EXPERIMENT ${i} / ${TOTAL_EXPERIMENTS}"
    echo "======================================================"
    echo "Parameters: LR=${LR}, EBS=$((BS * GA)), Epochs=${EPOCHS}, WR=${WR}, WD=${WD}, LS=${LS}"
    echo "Output Directory: ${OUTPUT_DIR}"
    echo "------------------------------------------------------"

    python train_xh_to_en.py \
        --base_model_path "${PWD}"/nllb-200 \
        --data_dir "${PWD}"/data-bin \
        --output_dir "${OUTPUT_DIR}" \
        --learning_rate "${LR}" \
        --batch_size "${BS}" \
        --num_epochs "${EPOCHS}" \
        --gradient_accumulation_steps "${GA}" \
        --warmup_ratio "${WR}" \
        --weight_decay "${WD}" \
        --label_smoothing "${LS}"

    if [ $? -eq 0 ]; then
        echo "--- EXPERIMENT ${i} FINISHED SUCCESSFULLY ---"
        echo "$((i + 1))" > "$STATE_FILE"
    else
        echo "--- !!! EXPERIMENT ${i} FAILED !!! ---"
        echo "Sweep stopped. Check logs."
        exit 1
    fi
done

echo ""
echo "=== isiXhosa to English Hyperparameter Sweep Complete ==="