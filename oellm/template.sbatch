#!/bin/bash
#SBATCH --job-name=oellm-eval
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:$NUM_GPU_PER_NODE
#SBATCH --output=logs/%x-%A-%a.out # TODO 
#SBATCH --partition=$DEFAULT_PARTITION
#SBATCH --account=$EVAL_ACCOUNT
#SBATCH --error=logs/%x-%A-%a.err # TODO
#SBATCH --array=0-{array_limit}%{max_array_len}


CSV_PATH="{csv_path}"
NUM_JOBS={num_jobs}

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "CSV Path: $CSV_PATH"
echo "Number of concurrent jobs: $NUM_JOBS"

TOTAL_EVALS=$(($(wc -l < "$CSV_PATH") - 1))
echo "Total evaluations to run: $TOTAL_EVALS"

if [ "$TOTAL_EVALS" -lt 1 ]; then
    echo "No evaluations to run. Exiting."
    exit 0
fi

EVALS_PER_JOB=$(((TOTAL_EVALS + NUM_JOBS - 1) / NUM_JOBS))

START_INDEX=$((SLURM_ARRAY_TASK_ID * EVALS_PER_JOB + 1))
END_INDEX=$(((SLURM_ARRAY_TASK_ID + 1) * EVALS_PER_JOB))

if [ "$END_INDEX" -gt "$TOTAL_EVALS" ]; then
    END_INDEX=$TOTAL_EVALS
fi

echo "This job will process evaluations from index $START_INDEX to $END_INDEX."

if [ "$START_INDEX" -gt "$END_INDEX" ]; then
    echo "No evaluations for this job to run. Exiting."
    exit 0
fi

# Use `tail` and `head` to slice the CSV file for the tasks assigned to this job.
# The +1 on START_INDEX accounts for the header row.
tail -n +$((START_INDEX + 1)) "$CSV_PATH" | head -n $((END_INDEX - START_INDEX + 1)) | \
while IFS=, read -r model_path task_path n_shot
do
    # Remove trailing carriage returns if script is edited on Windows
    model_path=$(echo "$model_path" | tr -d '\r')
    task_path=$(echo "$task_path" | tr -d '\r')
    n_shot=$(echo "$n_shot" | tr -d '\r')

    # Skip empty lines
    if [ -z "$model_path" ]; then
        continue
    fi

    echo "----------------------------------------------------"
    echo "Starting evaluation for:"
    echo "  Model: $model_path"
    echo "  Task: $task_path"
    echo "  N-shot: $n_shot"
    echo "----------------------------------------------------"

    $EVAL_VENV_DIR/bin/python -m lm_eval --model hf \
        --model_args pretrained="$model_path" \
        --tasks "$task_path" \
        --num_fewshot "$n_shot" \
        --output_path $EVAL_OUTPUT_DIR \
        --log_samples

    echo "Evaluation finished for model: $model_path"

done

echo "Job $SLURM_ARRAY_TASK_ID finished."
