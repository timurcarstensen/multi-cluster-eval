# OpenEuroLLM Evaluation Package (oellm)

A package for running evaluations across multiple HPC clusters using SLURM job arrays and Singularity containers. 

## Quick Example

**Prerequisites:**
- install [uv](https://docs.astral.sh/uv/#installation)
- Make sure you're part of the [OpenEuroLLM](https://huggingface.co/OpenEuroLLM) organization on HuggingFace and are logged in (i.e., `HF_TOKEN` is set in your environment). 

```bash
# Install the package
uv tool install git+https://github.com/OpenEuroLLM/multi-cluster-eval.git

# Run evaluations on multiple models and tasks
oellm schedule-eval \
    --models "microsoft/DialoGPT-medium,EleutherAI/pythia-160m" \
    --tasks "hellaswag,mmlu" \
    --n_shot 5
```

This will automatically:
- Detect your current HPC cluster (Leonardo, LUMI, or JURECA)
- Download and cache the specified models and datasets
- Generate a SLURM job array to evaluate all model-task combinations
- Submit the jobs with appropriate cluster-specific resource allocations


## Installation

Install directly from the git repository using uv:

```bash
uv tool install git+https://github.com/OpenEuroLLM/multi-cluster-eval.git
```

This makes the `oellm` command available globally in your shell.

### Dataset Pre-downloading

`oellm` automatically pre-downloads evaluation datasets to ensure they're available offline on compute nodes. This feature:

- Uses the HuggingFace `datasets` library (already included as a dependency)
- Works without requiring `torch` or `lm-eval-harness` 
- Supports all major evaluation tasks (HellaSwag, MMLU, GSM8K, ARC, etc.)
- Gracefully handles unknown tasks by skipping them with a warning
- Uses cached datasets when available to avoid re-downloading

The pre-download happens automatically before job submission and covers the most commonly used evaluation benchmarks.

### Updating the Package

If you've already installed the package, you can run the following command to update it:
```bash
uv tool upgrade oellm
```

If you had previously installed the package from a different source and would like to overwrite it, you can run the following command:
```bash
uv tool install git+https://github.com/OpenEuroLLM/multi-cluster-eval.git --force
```

## High-Level Evaluation Workflow

The `oellm` package orchestrates distributed LLM evaluations through the following workflow:

### 1. **Cluster Auto-Detection**
- Automatically detects the current HPC cluster based on hostname patterns
- Loads cluster-specific configurations from `clusters.json` including:
  - SLURM partition and account settings
  - Shared storage paths for models, datasets, and results
  - GPU allocation and queue limits
  - Singularity container specifications

### 2. **Resource Preparation**
- **Model Handling**: Processes both local model checkpoints and Hugging Face Hub models
  - For local paths: Automatically discovers and expands training checkpoint directories
  - For HF models: Pre-downloads to shared cache (`$HF_HOME`) for offline access on compute nodes
- **Dataset Caching**: Pre-downloads all evaluation datasets using lm-evaluation-harness TaskManager
- **Container Management**: Ensures the appropriate Singularity container is available for the target cluster

### 3. **Job Generation & Scheduling**
- Creates a comprehensive CSV manifest of all model-task-shot combinations
- Generates a SLURM batch script from a template with cluster-specific parameters
- Submits a job array where each array task processes a subset of evaluations
- Respects queue limits and current user load to avoid overwhelming the scheduler

### 4. **Distributed Execution**
- Each SLURM array job runs in a Singularity container with:
  - GPU access (NVIDIA CUDA or AMD ROCm as appropriate)
  - Mounted shared storage for models, datasets, and output
  - Offline execution using pre-cached resources
- Uses `lm-evaluation-harness` for the actual model evaluation
- Outputs results as JSON files

### 5. **Output Organization**
Results are organized in timestamped directories under `$EVAL_OUTPUT_DIR/$USER/`:
```
2024-01-15-14-30-45/
├── jobs.csv              # Complete evaluation manifest
├── submit_evals.sbatch    # Generated SLURM script
├── slurm_logs/           # SLURM output/error logs
└── results/              # Evaluation JSON outputs
```

## Supported Clusters

Currently supports three HPC clusters:

- **LEONARDO** - NVIDIA A100 GPUs (CUDA)
- **LUMI** - AMD MI250X GPUs (ROCm)
- **JURECA** - NVIDIA A100 GPUs (CUDA)

Each cluster has pre-configured:
- Shared evaluation directories with appropriate quotas
- Optimized Singularity containers with evaluation dependencies
- Account and partition settings for the OpenEuroLLM project

## Advanced Usage

### Custom Evaluation Configurations
For complex evaluations, create a CSV file with specific model arguments:

```csv
model_path,task_path,n_shot
"EleutherAI/pythia-160m,revision=step100000,dtype=float16",hellaswag,0
"/path/to/local/checkpoint",mmlu,5
```

```bash
oellm schedule-eval --eval_csv_path custom_evals.csv
```

### Development and Testing
Run in download-only mode to prepare resources without submitting jobs:

```bash
oellm schedule-eval --models "EleutherAI/pythia-160m" --tasks "hellaswag" --n_shot 0 --download_only
```
