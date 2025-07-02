# Multi-cluster LLM Evals

## Currently supported clusters
- LEONARDO
- LUMI
- JURECA

## Usage
> Make sure that your HF_TOKEN is in your environment, otherwise you may be rated limited when pulling new evaluation datasets

1. git clone `https://github.com/OpenEuroLLM/multi-cluster-eval.git`
2. `source multi-cluster-eval/cluster_env.sh  # sets all relevant environment variables and activates the python environment
4. Start an interactive session, e.g., `srun -p <partition-name> -t <time-limit> --gres="gpu:1" --pty <your-shell e.g., /bin/bash>`
5. Run the eval: `lm_eval --model hf --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float," --tasks mmlu --output_path $OUTPUT_DIR`

This will use the shared huggingface datasets directory at `$HF_HOME` and store the eval results in the shared output directory in `$OUTPUT_DIR`
