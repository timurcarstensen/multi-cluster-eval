# Multi-cluster LLM Evals

## Currently supported clusters
- LEONARDO
- LUMI
- JURECA

## Usage

1. git clone `https://github.com/OpenEuroLLM/multi-cluster-eval.git`
2. `source multi-cluster-eval/cluster_env.sh`  # sets all relevant environment variables and activates the python environment
3. Start an interactive session, e.g., `srun -p $DEFAULT_PARTITION -t $EVAL_TIME_LIMIT -a $EVAL_ACCOUNT --gres="gpu:$NUM_GPU_PER_NODE" --pty /bin/bash`
5. Run the eval: `lm_eval --model hf --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float," --tasks mmlu --output_path $EVAL_OUTPUT_DIR`

This will use the shared huggingface datasets directory at `$HF_HOME` and store the eval results in the shared output directory in `$EVAL_OUTPUT_DIR`

> Note: You may need to set your HF_TOKEN in case you get a rate limit when pulling new datasets.
