# Multi-cluster LLM Evals

## Currently supported clusters
- LEONARDO
- LUMI
- JURECA

## Usage

1. git clone `https://github.com/OpenEuroLLM/multi-cluster-eval.git`
2. `source multi-cluster-eval/cluster_env.sh`  # sets all relevant environment; pass `--activate` to activate the shared python environment
3. Start an interactive session, e.g., `srun -p $DEFAULT_PARTITION -t $EVAL_TIME_LIMIT -a $EVAL_ACCOUNT --gres="gpu:$NUM_GPU_PER_NODE" --pty /bin/bash`
4. Run the eval: `lm_eval --model hf --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float," --tasks mmlu --output_path $EVAL_OUTPUT_DIR`

This will use the shared huggingface datasets directory at `$HF_HOME` and store the eval results in the shared output directory in `$EVAL_OUTPUT_DIR`

> Note: You may need to set your HF_TOKEN in case you get a rate limit when pulling new datasets.


## Setting up a shared python environment
> make sure you have `uv` installed and are in a directory that you can share with other users in your project

1. Install python into `.python`: `uv python install -i .python 3.12`
2. Create a virtual environment: `uv venv -p .python/path/to/pythonbinary --relocatable`
3. Activate the environment: `source .venv/bin/activate`
4. Clone lm-eval: `git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness`
5. Install the dependencies: `cd lm-evaluation-harness && pip install -e . datasets<4.0.0 torch transformers accelerate sentencepiece wandb`
