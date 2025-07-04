# Multi-cluster LLM

This package allows to set environment variables in the same way across EuroHPC clusters.
The goal is to avoid duplicating pre-training/evaluation scripts and rather only depends on cluster-specific environment variables.

Currently, we support the following environment variables which allows to run easily evaluations across clusters:
* DEFAULT_PARTITION
* DEFAULT_ACCOUNT
* NUM_GPU_PER_NODE
* HF_HOME
* EVAL_BASE_DIR
* EVAL_OUTPUT_DIR
* EVAL_VENV_DIR

We also aim to support the following:
* Modules to initialize
* C++ env variable for LUMI
* Container setup
* Number of hours for maximum job
* Maximum memory to set per node
* Default Log location
* List of faulty nodes to be excluded
* Wandb account


The following clusters are currently supported:
- ✅ LEONARDO
- ✅ LUMI
- ✅ JURECA

## Usage

To install do:
```bash
git clone https://github.com/OpenEuroLLM/multi-cluster-eval.git ~/multi-cluster-eval
```

You can then run:

```bash
# Setup environment variables
source multi-cluster-eval/cluster_env.sh

# Starts an interactive session with gpu on the default partition/account for the cluster
srun -p $DEFAULT_PARTITION -t $EVAL_TIME_LIMIT --account $DEFAULT_ACCOUNT --gres="gpu:$NUM_GPU_PER_NODE" --pty /bin/bash

# Once the node is available, sets environment variables spin-up an evaluation
source multi-cluster-eval/cluster_env.sh --activate
lm_eval --model hf --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float," --tasks mmlu --output_path $EVAL_OUTPUT_DIR
```

This will use the shared huggingface datasets directory at `$HF_HOME` and store the eval results in the shared output directory in `$EVAL_OUTPUT_DIR`

> ⚠️ The tool assumes that you are into the logging node of one of the EuroHPC cluster. 
> ⚠️ You may need to set your HF_TOKEN in case you get a rate limit when pulling new datasets.