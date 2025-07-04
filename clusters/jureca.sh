# JURECA specific environment variables

export BASE_DIR="/p/data1/mmlaion/oellm_shared_evals"
export HF_HOME="${BASE_DIR}/hf_data"
export EVAL_OUTPUT_DIR="${BASE_DIR}/${USER}"
export EVAL_VENV_DIR="${BASE_DIR}/.venv"
export DEFAULT_PARTITION="standard"
export EVAL_ACCOUNT="AIFAC_L01_028"
export EVAL_TIME_LIMIT="00:30:00"
export NUM_GPU_PER_NODE=1
export UV_LINK_MODE="copy"
export PYTHONPATH="${BASE_DIR}/.venv"
