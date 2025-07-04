# LEONARDO specific environment variables

export EVAL_BASE_DIR="/leonardo_work/EUHPC_E03_068/shared_evals"
export HF_HOME="${EVAL_BASE_DIR}/hf_data"
export EVAL_OUTPUT_DIR="${EVAL_BASE_DIR}/${USER}"
export EVAL_VENV_DIR="${EVAL_BASE_DIR}/.venv"
export DEFAULT_PARTITION="boost_usr_prod"
export EVAL_ACCOUNT="AIFAC_L01_028"
export EVAL_TIME_LIMIT="00:30:00"
export NUM_GPU_PER_NODE=1
export UV_LINK_MODE="copy"
export PYTHONPATH="${EVAL_BASE_DIR}/.venv"

source "${EVAL_VENV_DIR}/bin/activate"
