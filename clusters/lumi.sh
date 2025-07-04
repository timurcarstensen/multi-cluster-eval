# LUMI specific environment variables

export BASE_DIR="/pfs/lustrep4/projappl/project_462000963/evals"
export HF_HOME="${BASE_DIR}/hf_data"
export EVAL_OUTPUT_DIR="${BASE_DIR}/${USER}"
export EVAL_VENV_DIR="${BASE_DIR}/.venv"
export DEFAULT_PARTITION="dev-g"
export EVAL_ACCOUNT="project_462000963"
export EVAL_TIME_LIMIT="00:30:00"
export NUM_GPU_PER_NODE=1
export UV_LINK_MODE="copy"
export PYTHONPATH="${BASE_DIR}/.venv"

source "${EVAL_VENV_DIR}/bin/activate"
