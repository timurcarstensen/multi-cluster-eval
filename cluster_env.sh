#!/bin/bash

# Source this file to set the environment variables for the current cluster
# e.g., `source cluster_env.sh`

# map from hostname pattern to cluster script name
map=(
    "*.leonardo.local:leonardo.sh"
    "*.jureca:jureca.sh"
    "uan*:lumi.sh"
)

_set_common_env_vars() {
    export EVAL_TIME_LIMIT="00:30:00"
    export NUM_GPU_PER_NODE=1
    export UV_LINK_MODE="copy"
    export PYTHONPATH="${EVAL_BASE_DIR}/.venv"
    export HF_HOME="${EVAL_BASE_DIR}/hf_data"
    export EVAL_OUTPUT_DIR="${EVAL_BASE_DIR}/${USER}"
    export EVAL_VENV_DIR="${EVAL_BASE_DIR}/.venv"
}

_setup_cluster_env_from_bash() {
    local verbose=false
    local activate=false
    for arg in "$@"; do
        case $arg in
            --verbose)
            verbose=true
            ;;
            --activate)
            activate=true
            ;;
        esac
    done

    local CURRENT_HOSTNAME
    CURRENT_HOSTNAME=$(hostname)
    local CLUSTERS_DIR
    local script_path="${BASH_SOURCE:-$0}"
    CLUSTERS_DIR="$(dirname "$script_path")/clusters"

    if [ ! -d "$CLUSTERS_DIR" ]; then
        echo "Error: Clusters directory '$CLUSTERS_DIR' not found." >&2
        return 1
    fi

    local cluster_found=false
    local cluster_script
    for entry in "${map[@]}"; do
        local pattern="${entry%%:*}"
        local cluster_script_name="${entry#*:}"
        regex="${pattern//./\\.}"
        regex="${regex//\*/.*}"
        if [[ "$CURRENT_HOSTNAME" =~ ^$regex$ ]]; then
            cluster_script="$CLUSTERS_DIR/$cluster_script_name"

            if [ ! -f "$cluster_script" ] && [ "$verbose" = true ]; then
                echo "Error: Cluster script '$cluster_script' not found for pattern '$pattern'." >&2
                return 1
            fi
            cluster_found=true
            break
        fi
    done
    
    if [ "$cluster_found" = false ] && [ "$verbose" = true ]; then
        echo "No matching cluster environment script found for hostname '$CURRENT_HOSTNAME' in '$CLUSTERS_DIR'" >&2
        return 1
    fi

    if [ "$verbose" = true ]; then
        echo "Activating environment from $cluster_script"
    fi
    
    # source the cluster script for environment variables
    source "$cluster_script"
    
    # export shared environment variables
    _set_common_env_vars

    # activate the virtual environment
    if [ "$activate" = true ]; then
        if [ "$verbose" = true ]; then
            echo "Activating Python virtual environment in ${EVAL_VENV_DIR}"
        fi
        source "${EVAL_VENV_DIR}/bin/activate"
    fi

}

_setup_cluster_env_from_bash "$@"
unset -f _setup_cluster_env_from_bash
unset map 
