#!/bin/bash

# Source this file to set the environment variables for the current cluster
# e.g., `source cluster_env.sh`

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

    local script_path
    script_path="${BASH_SOURCE:-$0}"
    local root_dir
    root_dir="$(dirname "$script_path")"
    local config_file="$root_dir/clusters/clusters.json"

    if [ ! -f "$config_file" ]; then
        echo "Error: Configuration file '$config_file' not found." >&2
        return 1
    fi

    local current_hostname
    current_hostname=$(hostname)

    # get all top-level keys except shared
    local clusters
    clusters=$(jq -r 'keys[] | select(. != "shared")' "$config_file")
    
    local matching_cluster
    for cluster_name in $clusters; do
        local pattern
        pattern=$(jq -r --arg cn "$cluster_name" '.[$cn].hostname_pattern' "$config_file")
        
        local regex
        regex="${pattern//./\\.}"
        regex="${regex//\*/.*}"
        
        if [[ "$current_hostname" =~ ^$regex$ ]]; then
            matching_cluster=$cluster_name
            break
        fi
    done

    if [ -z "$matching_cluster" ]; then
        echo "No matching cluster environment found for hostname '$current_hostname'" >&2
        return 1
    fi
    
    if [ "$verbose" = true ]; then
        echo "Loading environment for cluster '$matching_cluster'"
    fi

    # Merge shared and cluster-specific settings and export them
    local merged_config
    merged_config=$(jq -s --arg cn "$matching_cluster" '.[0].shared * .[0][$cn]' "$config_file")

    # Export all keys from the merged config as environment variables
    for key in $(echo "$merged_config" | jq -r 'keys[]'); do
        value=$(echo "$merged_config" | jq -r --arg k "$key" '.[$k]')
        export "$(echo "$key" | tr '[:lower:]' '[:upper:]')"="$value"
    done

    # Substitute variables
    export PYTHON_PATH="${OUTPUT_DIR}/.venv"
    export HF_HOME="${EVAL_BASE_DIR}/hf_data"
    export OUTPUT_DIR="${EVAL_BASE_DIR}/${USER}"
    export VENV_DIR="${OUTPUT_DIR}/.venv"

    # Install uv and create per-user venv if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        if [ "$verbose" = true ]; then
            echo "Creating per-user virtual environment in ${VENV_DIR}"
        fi

        # Install uv if not found
        if ! command -v uv &> /dev/null; then
            if [ "$verbose" = true ]; then
                echo "uv not found, installing via install script..."
            fi
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
        fi

        # Create venv with uv
        uv venv "$VENV_DIR" --python 3.12 --managed-python

        # Activate and install dependencies
        source "${VENV_DIR}/bin/activate"
        uv pip install -e "$root_dir"
        deactivate
    fi

    # activate the virtual environment
    if [ "$activate" = true ]; then
        if [ "$verbose" = true ]; then
            echo "Activating Python virtual environment in ${VENV_DIR}"
        fi
        # shellcheck disable=SC1090
        source "${VENV_DIR}/bin/activate"
    fi

}

_setup_cluster_env_from_bash "$@"
unset -f _setup_cluster_env_from_bash 
