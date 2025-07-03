#!/bin/bash

# Source this file to set the environment variables for the current cluster
# e.g., `source cluster_env_using_bash.sh`

# Using two indexed arrays for portability between bash and zsh.
patterns=(
    "*.leonardo.local"
    "*.jureca"
    "uan*"
)
scripts=(
    "leonardo.sh"
    "jureca.sh"
    "lumi.sh"
)

_setup_cluster_env_from_bash() {
    local CURRENT_HOSTNAME
    CURRENT_HOSTNAME=$(hostname)
    local CLUSTERS_DIR
    # Portable way to get script's directory when sourced in bash or zsh.
    # In bash, $BASH_SOURCE on its own refers to the first element of the
    # BASH_SOURCE array. In zsh, BASH_SOURCE is unset, so it falls back to $0.
    local script_path="${BASH_SOURCE:-$0}"
    CLUSTERS_DIR="$(dirname "$script_path")/clusters"


    if [ ! -d "$CLUSTERS_DIR" ]; then
        echo "Error: Clusters directory '$CLUSTERS_DIR' not found." >&2
        return 1
    fi

    local cluster_found=false
    for i in "${!patterns[@]}"; do
        local pattern="${patterns[i]}"
        # Using == for glob matching against the hostname
        if [[ "$CURRENT_HOSTNAME" == $pattern ]]; then
            local cluster_script_name="${scripts[i]}"
            local cluster_script="$CLUSTERS_DIR/$cluster_script_name"

            if [ ! -f "$cluster_script" ]; then
                echo "Error: Cluster script '$cluster_script' not found for pattern '$pattern'." >&2
                return 1
            fi
            echo "Activating environment from $cluster_script"
            # shellcheck source=/dev/null
            source "$cluster_script"
            cluster_found=true
            break
        fi
    done

    if [ "$cluster_found" = false ]; then
        echo "No matching cluster environment script found for hostname '$CURRENT_HOSTNAME' in '$CLUSTERS_DIR'" >&2
        return 1
    fi
}

_setup_cluster_env_from_bash
unset -f _setup_cluster_env_from_bash
unset patterns
unset scripts 