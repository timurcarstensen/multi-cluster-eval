#!/bin/bash

# Source this file to set the environment variables for the current cluster
# e.g., `source cluster_env_using_bash.sh`

_setup_cluster_env_from_bash() {
    local CURRENT_HOSTNAME
    CURRENT_HOSTNAME=$(hostname)
    local CLUSTERS_DIR
    CLUSTERS_DIR="$(dirname "${BASH_SOURCE[0]}")/clusters"

    if [ ! -d "$CLUSTERS_DIR" ]; then
        echo "Error: Clusters directory '$CLUSTERS_DIR' not found." >&2
        return 1
    fi

    local cluster_found=false
    for cluster_script in "$CLUSTERS_DIR"/*.sh; do
        if [ ! -f "$cluster_script" ]; then
            continue
        fi

        # The glob pattern is the filename without the .sh extension
        local pattern
        pattern=$(basename "$cluster_script" .sh)

        # Using == for glob matching against the hostname
        if [[ "$CURRENT_HOSTNAME" == $pattern ]]; then
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