#!/bin/bash

# source this file to set the environment variables for the current cluster
# e.g., `source cluster_env.sh`
# or `source cluster_env.sh --overwrite` to overwrite existing variables

_setup_cluster_env() {
    local overwrite_vars=false
    while [[ $# -gt 0 ]]; do
        case $1 in
            --overwrite)
                overwrite_vars=true
                shift
                ;;
            *)
                echo "Warning: Unknown argument '$1' ignored" >&2
                shift
                ;;
        esac
    done

    if ! command -v jq &> /dev/null; then
        echo "Error: jq is not installed or not in your PATH. Please install jq." >&2
        return 1
    fi

    local CURRENT_HOSTNAME
    CURRENT_HOSTNAME=$(hostname)
    local CONFIG_FILE
    CONFIG_FILE="$(dirname "${BASH_SOURCE[0]}")/clusters.json"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Configuration file '$CONFIG_FILE' not found." >&2
        return 1
    fi

    local cluster_found=false
    while IFS= read -r pattern; do
        # Convert glob to regex: escape dots, replace * with .*
        regex="${pattern//./\\.}"
        regex="${regex//\*/.*}"
        if [[ "$CURRENT_HOSTNAME" =~ ^$regex$ ]]; then
            cluster_found=true
            local cluster_name
            cluster_name=$(jq -r --arg p "$pattern" '.[$p].name' "$CONFIG_FILE")
            echo "Activating environment for cluster: $cluster_name"

            # Export variables
            jq -r --arg p "$pattern" '.[$p].vars | to_entries | .[] | "export \(.key)=\"\(.value | gsub("\""; "\\\""))\""' "$CONFIG_FILE" | while IFS= read -r line; do
                var_name=$(echo "$line" | sed 's/^export \([^=]*\)=.*/\1/')
                
                if eval "[ -n \"\${${var_name}+x}\" ]"; then
                    if [ "$overwrite_vars" = true ]; then
                        echo "Warning: Overwriting existing environment variable '$var_name'" >&2
                        eval "$line"
                    else
                        echo "Warning: Environment variable '$var_name' is already set and will not be overwritten" >&2
                    fi
                else
                    eval "$line"
                fi
            done

            # Execute post_commands
            jq -r --arg p "$pattern" '.[$p].post_commands[]? // empty' "$CONFIG_FILE" | while IFS= read -r cmd; do
                eval "$cmd"
            done

            # Print messages
            jq -r --arg p "$pattern" '.[$p].messages[]? // empty' "$CONFIG_FILE" | while IFS= read -r msg; do
                echo "$msg" >&2
            done
            break
        fi
    done < <(jq -r 'keys | .[]' "$CONFIG_FILE")

    if [ "$cluster_found" = false ]; then
        echo "No matching cluster environment found for hostname '$CURRENT_HOSTNAME'" >&2
        return 1
    fi
}

_setup_cluster_env "$@"
unset -f _setup_cluster_env
