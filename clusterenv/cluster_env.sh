#!/bin/bash


_setup_cluster_env() {
    local CURRENT_HOSTNAME
    CURRENT_HOSTNAME=$(hostname)

    case "$CURRENT_HOSTNAME" in
        uan*)
            echo "Activating environment for cluster: LUMI"
            export HF_HOME="/pfs/lustrep4/projappl/project_462000963/evals/hf_data"
            export OUTPUT_DIR="/pfs/lustrep4/projappl/project_462000963/evals/${USER}"
            export VENV_DIR="/pfs/lustrep4/projappl/project_462000963/evals/.venv"
            source $VENV_DIR/bin/activate
            export UV_LINK_MODE=copy
            export PYTHONPATH="/pfs/lustrep4/projappl/project_462000963/evals/.venv"
            ;;
        *.leonardo.local)
            echo "Activating environment for cluster: LEONARDO"
            # Note: Variables for leonardo are placeholders.
            # export HF_HOME="TODO"
            # export OUTPUT_DIR="TODO"
            export UV_LINK_MODE=copy
            echo "Warning: Environment variables for leonardo are not fully configured (TODO)." >&2
            ;;
        *.jureca)
            echo "Activating environment for cluster: JURECA"
            # Note: Variables for jureca are placeholders.
            # export HF_HOME="TODO"
            # export OUTPUT_DIR="TODO"
            export UV_LINK_MODE=copy
            echo "Warning: Environment variables for jureca are not fully configured (TODO)." >&2
            ;;
        *)
            echo "No matching cluster environment found for hostname '$CURRENT_HOSTNAME'" >&2
            return 1
            ;;
    esac
}

_setup_cluster_env
unset -f _setup_cluster_env
