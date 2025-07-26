# Standard library imports
import json
import logging
import os
import re
import socket
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path
from string import Template
from typing import Iterable

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from jsonargparse import auto_cli
from rich.console import Console
from rich.logging import RichHandler


def ensure_singularity_image(image_name: str) -> None:
    # TODO: switch to OELLM dataset repo once it is created
    hf_repo = os.environ.get("HF_SIF_REPO", "timurcarstensen/testing")
    hf_hub_download(
        repo_id=hf_repo,
        filename=image_name,
        repo_type="dataset",
        local_dir=os.getenv("EVAL_BASE_DIR"),
    )

    logging.info(
        "Singularity image ready at %s",
        Path(os.getenv("EVAL_BASE_DIR")) / os.getenv("EVAL_CONTAINER_IMAGE"),
    )


def _setup_logging(debug: bool = False):
    rich_handler = RichHandler(
        console=Console(),
        show_time=True,
        log_time_format="%H:%M:%S",
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )

    class RichFormatter(logging.Formatter):
        def format(self, record):
            # Define colors for different log levels
            record.msg = f"{record.getMessage()}"
            return record.msg

    rich_handler.setFormatter(RichFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers = []  # Remove any default handlers
    root_logger.addHandler(rich_handler)
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)


def _load_cluster_env() -> None:
    """
    Loads the correct cluster environment variables from `clusters.json` based on the hostname.
    """
    with open(Path(__file__).parent / "clusters.json", "r") as f:
        clusters = json.load(f)
    hostname = socket.gethostname()

    # First load shared environment variables
    shared_cfg = clusters.get("shared", {})

    # match hostname to the regex in the clusters.json
    for host in set(clusters.keys()) - {"shared"}:
        pattern = clusters[host]["hostname_pattern"]
        # Convert shell-style wildcards to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        if re.match(f"^{regex_pattern}$", hostname):
            cluster_cfg = clusters[host]
            break
    else:
        raise ValueError(f"No cluster found for hostname: {hostname}")

    # Combine shared and cluster-specific configs, with cluster-specific taking precedence
    # Remove hostname_pattern from the final config
    if "hostname_pattern" in cluster_cfg:
        del cluster_cfg["hostname_pattern"]

    # Set environment variables, expanding any template variables
    for k, v in cluster_cfg.items():
        # Expand template variables using existing environment variables
        os.environ[k] = str(v)

    for k, v in shared_cfg.items():
        try:
            os.environ[k] = str(v).format(**cluster_cfg)
        except KeyError as e:
            # when substituting env vars that are not in cluster_cfg but in the environment (e.g., $USER, $SHELL, etc...)
            if len(e.args) > 1:
                raise ValueError(
                    f"Env. variable substitution for {k} failed. Missing keys: {', '.join(e.args)}"
                )

            missing_key: str = e.args[0]
            os.environ[k] = str(v).format(
                **cluster_cfg, **{missing_key: os.environ[missing_key]}
            )


def _parse_user_queue_load() -> int:
    result = subprocess.run(
        "squeue -u $USER -h -t pending,running -r | wc -l",
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        try:
            return int(result.stdout.strip())
        except ValueError:
            logging.warning(f"Could not parse squeue output: {result.stdout}")
            return 0

    if result.stderr:
        logging.warning(f"squeue command produced an error: {result.stderr.strip()}")

    return 0


def _process_model_paths(models: Iterable[str]) -> dict[str, list[Path | str]]:
    """
    Processes model strings into a dict of model paths.

    Each model string can be a local path or a huggingface model identifier.
    This function expands directory paths that contain multiple checkpoints.
    """
    processed_model_paths = {}
    model_paths = []
    for model in models:
        if Path(model).exists() and Path(model).is_dir():
            # could either be the direct path to a local model checkpoint dir or a directory that contains a lot of
            # intermediate checkpoints from training of the structure: `model_name/hf/iter_1`, `model_name/hf/iter_2` ...
            # or `model_name/iter_1`, `model_name/iter_2` ...
            # The base case is that `model_name` is a directory that contains the model in a HF checkpoint format

            # Basecase: check if the directory contains a `.safetensors` file
            if any(Path(model).glob("*.safetensors")):
                model_paths.append(Path(model))

            # check if dir contains subdirs that themselves contain a `.safetensors` file
            model_path_base = (
                Path(model) / "hf" if "hf" not in Path(model).name else Path(model)
            )
            for subdir in model_path_base.glob("*"):
                if subdir.is_dir() and any(subdir.glob("*.safetensors")):
                    model_paths.append(subdir)

        else:
            logging.info(
                f"Model {model} not found locally, assuming it is a 🤗 hub model"
            )
            logging.debug(
                f"Downloading model {model} on the login node since the compute nodes may not have access to the internet"
            )

            if "," in model:
                model_kwargs = {
                    k: v
                    for k, v in [kv.split("=") for kv in model.split(",") if "=" in kv]
                }

                # The first element before the comma is the repository ID on the 🤗 Hub
                repo_id = model.split(",")[0]

                # snapshot_download kwargs
                snapshot_kwargs = {}
                if "revision" in model_kwargs:
                    snapshot_kwargs["revision"] = model_kwargs["revision"]

                try:
                    # Pre-download (or reuse cache) for the whole repository so that
                    # compute nodes can load it offline.
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=Path(os.getenv("HF_HOME")) / "hub",
                        **snapshot_kwargs,
                    )
                    model_paths.append(model)
                except Exception as e:
                    logging.debug(
                        f"Failed to download model {model} from Hugging Face Hub. Continuing..."
                    )
                    logging.debug(e)
            else:
                # Download the entire model repository to the local cache.  The
                # original identifier is kept in *model_paths* so downstream
                # code can still reference it; at runtime the files will be
                # read from cache, allowing offline execution.
                snapshot_download(
                    repo_id=model,
                    cache_dir=Path(os.getenv("HF_HOME")) / "hub",
                )
                model_paths.append(model)

        if not model_paths:
            logging.warning(
                f"Could not find any valid model for '{model}'. It will be skipped."
            )
        processed_model_paths[model] = model_paths
    return processed_model_paths


def _pre_download_task_datasets(tasks: Iterable[str]) -> None:
    """Ensure that all datasets required by the given `tasks` are present in the local 🤗 cache at $HF_HOME."""

    # Mapping of common task names to their HuggingFace dataset identifiers
    # This covers the most commonly used evaluation tasks
    TASK_TO_DATASET_MAP = {
        # Core language understanding tasks
        "hellaswag": "Rowan/hellaswag",
        "arc_easy": "allenai/ai2_arc",  # subset: ARC-Easy
        "arc_challenge": "allenai/ai2_arc",  # subset: ARC-Challenge
        "winogrande": "allenai/winogrande",
        "piqa": "ybisk/piqa",
        "boolq": "google/boolq",
        "openbookqa": "allenai/openbookqa",
        
        # Mathematical reasoning
        "gsm8k": "openai/gsm8k",
        "math": "hendrycks/competition_math",
        
        # Knowledge and reasoning
        "mmlu": "cais/mmlu", 
        "truthfulqa": "truthful_qa",
        "lambada_openai": "EleutherAI/lambada_openai",
        
        # Common sense reasoning  
        "commonsense_qa": "tau/commonsense_qa",
        "social_i_qa": "allenai/social_i_qa",
        
        # Reading comprehension
        "race": "ehovy/race",
        "squad": "rajpurkar/squad",
        "squad_v2": "rajpurkar/squad_v2",
        
        # Code tasks
        "humaneval": "openai/openai_humaneval",
        "mbpp": "google-research-datasets/mbpp",
        
        # Multilingual
        "xnli": "facebook/xnli",
        "xcopa": "cambridgeltl/xcopa",
        
        # Additional tasks that might have different naming
        "sciq": "allenai/sciq",
        "drop": "ucinlp/drop",
        "copa": "pkavumba/balanced-copa",
        "rte": "nyu-mll/glue",  # subset: RTE
        "wsc": "nyu-mll/glue",  # subset: WSC
        "cb": "nyu-mll/super_glue",  # subset: CB
        "wic": "nyu-mll/super_glue",  # subset: WiC
    }

    try:
        from datasets import load_dataset, DownloadMode  # type: ignore
    except ImportError as import_err:
        logging.warning(
            "Could not import datasets library – skipping dataset pre-download.\n%s",
            import_err,
        )
        return

    processed: set[str] = set()

    for task_name in tasks:
        if not isinstance(task_name, str) or task_name in processed:
            continue
        processed.add(task_name)

        # Handle task groups (tasks with wildcards or variations)
        base_task = task_name.split("_")[0] if "_" in task_name else task_name
        
        # Try exact match first, then base task
        dataset_name = TASK_TO_DATASET_MAP.get(task_name) or TASK_TO_DATASET_MAP.get(base_task)
        
        if not dataset_name:
            logging.info(f"No known dataset mapping for task '{task_name}' - skipping pre-download")
            continue

        try:
            logging.info(f"Pre-downloading dataset '{dataset_name}' for task '{task_name}'...")

            # Special handling for tasks that need specific subsets
            if task_name in ["arc_easy", "arc_challenge"]:
                subset = "ARC-Easy" if "easy" in task_name else "ARC-Challenge"
                load_dataset(dataset_name, subset, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
            elif task_name in ["rte", "wsc"] and dataset_name == "nyu-mll/glue":
                subset = task_name.upper()
                load_dataset(dataset_name, subset, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
            elif task_name in ["cb", "wic"] and dataset_name == "nyu-mll/super_glue":
                subset = task_name.upper() if task_name == "cb" else "WiC"
                load_dataset(dataset_name, subset, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
            elif task_name == "mmlu":
                # MMLU requires 'all' config to get the full dataset
                load_dataset(dataset_name, "all", download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
            elif task_name == "gsm8k":
                # GSM8K uses 'main' config
                load_dataset(dataset_name, "main", download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
            elif task_name == "winogrande":
                # WinoGrande typically uses the debiased version
                load_dataset(dataset_name, "winogrande_debiased", download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
            else:
                # For most datasets, load without subset
                load_dataset(dataset_name, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

            logging.debug(f"Successfully pre-downloaded dataset for task '{task_name}'")
            
        except Exception as e:
            logging.warning(
                "Failed to pre-download dataset for task '%s' (dataset: %s). The evaluation job might fail on the compute node.\n%s",
                task_name,
                dataset_name,
                e,
            )


def schedule_evals(
    models: str | None = None,
    tasks: str | None = None,
    n_shot: int | list[int] | None = None,
    eval_csv_path: str | None = None,
    *,
    max_array_len: int = 32,
    debug: bool = False,
    download_only: bool = False,
) -> None:
    """
    Schedule evaluation jobs for a given set of models, tasks, and number of shots.

    Args:
        models: A string of comma-separated model paths or Hugging Face model identifiers.
            Warning: does not allow passing model args such as `EleutherAI/pythia-160m,revision=step100000`
            since we split on commas. If you need to pass model args, use the `eval_csv_path` option.
            For local paths, the path must either be a directory that contains a `.safetensors` file or a directory that contains a lot of
            intermediate checkpoints from training of the structure: `model_name/hf/iter_XXXXX`. The function will expand the path to
            include all the intermediate checkpoints and schedule jobs for each checkpoint.
        tasks: A string of comma-separated task paths.
        n_shot: An integer or list of integers specifying the number of shots for each task.
        eval_csv_path: A path to a CSV file containing evaluation data.
            Warning: exclusive argument. Cannot specify `models`, `tasks`, or `n_shot` when `eval_csv_path` is provided.
        max_array_len: The maximum number of jobs to schedule to run concurrently.
            Warning: this is not the number of jobs in the array job. This is determined by the environment variable `QUEUE_LIMIT`.
        download_only: If True, only download the datasets and models and exit.
    """
    _setup_logging(debug)

    # Load cluster-specific environment variables (paths, etc.)
    _load_cluster_env()

    # ------------------------------------------------------------------
    # Ensure that the shared Singularity image derived from the Docker
    # reference is present (or freshly rebuilt if missing). All users on
    # a cluster share the same image under EVAL_SIF_PATH (configured in
    # clusters.json). This avoids the brittle shared-venv approach.
    # ------------------------------------------------------------------
    image_name = os.environ.get("EVAL_CONTAINER_IMAGE")
    if image_name is None:
        raise ValueError(
            "EVAL_CONTAINER_IMAGE is not set. Please set it in clusters.json."
        )

    ensure_singularity_image(image_name)

    if eval_csv_path:
        if models or tasks or n_shot:
            raise ValueError(
                "Cannot specify `models`, `tasks`, or `n_shot` when `eval_csv_path` is provided."
            )
        df = pd.read_csv(eval_csv_path)
        required_cols = {"model_path", "task_path", "n_shot"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"CSV file must contain the columns: {', '.join(required_cols)}"
            )

        unique_models = df["model_path"].unique()
        model_path_map = _process_model_paths(unique_models)

        # Create a new DataFrame with the expanded model paths
        expanded_rows = []
        for _, row in df.iterrows():
            original_model_path = row["model_path"]
            if original_model_path in model_path_map:
                for expanded_path in model_path_map[original_model_path]:
                    new_row = row.copy()
                    new_row["model_path"] = expanded_path
                    expanded_rows.append(new_row)
        df = pd.DataFrame(expanded_rows)

    elif models and tasks and n_shot is not None:
        model_path_map = _process_model_paths(models.split(","))
        model_paths = [p for paths in model_path_map.values() for p in paths]
        tasks_list = tasks.split(",")

        # cross product of model_paths and tasks into a dataframe
        df = pd.DataFrame(
            product(
                model_paths,
                tasks_list,
                n_shot if isinstance(n_shot, list) else [n_shot],
            ),
            columns=["model_path", "task_path", "n_shot"],
        )
    else:
        raise ValueError(
            "Either `eval_csv_path` must be provided, or all of `models`, `tasks`, and `n_shot`."
        )

    if df.empty:
        logging.warning("No evaluation jobs to schedule.")
        return None

    # Ensure that all datasets required by the tasks are cached locally to avoid
    # network access on compute nodes.
    _pre_download_task_datasets(df["task_path"].unique())

    if download_only:
        return None

    queue_limit = int(os.environ.get("QUEUE_LIMIT", 250))
    remaining_queue_capacity = queue_limit - _parse_user_queue_load()

    if remaining_queue_capacity <= 0:
        logging.warning("No remaining queue capacity. Not scheduling any jobs.")
        return None

    logging.debug(
        f"Remaining capacity in the queue: {remaining_queue_capacity}. Number of "
        f"evals to schedule: {len(df)}."
    )

    evals_dir = (
        Path(os.environ["EVAL_OUTPUT_DIR"])
        / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    evals_dir.mkdir(parents=True, exist_ok=True)

    slurm_logs_dir = evals_dir / "slurm_logs"
    slurm_logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = evals_dir / "jobs.csv"

    df.to_csv(csv_path, index=False)

    logging.debug(f"Saved evaluation dataframe to temporary CSV: {csv_path}")

    with open(Path(__file__).parent / "template.sbatch", "r") as f:
        sbatch_template = f.read()

    # replace the placeholders in the template with the actual values
    # First, replace python-style placeholders
    sbatch_script = sbatch_template.format(
        csv_path=csv_path,
        max_array_len=max_array_len,
        array_limit=len(df) - 1,
        num_jobs=len(df),
        log_dir=evals_dir / "slurm_logs",
        evals_dir=str(evals_dir / "results"),
    )

    # substitute any $ENV_VAR occurrences (e.g., $TIME_LIMIT) since env vars are not
    # expanded in the #SBATCH directives
    sbatch_script = Template(sbatch_script).safe_substitute(os.environ)

    # Save the sbatch script to the evals directory
    sbatch_script_path = evals_dir / "submit_evals.sbatch"
    with open(sbatch_script_path, "w") as f:
        f.write(sbatch_script)

    logging.debug(f"Saved sbatch script to {sbatch_script_path}")

    # Submit the job script to slurm by piping the script content to sbatch
    try:
        result = subprocess.run(
            ["sbatch"],
            input=sbatch_script,
            text=True,
            check=True,
            capture_output=True,
            env=os.environ,
        )
        logging.info("Job submitted successfully.")
        logging.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to submit job: {e}")
        logging.error(f"sbatch stderr: {e.stderr}")
    except FileNotFoundError:
        logging.error(
            "sbatch command not found. Please make sure you are on a system with SLURM installed."
        )


def main():
    auto_cli({"schedule-eval": schedule_evals}, as_positional=False)
