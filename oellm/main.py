from itertools import product
from pathlib import Path
import logging
import subprocess
import os
import tempfile
from typing import Iterable

import pandas as pd
from jsonargparse import CLI
from transformers import AutoModelForCausalLM


def _parse_user_queue_load() -> int:
    command = "squeue -u $USER -h -t pending,running -r | wc -l"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    import pdb

    pdb.set_trace()
    if result.stdout:
        try:
            return int(result.stdout.strip())
        except ValueError:
            logging.warning(f"Could not parse squeue output: {result.stdout}")
            return 0

    if result.stderr:
        logging.warning(f"squeue command produced an error: {result.stderr.strip()}")

    return 0


def _process_model_paths(
    models: Iterable[str], debug: bool = False
) -> dict[str, list[Path | str]]:
    """
    Processes model strings into a dict of model paths.

    Each model string can be a local path or a huggingface model identifier.
    This function expands directory paths that contain multiple checkpoints.
    """
    processed_model_paths = {}
    for model in models:
        model_paths = []
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
            logging.info(f"Model {model} not found, assuming it is a huggingface model")
            logging.debug(
                f"Downloading model {model} on the login node since the compute nodes may not have access to the internet"
            )

            if not debug:
                if "," in model:
                    model_kwargs = {
                        k: v
                        for k, v in [
                            kv.split("=") for kv in model.split(",") if "=" in kv
                        ]
                    }
                    model_kwargs["pretrained_model_name_or_path"] = model.split(",")[0]
                    try:
                        AutoModelForCausalLM.from_pretrained(**model_kwargs)
                        model_paths.append(model)
                    except Exception as e:
                        logging.debug(
                            f"Failed to download model {model} from huggingface. Continuing..."
                        )
                        logging.debug(e)
                else:
                    AutoModelForCausalLM.from_pretrained(model)
                    model_paths.append(model)
            else:
                model_paths.append(model)

        if not model_paths:
            logging.warning(
                f"Could not find any valid model for '{model}'. It will be skipped."
            )
        processed_model_paths[model] = model_paths
    return processed_model_paths


def schedule_evals(
    models: str | None = None,
    tasks: str | None = None,
    n_shot: int | list[int] | None = None,
    eval_csv_path: str | None = None,
    *,
    max_array_len: int,
    debug: bool = False,
) -> pd.DataFrame | None:
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
        model_path_map = _process_model_paths(unique_models, debug)

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
        model_path_map = _process_model_paths(models.split(";"), debug)
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

    if debug:
        remaining_queue_capacity = 10
    else:
        queue_limit = int(os.environ.get("QUEUE_LIMIT", 250))
        remaining_queue_capacity = queue_limit - _parse_user_queue_load()

    if remaining_queue_capacity <= 0:
        logging.warning("No remaining queue capacity. Not scheduling any jobs.")
        return None

    logging.debug(
        f"Remaining capacity in the queue: {remaining_queue_capacity}. Number of "
        f"evals to schedule: {len(df)}."
    )

    # Save df to temporary CSV file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, delete_on_close=False
    ) as temp_file:
        df.to_csv(temp_file.name, index=False)
        temp_csv_path = temp_file.name

    logging.debug(f"Saved evaluation dataframe to temporary CSV: {temp_csv_path}")

    with open(Path(__file__).parent / "template.sbatch", "r") as f:
        sbatch_template = f.read()

    # replace the placeholders in the template with the actual values
    sbatch_script = sbatch_template.format(
        csv_path=temp_csv_path,
        max_array_len=max_array_len,
        array_limit=len(df) - 1,
        num_jobs=len(df),
    )

    logging.debug("--- Generated sbatch script ---")
    logging.debug(sbatch_script)
    logging.debug("-------------------------------")

    # Submit the job script to slurm by piping the script content to sbatch
    try:
        result = subprocess.run(
            ["sbatch"],
            input=sbatch_script,
            text=True,
            check=True,
            capture_output=True,
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

    return df


def main():
    logging.basicConfig(level=logging.DEBUG)
    """The main entrypoint for the CLI."""
    CLI({"schedule": schedule_evals}, as_positional=False)
