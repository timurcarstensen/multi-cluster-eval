from itertools import product
from pathlib import Path
import logging
import subprocess
import os
import tempfile

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


def schedule_evals(
    models: str,
    tasks: str,
    n_shot: int,
    max_array_len: int,
    debug: bool = False,
) -> pd.DataFrame | None:
    model_paths = []
    # parse models into either local paths or huggingface models
    for model in models.split(";"):
        if Path(model).exists() and Path(model).is_dir():
            # could either be the direct path to a local model checkpoint dir or a directory that contains a lot of
            # intermediate checkpoints from training of the structure: `model_name/hf/iter_1`, `model_name/hf/iter_2` ...
            # or `model_name/iter_1`, `model_name/iter_2` ...
            # The base case is that `model_name` is a directory that contains the model in a HF checkpoint format

            # Basecase: check if the directory contains a `.safetensors` file
            if any(Path(model).glob("*.safetensors")):
                model_paths.append(Path(model))

            # check if dir contains subdirs that themselves contain a `.safetensors` file
            model_path = (
                Path(model) / "hf" if "hf" not in Path(model).name else Path(model)
            )
            for subdir in model_path.glob("*"):
                if subdir.is_dir() and any(subdir.glob("*.safetensors")):
                    model_paths.append(subdir)

        else:
            logging.info(f"Model {model} not found, assuming it is a huggingface model")
            logging.debug(
                f"Downloading model {model} on the login node since the compute nodes may not have access to the internet"
            )

            # TODO: remove later
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
                        model_paths.append(model_path)
                    except Exception as e:
                        logging.debug(
                            f"Failed to download model {model} from huggingface. Continuing..."
                        )
                        logging.debug(e)
                else:
                    AutoModelForCausalLM.from_pretrained(model)
                    model_paths.append(model_path)
            else:
                model_paths.append(model)

    tasks = tasks.split(",")

    # cross product of model_paths and tasks into a dataframe
    df = pd.DataFrame(
        product(model_paths, tasks, n_shot if isinstance(n_shot, list) else [n_shot]),
        columns=["model_path", "task_path", "n_shot"],
    )

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
        array_limit=remaining_queue_capacity - 1,
        num_jobs=remaining_queue_capacity,
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
    CLI({"schedule-eval": schedule_evals}, as_positional=False)
