from itertools import product
from pathlib import Path
import logging
import subprocess
import os

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
            if "," in model:
                model_kwargs = {
                    k: v
                    for k, v in [kv.split("=") for kv in model.split(",") if "=" in kv]
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

    tasks = tasks.split(";")

    # cross product of model_paths and tasks into a dataframe
    df = pd.DataFrame(
        product(model_paths, tasks, n_shot if isinstance(n_shot, list) else [n_shot]),
        columns=["model_path", "task_path", "n_shot"],
    )

    # check if the user queue load is too high
    # user_queue_load = _parse_user_queue_load()
    # if user_queue_load > os.getenv("QUEUE_LIMIT"):
    #     raise RuntimeError(
    #         f"The user queue load is too high. Detected {user_queue_load} "
    #         f"running or pending jobs in the queue, max allowed is {max_array_len}. "
    #         f"Please wait for some jobs to finish or reduce the number of jobs you are running."
    #     )

    remaining_queue_capacity = os.getenv("QUEUE_LIMIT") - _parse_user_queue_load()
    logging.debug(
        f"Remaining capacity in the array: {remaining_queue_capacity}. Number of "
        f"evals to schedule: {len(df)} with {len(df) / remaining_queue_capacity} evals per job."
    )

    # distribute indices of df into chunks of size remaining_capacity (e.g., each job gets total_evals / remaining_capacity evals)

    # Calculate how many evals each job should handle
    evals_per_job = max(1, len(df) // remaining_queue_capacity)

    # Create chunks of indices for each job
    chunks = []
    for i in range(0, len(df), evals_per_job):
        chunk = df.iloc[i : i + evals_per_job]
        chunks.append(chunk)

    # If we have more chunks than remaining capacity, adjust the last chunk
    if len(chunks) > remaining_queue_capacity:
        # Merge excess chunks into the last allowed chunk
        excess_chunks = chunks[remaining_queue_capacity:]
        chunks = chunks[:remaining_queue_capacity]
        chunks[-1] = pd.concat([chunks[-1]] + excess_chunks, ignore_index=True)

    # Store the chunks in df for later use
    df["chunk_id"] = -1  # Initialize chunk_id column
    for chunk_idx, chunk in enumerate(chunks):
        df.loc[chunk.index, "chunk_id"] = chunk_idx

    import pdb

    pdb.set_trace()
    logging.info(f"Distributed {len(df)} evaluations into {len(chunks)} chunks")

    return df


def main():
    """The main entrypoint for the CLI."""
    CLI(schedule_evals, as_positional=False)
