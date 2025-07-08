from itertools import product
from pathlib import Path
import logging
import subprocess

import huggingface_hub
import pandas as pd


def _parse_user_queue_load() -> int:
    return int(
        subprocess.run(
            [
                "squeue",
                "-u",
                "$USER",
                "-h",
                "-t",
                "pending,running",
                "-r",
                "|",
                "wc",
                "-l",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def schedule_evals(
    models: list[str],
    tasks: list[str],
    n_shot: int | list[int],
    max_array_len: int,
) -> pd.DataFrame | None:
    model_paths = []
    # parse models into either local paths or huggingface models
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
                model_path = huggingface_hub.hf_hub_download(**model_kwargs)
            else:
                model_path = huggingface_hub.hf_hub_download(model)

            model_paths.append(model_path)

    # cross product of model_paths and tasks into a dataframe
    df = pd.DataFrame(
        product(model_paths, tasks, n_shot if isinstance(n_shot, list) else [n_shot]),
        columns=["model_path", "task_path", "n_shot"],
    )

    # check if the user queue load is too high
    user_queue_load = _parse_user_queue_load()
    if user_queue_load > max_array_len:
        raise RuntimeError(
            f"The user queue load is too high. Detected {user_queue_load} "
            f"running or pending jobs in the queue, max allowed is {max_array_len}. "
            f"Please wait for some jobs to finish or reduce the number of jobs you are running."
        )

    remaining_capacity = max_array_len - user_queue_load
    logging.debug(
        f"Remaining capacity in the array: {remaining_capacity}. Number of "
        f"evals to schedule: {len(df)} with {len(df) / remaining_capacity} evals per job."
    )

    return df


def main():
    print("Hello from oellm-eval!")


if __name__ == "__main__":
    main()
