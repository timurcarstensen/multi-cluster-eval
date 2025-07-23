import logging
import shutil

# Hugging Face Hub utilities (already dependency in pyproject.toml)
from huggingface_hub import hf_hub_download  # type: ignore
import os


def ensure_singularity_image(
    image_name: str,
) -> None:
    """Ensure that `sif_path` exists, downloading a pre-built SIF from the 🤗 Hub.

    The function behaves as follows:
    1. If *sif_path* already exists (and *force* is False) → return immediately.
    2. Download the file *sif_path.name* from a dataset repository on the
       Hugging Face Hub (defaults to the repo specified by the *HF_SIF_REPO*
       environment variable or ``timurcarstensen/testing``).  If the download
       succeeds, copy the cached file to *sif_path* and return.
    """

    # ------------------------------------------------------------------
    # Download the SIF from a Hugging Face Hub dataset.
    # ------------------------------------------------------------------
    hf_repo = os.environ.get("HF_SIF_REPO", "timurcarstensen/testing")
    sif_filename = image_name

    download_path = hf_hub_download(
        repo_id=hf_repo,
        filename=sif_filename,
        repo_type="dataset",
        local_dir=os.getenv("EVAL_BASE_DIR"),
    )

    # hf_hub_download returns a cached path; copy into place
    logging.info("Downloaded %s from 🤗 Hub dataset %s", sif_filename, hf_repo)
    
    shutil.copy2(download_path, os.getenv("EVAL_SIF_PATH"))

    logging.info("Singularity image ready at %s", os.getenv("EVAL_SIF_PATH"))
