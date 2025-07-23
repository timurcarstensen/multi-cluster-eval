import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import shutil

# Hugging Face Hub utilities (already dependency in pyproject.toml)
from huggingface_hub import hf_hub_download, HfHubDownloadError  # type: ignore
import os


def _inspect_sif_digest(sif_path: Path) -> Optional[str]:
    """Return the originating container digest stored in the SIF, if available."""
    try:
        result = subprocess.run(
            ["singularity", "inspect", "--json", str(sif_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        # Data schema: {"data": {"attributes": {"labels": {...}}}}
        labels = data.get("data", {}).get("attributes", {}).get("labels", {}) or {}
        # OCI annotations may store the digest under this key (depends on apptainer version)
        for k in (
            "org.containers.image.digest",
            "org.container.image.digest",
            "ContainerImageDigest",
        ):
            if k in labels:
                return labels[k]
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        logging.debug(f"Could not inspect SIF digest: {exc}")
    return None


# NOTE: Proper registry API handling (auth, different registries) is non-trivial.
# For now we always rebuild if the local file does not exist; otherwise we assume
# it is current unless the caller passes force=True.


def ensure_singularity_image(
    docker_image: str,  # e.g. ghcr.io/oellm/eval_env:latest
    sif_path: Path,
    debug: bool = False,
    *,
    force: bool = False,
) -> None:
    """Ensure that `sif_path` exists, downloading a pre-built SIF from the 🤗 Hub if
    possible and falling back to building it from *docker_image* otherwise.

    The function behaves as follows:
    1. If *sif_path* already exists (and *force* is False) → return immediately.
    2. Try to download the file *sif_path.name* from a dataset repository on the
       Hugging Face Hub (defaults to the repo specified by the *HF_SIF_REPO*
       environment variable or ``timurcarstensen/testing``).  If the download
       succeeds, copy/rename the cached file to *sif_path* and return.
    3. If the download fails (e.g. repo/file not found or network issues), build
       the image locally from *docker_image* using ``singularity build``.

    This keeps the original build-from-Docker logic as a graceful fallback while
    enabling fast, bandwidth-efficient retrieval of large SIFs (>10 GB) that are
    inconvenient to store on container registries.
    """

    sif_path = sif_path.expanduser().resolve()

    if sif_path.exists() and not force:
        logging.debug(f"Re-using existing Singularity image: {sif_path}")
        return

    # ------------------------------------------------------------------
    # Attempt to obtain the SIF from a Hugging Face Hub dataset first.
    # ------------------------------------------------------------------
    hf_repo = os.environ.get("HF_SIF_REPO", "timurcarstensen/testing")
    hf_token = os.environ.get("HF_TOKEN")  # optional; can rely on cached creds
    sif_filename = sif_path.name

    try:
        download_path = hf_hub_download(
            repo_id=hf_repo,
            filename=sif_filename,
            repo_type="dataset",
            token=hf_token,
        )

        # hf_hub_download returns a cached path; copy (atomic) into place
        logging.info("Downloaded %s from 🤗 Hub dataset %s", sif_filename, hf_repo)

        # Use a temporary path in the destination dir then atomic replace
        tmp_path = sif_path.with_suffix(".tmp")
        shutil.copy2(download_path, tmp_path)
        tmp_path.replace(sif_path)

        logging.info("Singularity image ready at %s", sif_path)
        return

    except HfHubDownloadError as exc:
        logging.warning(
            "Could not download %s from 🤗 Hub (%s); falling back to local build.",
            sif_filename,
            exc,
        )
    except Exception as exc:
        # Any unexpected error should still trigger fallback build to ensure we
        # don't block execution.
        logging.warning(
            "Unexpected error while downloading SIF from 🤗 Hub: %s. Falling back to build.",
            exc,
        )

    logging.info(
        (
            "Building Singularity image at %s from Docker image %s. "
            "This may take a few minutes…"
        ),
        sif_path,
        docker_image,
    )

    sif_path.parent.mkdir(parents=True, exist_ok=True)

    # Build to a temporary file first, then atomically move into place so that
    # concurrent invocations do not race on the final path.
    with tempfile.NamedTemporaryFile(
        dir=sif_path.parent, suffix=".sif", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    build_cmd = [
        "singularity",
        "build",
        "--force",  # skip interactive overwrite prompts
        str(tmp_path),
        f"docker://{docker_image}",
    ]

    if debug:
        logging.debug("Running: %s", " ".join(build_cmd))

    try:
        subprocess.run(build_cmd, check=True)
        # Atomic replacement
        tmp_path.replace(sif_path)
    except subprocess.CalledProcessError as exc:
        logging.error("Failed to build Singularity image: %s", exc)
        # Clean up temporary file on failure
        tmp_path.unlink(missing_ok=True)
        raise

    logging.info("Singularity image built at %s", sif_path)
