import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def _inspect_sif_digest(sif_path: Path) -> Optional[str]:
    """Return the originating container digest stored in the SIF, if available."""
    try:
        result = subprocess.run(
            ["apptainer", "inspect", "--json", str(sif_path)],
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
    """Ensure that `sif_path` exists and is (roughly) in sync with `docker_image`.

    This downloads/builds the SIF in-place using `apptainer build` when necessary.
    It is safe to call repeatedly; construction is skipped when the file exists
    unless *force* is True.
    """

    sif_path = sif_path.expanduser().resolve()
    if sif_path.exists() and not force:
        logging.debug(f"Re-using existing Singularity image: {sif_path}")
        return

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
        "apptainer",
        "build",
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
