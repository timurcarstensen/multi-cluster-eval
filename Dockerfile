FROM nvcr.io/nvidia/pytorch:25.06-py3

# uv + deps
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install lm-eval-harness directly from PyPI to avoid git cloning issues
RUN uv pip install lm-eval \
    transformers "datasets<4.0.0" wandb sentencepiece accelerate

WORKDIR /workspace

CMD ["bash"] 