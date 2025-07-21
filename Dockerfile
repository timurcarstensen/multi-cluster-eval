FROM nvcr.io/nvidia/pytorch:25.06-py3

# 1. Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && echo 'export PATH=$HOME/.local/bin:$PATH' >> /etc/profile

# Make uv visible for every subsequent RUN/CMD
ENV PATH=/root/.local/bin:$PATH

# 2. Install lm-eval and deps from PyPI
RUN uv pip install lm-eval \
    transformers "datasets<4.0.0" wandb sentencepiece accelerate

WORKDIR /workspace
CMD ["bash"] 