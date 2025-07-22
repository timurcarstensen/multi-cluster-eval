FROM rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1

# 1. Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && echo 'export PATH=$HOME/.local/bin:$PATH' >> /etc/profile

# Make uv visible for every subsequent RUN/CMD
ENV PATH=/root/.local/bin:$PATH

# 2. Install lm-eval and deps from PyPI
RUN uv pip install --system --break-system-packages lm-eval \
    transformers "datasets<4.0.0" wandb sentencepiece accelerate

LABEL org.opencontainers.image.source=https://github.com/OpenEuroLLM/multi-cluster-eval

WORKDIR /workspace
CMD ["bash"] 