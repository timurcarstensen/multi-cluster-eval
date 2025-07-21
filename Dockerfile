FROM nvcr.io/nvidia/pytorch:25.06-py3

# uv + deps
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace
RUN git clone --depth 1 git@github.com:EleutherAI/lm-eval-harness.git \
    && uv pip install -e ./lm-eval-harness \
       transformers "datasets<4.0.0" wandb sentencepiece accelerate

CMD ["bash"] 