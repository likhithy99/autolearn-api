# Hugging Face Space (free CPU tier: 2 vCPU, 16 GB RAM) running the public AutoLearn demo.
# Builds straight from the GitHub repo; set the AUTOLEARN_ADMIN_KEY secret in the Space settings.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HOME=/home/user AUTOLEARN_DATA_DIR=/home/user/data AUTOLEARN_OPEN_MODELS_DIR=/home/user/models/open \
    AUTOLEARN_PUBLIC_DEMO=1 OMP_NUM_THREADS=2 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ARG REPO=https://github.com/likhithy99/autolearn-api.git
ARG REF=main
RUN git clone --depth 1 --branch $REF $REPO . && \
    pip install --user torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --user -r requirements.txt
ENV PATH=/home/user/.local/bin:$PATH
# Models are baked into the image so the Space starts without downloading.
RUN python -m autolearn.llm.open_models --root $AUTOLEARN_OPEN_MODELS_DIR add Qwen/Qwen2.5-1.5B-Instruct \
        --name autolearn-pro --description "1.5B instruction model (Qwen2.5, Apache-2.0) on AutoLearn's engine" && \
    python -m autolearn.llm.open_models --root $AUTOLEARN_OPEN_MODELS_DIR add Qwen/Qwen2.5-0.5B-Instruct \
        --name autolearn-lite --description "Fast 0.5B model (Qwen2.5, Apache-2.0) for simple tasks" && \
    python -m autolearn.llm.embeddings --root $AUTOLEARN_OPEN_MODELS_DIR add BAAI/bge-small-en-v1.5 \
        --name autolearn-embed
EXPOSE 7860
CMD ["sh", "-c", "AUTOLEARN_PUBLIC_URL=https://$SPACE_HOST exec uvicorn autolearn.main:app --host 0.0.0.0 --port 7860 --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
