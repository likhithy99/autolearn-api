# CPU image: API + AutoML + LLM inference (fine for small LLMs; use Dockerfile.gpu for large ones)
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    AUTOLEARN_DATA_DIR=/data AUTOLEARN_LLM_DIR=/models/llm
WORKDIR /app
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY autolearn ./autolearn
RUN useradd --create-home --uid 10001 app && mkdir -p /data /models/llm && chown -R app /data /models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
# One worker per container: models + the learner live in-process. Scale out with more containers.
CMD ["uvicorn", "autolearn.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
