# syntax=docker/dockerfile:1

FROM python:3.11.9-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /usr/local/bin/uv
WORKDIR /app

# The API image installs the base runtime plus deploy and serve only, so torch
# never enters it; the ONNX model and policy are mounted at run time.
FROM base AS api
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra deploy --extra serve
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uv", "run", "--no-dev", "--extra", "deploy", "--extra", "serve", \
     "uvicorn", "--factory", "castinspect.service.main:app_from_env", \
     "--host", "0.0.0.0", "--port", "8000"]

FROM base AS ui
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra deploy --extra ui
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"
CMD ["uv", "run", "--no-dev", "--extra", "deploy", "--extra", "ui", \
     "streamlit", "run", "src/castinspect/ui/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
