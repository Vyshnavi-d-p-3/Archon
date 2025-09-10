FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps (layer cached)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Non-root user
RUN useradd -m agent && chown -R agent:agent /app
USER agent

# Default: run demo (no API keys needed)
ENTRYPOINT ["python", "main.py"]
CMD ["demo"]
