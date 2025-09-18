# Stage 1: Builder stage for installing dependencies
FROM python:3.10.7-slim AS builder

# Install system dependencies including ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir \
 && pip install nemo_toolkit["asr"] \
 && pip install 'uvicorn[standard]' --no-cache-dir \
 && pip install --no-cache-dir -r requirements.txt \
 && pip cache purge

# Stage 2: Runtime stage
FROM python:3.10.7-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./parakeet_service ./parakeet_service
COPY .env.example .env
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "parakeet_service.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
