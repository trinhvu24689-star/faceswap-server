FROM python:3.10.11 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN python -m venv .venv

COPY requirements.txt .
RUN .venv/bin/pip install --no-cache-dir -r requirements.txt


# ===========================
#       FINAL IMAGE
# ===========================

FROM python:3.10.11-slim
WORKDIR /app

# ---- FIX BẮT BUỘC CHO INSIGHTFACE / OPENCV ----
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 libglib2.0-dev libopencv-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv .venv/
ENV PATH="/app/.venv/bin:$PATH"

# COPY toàn bộ mã nguồn CHÍNH XÁC
COPY main.py .
COPY server_state.py .
COPY Procfile .
COPY models.py .
COPY database.py .
COPY db_engine.py .
COPY routers/ ./routers/

# ---- COPY saved/ nếu tồn tại ----
RUN mkdir -p saved
COPY saved/ ./saved/

EXPOSE 8080

# ✅ CMD CHUẨN FLY - KHÔNG CRASH - KHÔNG 503
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]