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

# ---- FIX BẮT BUỘC CHO INSIGHTFACE / OPENCV (KHÔNG ĐỤNG CHỮ CỦA BÉ) ----
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 libglib2.0-dev libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv .venv/

# COPY toàn bộ mã nguồn CHÍNH XÁC
COPY main.py .
COPY server_state.py .
COPY db_engine.py .
COPY rate_limit.py .
COPY task_queue.py .
COPY Procfile .

# ---- FIX NHỎ: đúng tên file (services_video_processor.py) ----
COPY servicesvideo_processor.py ./servicesvideo_processor.py

COPY models_extended.py .
COPY models.py .
COPY database.py .
COPY routers/ ./routers/


# ---- COPY saved/ nếu tồn tại ----
RUN mkdir -p saved
COPY saved/ ./saved/

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]