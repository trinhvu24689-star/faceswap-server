import time
import threading
from fastapi import Request, HTTPException

# =======================
# RATE LIMIT CONFIG
# =======================

REQUEST_LIMIT = 10      # số request tối đa
TIME_WINDOW = 60        # trong 60 giây
BLOCK_TIME = 120        # khóa IP 120 giây nếu vượt limit

# =======================
# MEMORY STORE (SAFE)
# =======================

ip_requests = {}
ip_blocked = {}
lock = threading.Lock()

# =======================
# MAIN CHECK FUNCTION
# =======================

def check_rate_limit(request: Request):
    ip = request.client.host
    now = time.time()

    with lock:
        # IP đang bị block
        if ip in ip_blocked:
            if now < ip_blocked[ip]:
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Try again later."
                )
            else:
                del ip_blocked[ip]

        # Lần đầu truy cập
        if ip not in ip_requests:
            ip_requests[ip] = []

        # Xóa request cũ
        ip_requests[ip] = [
            t for t in ip_requests[ip] if now - t < TIME_WINDOW
        ]

        # Nếu vượt giới hạn
        if len(ip_requests[ip]) >= REQUEST_LIMIT:
            ip_blocked[ip] = now + BLOCK_TIME
            ip_requests[ip] = []
            raise HTTPException(
                status_code=429,
                detail="Too many requests. IP temporarily blocked."
            )

        # Lưu request mới
        ip_requests[ip].append(now)

    return True