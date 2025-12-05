from database import SessionLocal, init_swap_db

# Fallback an toàn nếu project không có SwapHistoryModel
try:
    from database import SwapHistoryModel
except ImportError:
    SwapHistoryModel = None