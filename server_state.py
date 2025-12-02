import datetime as dt
import threading
import time


class ServerState:
    """
    Trạng thái server dùng cho hệ thống:
    - loading / khởi động (wake)
    - theo dõi activity để sleep / keep-alive
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status: str = "sleeping"  # sleeping | waking | awake
        self.wake_progress: int = 0    # 0–100
        self.last_activity: dt.datetime = dt.datetime.utcnow()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "wake_progress": self.wake_progress,
                "last_activity": self.last_activity.isoformat() + "Z",
            }

    def set_status(self, status: str, progress: int | None = None) -> None:
        with self._lock:
            self.status = status
            if progress is not None:
                self.wake_progress = max(0, min(100, progress))

    def touch(self) -> None:
        with self._lock:
            self.last_activity = dt.datetime.utcnow()
            # khi có activity thì coi như đã thức
            if self.status == "sleeping":
                self.status = "awake"
            self.wake_progress = 100


SERVER_STATE = ServerState()


def start_wake_progress() -> None:
    """
    Fake progress bar cho UI phía frontend:
    tăng dần wake_progress từ 0 -> 100,
    status: waking -> awake.
    """

    def worker():
        SERVER_STATE.set_status("waking", 0)
        for p in range(0, 101, 5):
            SERVER_STATE.set_status("waking", p)
            time.sleep(0.3)  # tùy, chỉ là hiệu ứng
        SERVER_STATE.set_status("awake", 100)
        SERVER_STATE.touch()

    t = threading.Thread(target=worker, daemon=True)
    t.start()


def mark_activity() -> None:
    """
    Được gọi mỗi khi có request quan trọng (swap, check credit, v.v...)
    để giữ server ở trạng thái awake.
    """
    SERVER_STATE.touch()
