"""Per-IP rate limit using a salted, daily-rotating hash so raw IPs are never stored."""
import hashlib
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone


class RateLimiter:
    # shortcut: in-memory, single process, resets on restart; move to Redis/SQLite if we run >1 worker
    def __init__(self, limit: int = 10, window_s: int = 3600, salt: str | None = None):
        self.limit, self.window = limit, window_s
        self.salt = salt or os.environ.get("SIXFINGERS_IP_SALT") or os.urandom(16).hex()
        self.hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def ip_hash(self, ip: str) -> str:
        day = datetime.now(timezone.utc).date().isoformat()
        return hashlib.sha256(f"{self.salt}|{day}|{ip}".encode()).hexdigest()[:16]

    def allow(self, key: str) -> int:
        """Return 0 if allowed (and count it), else seconds until a slot frees up."""
        now = time.monotonic()
        with self._lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return int(self.window - (now - q[0])) + 1
            q.append(now)
            if len(self.hits) > 50_000:  # drop stale keys so memory stays bounded
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] > self.window]:
                    del self.hits[k]
            return 0
