"""Lock condiviso per gli avvii MarketSentinel."""
import fcntl
from contextlib import contextmanager
from pathlib import Path

LOCK = (
    Path(__file__).resolve().parent
    / "data"
    / "marketsentinel_intraday"
    / ".shared_start.lock"
)

@contextmanager
def shared_start_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
