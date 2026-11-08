"""Supervisore WEEKLY MarketSentinel."""
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "data" / "marketsentinel_intraday"
STATE = DIR / ".weekly_auto_state.json"
LOCK = DIR / ".weekly_controller.lock"
LOG = DIR / "weekly_auto_refresh.log"

CHECK_INTERVAL = timedelta(minutes=15)

def utcnow():
    return datetime.now(timezone.utc)

def read_state():
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}

def write_state(data):
    DIR.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
    os.replace(temp, STATE)

def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False

def weekly_status():
    state = read_state()
    if state.get("status") == "running":
        if not pid_alive(state.get("pid")):
            state["status"] = "error"
            state["error"] = "Processo WEEKLY interrotto"
    return state

def start_weekly(force=False):
    DIR.mkdir(parents=True, exist_ok=True)

    from ms_shared_process_lock import shared_start_lock
    with shared_start_lock():
        from ms_auto_refresh import daily_status
        if daily_status().get("status") == "running":
            return False, "DAILY in esecuzione"
        with open(LOCK, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

            state = weekly_status()

            if state.get("status") == "running":
                return False, "WEEKLY già in aggiornamento"

            last_check = state.get("last_check_at")

            if not force and last_check:
                try:
                    last = datetime.fromisoformat(last_check)
                    if utcnow() - last < CHECK_INTERVAL:
                        return False, "WEEKLY già verificato recentemente"
                except (ValueError, TypeError):
                    pass

            with open(LOG, "a") as logfile:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "ms_weekly_supervisor.py"),
                        "--worker",
                    ],
                    cwd=str(ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

            state.update({
                "status": "running",
                "pid": process.pid,
                "started_at": utcnow().isoformat(),
                "last_check_at": utcnow().isoformat(),
            })
            write_state(state)

            return True, f"WEEKLY avviato: PID {process.pid}"

def worker():
    import traceback
    os.environ.setdefault("MS_PHASE_V4_FAST", "1")
    from ms_weekly_worker import run_weekly

    try:
        run_weekly()
        state = read_state()
        state.update({
            "status": "success",
            "finished_at": utcnow().isoformat(),
            "last_success_at": utcnow().isoformat(),
            "error": None,
        })
        write_state(state)
    except Exception:
        error = traceback.format_exc()
        state = read_state()
        state.update({
            "status": "error",
            "finished_at": utcnow().isoformat(),
            "error": error[-4000:],
        })
        write_state(state)
        print(error, flush=True)
        raise

if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        print(json.dumps(weekly_status(), indent=2))
