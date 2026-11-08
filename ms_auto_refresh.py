"""Coordinamento sicuro aggiornamento DAILY MarketSentinel."""
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "data" / "marketsentinel_intraday"
STATE_FILE = STATE_DIR / ".auto_refresh_state.json"
LOCK_FILE = STATE_DIR / ".auto_refresh_controller.lock"
LOG_FILE = STATE_DIR / "auto_refresh.log"
RUNNER = ROOT / "ms_intraday_runner.py"
def now():
    return datetime.now(timezone.utc).isoformat()
def read_state():
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
def write_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_DIR / (".auto_refresh_state_" + str(os.getpid()) + ".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, STATE_FILE)
def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
def daily_status():
    state = read_state()
    if state.get("status") == "running":
        pid = state.get("pid")
        if pid and not pid_alive(pid):
            state["status"] = "unknown"
            state["error"] = "Processo interrotto senza esito registrato"
    return state
def start_daily(force=False):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    from ms_shared_process_lock import shared_start_lock
    with shared_start_lock():
        from ms_weekly_supervisor import weekly_status
        if weekly_status().get("status") == "running":
            return False, "WEEKLY in esecuzione"
        with open(LOCK_FILE, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = daily_status()
            if state.get("status") == "running" and pid_alive(state.get("pid")):
                return False, "Aggiornamento DAILY già in corso"
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"]
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_state({"pid": process.pid, "status": "running", "started_at": now(), "last_check_at": now(), "last_success_at": state.get("last_success_at"), "error": None})
            return True, "Aggiornamento DAILY avviato"
def worker():
    state = read_state()
    worker_pid = os.getpid()
    try:
        with open(LOG_FILE, "a", encoding="utf-8", buffering=1) as log:
            log.write("\n=== DAILY START " + now() + " ===\n")
            env = os.environ.copy()
            # DAILY ottimizzato: WINDOW520 + FASTCSV V4.
            env["MS_DAILY_WINDOW_520"] = "1"
            env["MS_DAILY_INCREMENTAL"] = "1"
            env["MS_DAILY_FASTCOPY"] = "0"
            completed = subprocess.run([sys.executable, "-u", str(RUNNER)], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=False)
            log.write("\n=== DAILY EXIT " + str(completed.returncode) + " " + now() + " ===\n")
        if completed.returncode != 0:
            raise RuntimeError("Runner DAILY terminato con codice " + str(completed.returncode) + " (consultare auto_refresh.log)")
        try:
            from ms_weekly_refresh_guard import check_weekly_refresh

            should_refresh, reason, latest_daily = (
                check_weekly_refresh()
            )

            if not should_refresh:
                with open(LOG_FILE, "a", encoding="utf-8") as log:
                    log.write(
                        "\n=== WEEKLY SKIP "
                        + now()
                        + " === "
                        + reason
                        + " | DAILY "
                        + str(latest_daily)
                        + "\n"
                    )

                state["weekly_provisional_status"] = "skipped"
                state["weekly_provisional_error"] = None
            else:
                # WEEKLY usa esclusivamente la cache DAILY già pubblicata.
                # Nessun secondo download Yahoo.
                with open(LOG_FILE, "a", encoding="utf-8", buffering=1) as log:
                    log.write("\n=== WEEKLY PROVISIONAL START " + now() + " ===\n")
                    weekly = subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            str(ROOT / "ms_weekly_isolated_runner.py"),
                        ],
                        cwd=str(ROOT),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=2700,
                        check=False,
                    )

                    weekly_code = weekly.returncode

                    log.write(
                        "\n=== WEEKLY PROVISIONAL EXIT "
                        + str(weekly_code) + " " + now() + " ===\n"
                    )

                if weekly_code != 0:
                    state["weekly_provisional_status"] = "error"
                    state["weekly_provisional_error"] = (
                        "Calcolo WEEKLY provvisorio non riuscito "
                        f"(codice {weekly_code}); consultare auto_refresh.log"
                    )
                else:
                    state["weekly_provisional_status"] = "success"
                    state["weekly_provisional_error"] = None
                    state["weekly_provisional_last_success_at"] = now()

                    # Consolidamento solo dopo WEEKLY pubblicato e validato.
                    from ms_weekly_consolidation import consolidate
                    try:
                        consolidated, message = consolidate()
                    except Exception as exc:
                        consolidated = False
                        message = f"{type(exc).__name__}: {exc}"
                        state["weekly_consolidation_status"] = "error"
                    else:
                        state["weekly_consolidation_status"] = (
                            "success" if consolidated else "skipped"
                        )

                    state["weekly_consolidation_message"] = message

                    with open(LOG_FILE, "a", encoding="utf-8") as log:
                        log.write(
                            "\n=== WEEKLY CONSOLIDATION "
                            + now() + " === " + message + "\n"
                        )

        except Exception as weekly_exc:
            state["weekly_provisional_status"] = "error"
            state["weekly_provisional_error"] = (
                f"{type(weekly_exc).__name__}: {weekly_exc}"
            )
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(
                    "\n=== WEEKLY PROVISIONAL ERROR "
                    + now() + " ===\n"
                    + state["weekly_provisional_error"] + "\n"
                )
        # Il DAILY è già stato pubblicato correttamente.
        # Un errore WEEKLY non ne annulla il successo.
        state["status"] = "success"
        state["last_success_at"] = now()
        state["error"] = None

        # Timestamp comune soltanto se entrambi i timeframe
        # sono stati aggiornati o verificati come allineati.
        if state.get("weekly_provisional_status") in (
            "success", "skipped"
        ):
            state["coordinated_last_success_at"] = (
                state["last_success_at"]
            )
    except Exception as exc:
        state["status"] = "error"
        state["error"] = str(exc)
    state["finished_at"] = now()
    current = read_state()
    if current.get("pid") == worker_pid:
        current.update({
            "status": state["status"],
            "finished_at": state["finished_at"],
            "error": state.get("error"),
            "last_success_at": state.get(
                "last_success_at", current.get("last_success_at")
            ),
            "weekly_provisional_status": state.get(
                "weekly_provisional_status",
                current.get("weekly_provisional_status")
            ),
            "weekly_provisional_error": state.get(
                "weekly_provisional_error",
                current.get("weekly_provisional_error")
            ),
            "weekly_provisional_last_success_at": state.get(
                "weekly_provisional_last_success_at",
                current.get("weekly_provisional_last_success_at")
            ),
            "coordinated_last_success_at": state.get(
                "coordinated_last_success_at",
                current.get("coordinated_last_success_at")
            ),
            "weekly_consolidation_status": state.get(
                "weekly_consolidation_status",
                current.get("weekly_consolidation_status")
            ),
            "weekly_consolidation_message": state.get(
                "weekly_consolidation_message",
                current.get("weekly_consolidation_message")
            ),
        })
        write_state(current)
if __name__ == "__main__" and "--worker" in sys.argv:
    worker()
