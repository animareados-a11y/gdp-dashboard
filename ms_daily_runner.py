# -*- coding: utf-8 -*-
"""DAILY consolidated refresh in a process isolated from WEEKLY imports.
Scientific methods are imported unchanged from ms_daily_engine.py.
"""
import sys
import traceback

class _Progress:
    def progress(self, value, text=None):
        if text:
            print(text, flush=True)
        return self


def main():
    from ms_daily_engine import update_daily_prices, recalculate, refresh_targets
    bar = _Progress()
    cutoff, elapsed, needs_calculation = update_daily_prices(bar)
    if needs_calculation:
        recalculate(bar)
        refresh_targets(bar)
        print(f"DAILY COMPLETATO: {cutoff:%d/%m/%Y}", flush=True)
    else:
        print(f"DAILY GIA' AGGIORNATO: {cutoff:%d/%m/%Y}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
