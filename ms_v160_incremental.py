
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["Ticker", "Date"]


def incremental_write(old_path, updated, output_path):
    old_path = Path(old_path)
    output_path = Path(output_path)

    if old_path.resolve() == output_path.resolve():
        raise RuntimeError("Sorgente e destinazione coincidono")

    if not old_path.is_file():
        raise FileNotFoundError(old_path)

    if output_path.exists():
        raise RuntimeError("Destinazione già esistente")

    old = pd.read_csv(old_path, low_memory=False)
    updated = updated.copy()

    if list(old.columns) != list(updated.columns):
        raise RuntimeError("Schema V160 differente")

    if old.duplicated(KEYS).any():
        raise RuntimeError("Duplicati nello storico")

    if updated.duplicated(KEYS).any():
        raise RuntimeError("Duplicati nei nuovi dati")

    old_dates = pd.to_datetime(old["Date"], utc=True)
    new_dates = pd.to_datetime(updated["Date"], utc=True)

    if old_dates.isna().any() or new_dates.isna().any():
        raise RuntimeError("Date non valide")

    old["Date"] = old_dates.dt.strftime("%Y-%m-%d")
    updated["Date"] = new_dates.dt.strftime("%Y-%m-%d")

    if not old[KEYS].reset_index(drop=True).equals(
        old.sort_values(KEYS, kind="stable")[KEYS]
           .reset_index(drop=True)
    ):
        raise RuntimeError("Storico non ordinato Ticker/Date")

    old_index = old.set_index(KEYS)
    new_index = updated.set_index(KEYS)

    if not old_index.index.isin(new_index.index).all():
        raise RuntimeError("Righe storiche mancanti")

    for col in old_index.columns:
        a = old_index[col]
        b = new_index.loc[old_index.index, col]

        if (
            pd.api.types.is_numeric_dtype(a)
            and pd.api.types.is_numeric_dtype(b)
        ):
            equal = np.isclose(
                a.to_numpy(dtype=float),
                b.to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True
            ).all()
        else:
            equal = (
                a.fillna("<NULL>").astype(str).to_numpy()
                == b.fillna("<NULL>").astype(str).to_numpy()
            ).all()

        if not equal:
            raise RuntimeError(
                f"Differenza scientifica: {col}"
            )

    old_keys = old_index.index
    updated_keys = pd.MultiIndex.from_frame(updated[KEYS])

    additions = updated.loc[
        ~updated_keys.isin(old_keys)
    ].copy()

    additions = additions.sort_values(KEYS)

    if additions.empty:
        return {
            "new_rows": 0,
            "status": "unchanged"
        }

    last_dates = (
        old.groupby("Ticker")["Date"].max()
    )

    for ticker, group in additions.groupby("Ticker"):
        if ticker not in last_dates.index:
            raise RuntimeError(
                f"Nuovo ticker non previsto: {ticker}"
            )

        if (group["Date"] <= last_dates.loc[ticker]).any():
            raise RuntimeError(
                f"Inserimento retroattivo vietato: {ticker}"
            )

    additions_by_ticker = {
        str(ticker): group
        for ticker, group in additions.groupby("Ticker")
    }

    temp = output_path.with_name(
        output_path.name + ".fastcsv.tmp"
    )

    if temp.exists():
        raise RuntimeError("File temporaneo già presente")

    try:
        with (
            old_path.open("r", encoding="utf-8", newline="") as source,
            temp.open("w", encoding="utf-8", newline="") as target
        ):
            header = source.readline()
            if not header:
                raise RuntimeError("CSV sorgente vuoto")

            target.write(header)

            previous_ticker = None
            inserted = set()
            last_seen_date = {}

            def insert_new(ticker):
                if ticker not in additions_by_ticker:
                    return

                group = additions_by_ticker[ticker]
                group.to_csv(
                    target,
                    index=False,
                    header=False,
                    lineterminator="\n"
                )
                inserted.add(ticker)

            ticker_idx = list(old.columns).index("Ticker")
            date_idx = list(old.columns).index("Date")

            for line in source:
                if not line.strip():
                    raise RuntimeError("Riga CSV vuota")

                # Le prime due chiavi vengono lette con il parser CSV.
                # La riga originale viene copiata senza serializzare
                # nuovamente le 257 colonne.
                fields = next(csv.reader([line]))

                if len(fields) != len(old.columns):
                    raise RuntimeError("Numero colonne CSV errato")

                ticker = fields[ticker_idx]
                date = fields[date_idx]

                if previous_ticker is not None and ticker != previous_ticker:
                    insert_new(previous_ticker)

                if ticker in last_seen_date:
                    if date <= last_seen_date[ticker]:
                        raise RuntimeError(
                            f"Ordine Date errato: {ticker}"
                        )

                last_seen_date[ticker] = date
                target.write(line)
                previous_ticker = ticker

            if previous_ticker is not None:
                insert_new(previous_ticker)

            if set(additions_by_ticker) != inserted:
                raise RuntimeError(
                    "Nuovi ticker non inseriti"
                )

        actual = pd.read_csv(temp, low_memory=False)

        expected = updated.sort_values(
            KEYS
        ).reset_index(drop=True)

        actual = actual.sort_values(
            KEYS
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-6,
            atol=1e-8
        )

        os.replace(temp, output_path)

        return {
            "new_rows": len(additions),
            "status": "validated"
        }

    finally:
        if temp.exists():
            temp.unlink()



def incremental_write_v3(old_path, reference, updated, output_path):
    """
    FASTCSV V3:
    - riutilizza i DataFrame ricevuti;
    - conserva le righe CSV storiche;
    - inserisce nuove date alla fine del blocco ticker;
    - verifica integralmente il risultato;
    - non modifica il file sorgente.
    """
    import csv
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd

    old_path = Path(old_path)
    output_path = Path(output_path)
    keys = ["Ticker", "Date"]

    if old_path.resolve() == output_path.resolve():
        raise RuntimeError("Sorgente e destinazione coincidono")

    if not old_path.is_file():
        raise FileNotFoundError(old_path)

    if output_path.exists():
        raise RuntimeError("Destinazione già esistente")

    if list(reference.columns) != list(updated.columns):
        raise RuntimeError("Schema V160 differente")

    if reference.duplicated(keys).any():
        raise RuntimeError("Duplicati nello storico")

    if updated.duplicated(keys).any():
        raise RuntimeError("Duplicati nei dati aggiornati")

    ref = reference.copy()
    new = updated.copy()

    for df in (ref, new):
        dates = pd.to_datetime(df["Date"], utc=True, errors="raise")
        if dates.isna().any():
            raise RuntimeError("Date non valide")
        df["Date"] = dates.dt.strftime("%Y-%m-%d")
        df["Ticker"] = df["Ticker"].astype(str)

    if not ref[keys].reset_index(drop=True).equals(
        ref.sort_values(keys, kind="stable")[keys].reset_index(drop=True)
    ):
        raise RuntimeError("Storico non ordinato Ticker/Date")

    ri = ref.set_index(keys)
    ni = new.set_index(keys)

    if not ri.index.isin(ni.index).all():
        raise RuntimeError("Righe storiche mancanti")

    # Validazione scientifica completa dello storico.
    for col in ri.columns:
        a = ri[col]
        b = ni.loc[ri.index, col]

        if (
            pd.api.types.is_numeric_dtype(a)
            and pd.api.types.is_numeric_dtype(b)
        ):
            equal = np.isclose(
                a.to_numpy(dtype=float),
                b.to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True
            ).all()
        else:
            equal = (
                a.fillna("<NULL>").astype(str).to_numpy()
                == b.fillna("<NULL>").astype(str).to_numpy()
            ).all()

        if not equal:
            raise RuntimeError(
                f"Differenza scientifica: {col}"
            )

    new_keys = pd.MultiIndex.from_frame(new[keys])
    additions = new.loc[~new_keys.isin(ri.index)].copy()
    additions = additions.sort_values(keys)

    if additions.empty:
        return {"new_rows": 0, "status": "unchanged"}

    last_dates = ref.groupby("Ticker")["Date"].max()

    for ticker, group in additions.groupby("Ticker"):
        if ticker not in last_dates.index:
            raise RuntimeError(f"Ticker sconosciuto: {ticker}")

        if (group["Date"] <= last_dates.loc[ticker]).any():
            raise RuntimeError(
                f"Inserimento retroattivo: {ticker}"
            )

    additions_by_ticker = {
        str(ticker): group
        for ticker, group in additions.groupby("Ticker", sort=False)
    }

    temp = output_path.with_name(
        output_path.name + ".v3.tmp"
    )

    if temp.exists():
        raise RuntimeError("File temporaneo già presente")

    try:
        with (
            old_path.open("r", encoding="utf-8", newline="") as source,
            temp.open("w", encoding="utf-8", newline="") as target
        ):
            header = source.readline()
            if not header:
                raise RuntimeError("CSV storico vuoto")

            parsed_header = next(csv.reader([header]))
            if parsed_header != list(ref.columns):
                raise RuntimeError("Intestazione CSV differente")

            target.write(header)

            previous_ticker = None
            last_seen_date = {}
            inserted = set()
            row_count = 0

            ticker_idx = parsed_header.index("Ticker")
            date_idx = parsed_header.index("Date")

            def insert_new(ticker):
                if ticker not in additions_by_ticker:
                    return

                additions_by_ticker[ticker].to_csv(
                    target,
                    index=False,
                    header=False,
                    lineterminator="\n"
                )
                inserted.add(ticker)

            for line in source:
                if not line.strip():
                    raise RuntimeError("Riga CSV vuota")

                fields = next(csv.reader([line]))

                if len(fields) != len(parsed_header):
                    raise RuntimeError("Numero colonne non valido")

                ticker = fields[ticker_idx]
                date = fields[date_idx]

                if (
                    previous_ticker is not None
                    and ticker != previous_ticker
                ):
                    insert_new(previous_ticker)

                if (
                    ticker in last_seen_date
                    and date <= last_seen_date[ticker]
                ):
                    raise RuntimeError(
                        f"Ordine date non valido: {ticker}"
                    )

                last_seen_date[ticker] = date
                target.write(line)
                previous_ticker = ticker
                row_count += 1

            if previous_ticker is not None:
                insert_new(previous_ticker)

            if row_count != len(ref):
                raise RuntimeError(
                    "Numero righe sorgente diverso dal riferimento"
                )

            if set(additions_by_ticker) != inserted:
                raise RuntimeError("Nuove righe non inserite")

        # Validazione del CSV finale prima della pubblicazione.
        actual = pd.read_csv(temp, low_memory=False)

        actual["Date"] = pd.to_datetime(
            actual["Date"], utc=True
        ).dt.strftime("%Y-%m-%d")

        expected = new.sort_values(keys).reset_index(drop=True)
        actual = actual.sort_values(keys).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-6,
            atol=1e-8
        )

        os.replace(temp, output_path)

        return {
            "new_rows": len(additions),
            "status": "validated"
        }

    finally:
        if temp.exists():
            temp.unlink()



def incremental_write_v4(old_path, reference, updated, output_path):
    """
    FASTCSV V4 - sperimentale.
    Mantiene intatte le righe CSV storiche e aggiunge nuove sedute.
    Confronta i dati scientifici e verifica l'output per blocchi.
    Non sovrascrive mai il CSV sorgente.
    """
    import csv
    import os
    import shutil
    from pathlib import Path

    import numpy as np
    import pandas as pd

    old_path = Path(old_path)
    output_path = Path(output_path)
    keys = ["Ticker", "Date"]

    if not old_path.is_file():
        raise FileNotFoundError(old_path)

    if old_path.resolve() == output_path.resolve():
        raise RuntimeError("Sorgente e destinazione coincidono")

    if output_path.exists():
        raise RuntimeError("Destinazione già esistente")

    if list(reference.columns) != list(updated.columns):
        raise RuntimeError("Schema V160 differente")

    if not set(keys).issubset(reference.columns):
        raise RuntimeError("Chiavi Ticker/Date mancanti")

    def normalized_keys(df):
        tickers = df["Ticker"].astype(str)
        dates = pd.to_datetime(
            df["Date"], errors="raise", utc=True, format="mixed"
        )
        if dates.isna().any():
            raise RuntimeError("Date mancanti")
        return pd.MultiIndex.from_arrays(
            [tickers, dates.dt.strftime("%Y-%m-%d")],
            names=keys
        )

    old_keys = normalized_keys(reference)
    new_keys = normalized_keys(updated)

    if not old_keys.is_unique or not new_keys.is_unique:
        raise RuntimeError("Chiavi duplicate")

    if not old_keys.is_monotonic_increasing:
        raise RuntimeError("Storico non ordinato")

    if not new_keys.is_monotonic_increasing:
        raise RuntimeError("Aggiornamento non ordinato")

    old_positions = new_keys.get_indexer(old_keys)

    if (old_positions < 0).any():
        raise RuntimeError("Righe storiche mancanti")

    # Confronto scientifico per singola colonna.
    # Evita copie complete dei DataFrame.
    for col in reference.columns:
        if col in keys:
            continue

        a = reference[col]
        b = updated[col].iloc[old_positions]

        if (
            pd.api.types.is_numeric_dtype(a)
            and pd.api.types.is_numeric_dtype(b)
        ):
            equal = np.isclose(
                a.to_numpy(dtype=float, copy=False),
                b.to_numpy(dtype=float, copy=False),
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True
            ).all()
        else:
            equal = (
                a.fillna("<NULL>").astype(str).to_numpy()
                ==
                b.fillna("<NULL>").astype(str).to_numpy()
            ).all()

        if not equal:
            raise RuntimeError(
                f"Differenza scientifica: {col}"
            )

    addition_positions = np.flatnonzero(
        ~new_keys.isin(old_keys)
    )

    last_dates = {
        ticker: date
        for ticker, date in (
            old_keys.to_frame(index=False)
            .groupby("Ticker", sort=False)["Date"]
            .max()
            .items()
        )
    }

    additions = {}

    for pos in addition_positions:
        ticker, date = new_keys[pos]

        if ticker not in last_dates:
            raise RuntimeError(
                f"Ticker nuovo non previsto: {ticker}"
            )

        if date <= last_dates[ticker]:
            raise RuntimeError(
                f"Inserimento retroattivo: {ticker}/{date}"
            )

        additions.setdefault(ticker, []).append(int(pos))

    temp = output_path.with_name(
        output_path.name + ".v4.tmp"
    )

    if temp.exists():
        raise RuntimeError("File temporaneo già presente")

    try:
        with (
            old_path.open("r", encoding="utf-8", newline="") as source,
            temp.open("w", encoding="utf-8", newline="") as target
        ):
            header = source.readline()

            if not header:
                raise RuntimeError("CSV sorgente vuoto")

            fields = next(csv.reader([header]))

            if fields != list(reference.columns):
                raise RuntimeError("Intestazione CSV differente")

            ticker_idx = fields.index("Ticker")
            date_idx = fields.index("Date")

            target.write(header)

            previous_key = None
            previous_ticker = None
            inserted = set()
            row_count = 0

            def append_new(ticker):
                if ticker not in additions:
                    return

                positions = additions[ticker]

                updated.iloc[positions].to_csv(
                    target,
                    index=False,
                    header=False,
                    lineterminator="\n"
                )

                inserted.add(ticker)

            for line in source:
                if not line.strip():
                    raise RuntimeError("Riga CSV vuota")

                values = next(csv.reader([line]))

                if len(values) != len(fields):
                    raise RuntimeError("Riga CSV non valida")

                ticker = values[ticker_idx]
                date = values[date_idx]
                normalized_date = pd.to_datetime(
                    date, utc=True, errors="raise"
                ).strftime("%Y-%m-%d")
                key = (ticker, normalized_date)

                if previous_key is not None and key <= previous_key:
                    raise RuntimeError("CSV sorgente non ordinato")

                if (
                    previous_ticker is not None
                    and ticker != previous_ticker
                ):
                    append_new(previous_ticker)

                if row_count >= len(old_keys):
                    raise RuntimeError("Righe sorgente in eccesso")

                if key != old_keys[row_count]:
                    raise RuntimeError(
                        "CSV sorgente diverso dal DataFrame di riferimento"
                    )

                target.write(line)

                previous_key = key
                previous_ticker = ticker
                row_count += 1

            if previous_ticker is not None:
                append_new(previous_ticker)

            if row_count != len(reference):
                raise RuntimeError("Righe sorgente mancanti")

            if set(additions) != inserted:
                raise RuntimeError("Inserimento incompleto")

        # Verifica dell'intero CSV prodotto in blocchi:
        # non carica contemporaneamente tutto il file finale.
        offset = 0
        chunksize = 2000

        for chunk in pd.read_csv(
            temp,
            chunksize=chunksize,
            low_memory=False
        ):
            end = offset + len(chunk)

            if end > len(updated):
                raise RuntimeError("Output con righe in eccesso")

            expected = updated.iloc[offset:end]

            actual_keys = normalized_keys(chunk)

            if not actual_keys.equals(new_keys[offset:end]):
                raise RuntimeError(
                    f"Chiavi output errate al blocco {offset}"
                )

            for col in updated.columns:
                if col in keys:
                    continue

                a = chunk[col]
                b = expected[col]

                if (
                    pd.api.types.is_numeric_dtype(a)
                    and pd.api.types.is_numeric_dtype(b)
                ):
                    equal = np.isclose(
                        a.to_numpy(dtype=float, copy=False),
                        b.to_numpy(dtype=float, copy=False),
                        rtol=1e-6,
                        atol=1e-8,
                        equal_nan=True
                    ).all()
                else:
                    equal = (
                        a.fillna("<NULL>").astype(str).to_numpy()
                        ==
                        b.fillna("<NULL>").astype(str).to_numpy()
                    ).all()

                if not equal:
                    raise RuntimeError(
                        f"Output differente: {col}, blocco {offset}"
                    )

            offset = end

        if offset != len(updated):
            raise RuntimeError("Output incompleto")

        os.replace(temp, output_path)

        return {
            "status": "validated",
            "new_rows": len(addition_positions),
            "total_rows": len(updated)
        }

    finally:
        if temp.exists():
            temp.unlink()
