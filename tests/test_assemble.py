"""Tests für Aufbereitung und Hauptablauf mit künstlichen Daten (ohne Netzwerk)."""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import fetch_data as fd  # noqa: E402
import make_fake  # noqa: E402
import turtle_core as tc  # noqa: E402

RAW, IX, NAMES, SECS, GOT = make_fake.build_fake(n_random=60)


def test_assemble_structure():
    p = fd.assemble(RAW, IX, NAMES, sectors=SECS, min_price=5, min_avg_vol=0)
    assert p["asof"] == "2026-10-30" and len(p["dates"]) == tc.NH
    by = {s["t"]: s for s in p["stocks"]}
    for s in p["stocks"]:
        for k in ("h", "l", "c", "a"):
            assert len(s[k]) == tc.NH
        assert s["c"][-1] == s["p"] and s["N"] > 0 and len(s["hh"]) == 3 and len(s["f"]) == 2
        assert s["hh"][0] <= s["hh"][1] <= s["hh"][2] and s["ll"][0] >= s["ll"][1] >= s["ll"][2]
        assert abs(s["a"][-1] - s["N"]) < 0.002
        assert s["h"][-1] >= s["c"][-1] >= s["l"][-1]
    assert "1L" in by["S1TKA"]["sg"] and "1L" in by["S1SKA"]["sk"] and "1L" not in by["S1SKA"]["sg"]
    assert "2L" in by["S2ONA"]["sg"] and by["S1TKA"]["sec"] in make_fake.SECS
    json.dumps(p)


def test_filters_and_stale():
    raw = dict(RAW)
    df = raw["RUN"].copy()
    raw["OLD"] = df.iloc[:-12]                          # keine Kerze für die letzten 2 Wochen
    raw["PENNY"] = df * np.array([0.01, 0.01, 0.01, 0.01, 1])        # Kurs unter 5 $
    raw["THIN"] = df.assign(Volume=1000.0)
    p = fd.assemble(raw, {**IX, "OLD": {"SP500"}, "PENNY": {"SP500"}, "THIN": {"SP500"}}, NAMES, min_price=5, min_avg_vol=100000)
    got = {s["t"] for s in p["stocks"]}
    assert "OLD" not in got and "PENNY" not in got and "THIN" not in got and "RUN" in got


def test_bad_rows_are_cleaned():
    df = RAW["RUN"].copy()
    df.iloc[100, df.columns.get_loc("High")] = np.nan
    df.iloc[101, df.columns.get_loc("Low")] = 0.0
    df.iloc[102, df.columns.get_loc("High")] = df.iloc[102]["Low"] * 0.5     # Hoch < Tief
    d = fd.clean_frame(df)
    assert len(d) == len(df) - 3 and (d["High"] >= d["Low"]).all() and (d[["Open", "Low"]] > 0).all().all()


def test_drop_after_removes_unfinished_week():
    # Donnerstag: die Woche ist noch nicht zu Ende, die letzte vollständige Wochenkerze endet am Freitag davor
    p = fd.assemble(RAW, IX, NAMES, min_price=5, min_avg_vol=0, drop_after=pd.Timestamp("2026-10-29"))
    assert p["asof"] == "2026-10-23"
    # Freitag (Börsenschluss durch): die Woche zählt
    p = fd.assemble(RAW, IX, NAMES, min_price=5, min_avg_vol=0, drop_after=pd.Timestamp("2026-10-30"))
    assert p["asof"] == "2026-10-30" and p["tf"] == "W"


def test_weekly_candles_are_correct():
    d = RAW["RUN"]
    w = tc.to_weekly(d)
    last = d[d.index > pd.Timestamp("2026-10-23")]                     # Woche bis Freitag 30.10.
    row = w.loc["2026-10-30"]
    assert row["Open"] == last["Open"].iloc[0] and row["Close"] == last["Close"].iloc[-1]
    assert row["High"] == last["High"].max() and row["Low"] == last["Low"].min()
    assert all(x.weekday() == 4 for x in w.index)


def test_main_writes_files(monkeypatch=None):
    with tempfile.TemporaryDirectory() as tmp:
        fd.OUT_DIR = tmp
        fd.MIN_STOCKS = 10
        fd.build_universe = lambda: (IX, {"SP500": {"ok": True, "n": 1, "new": 1}}, NAMES, SECS)
        fd.download = lambda t: ({k: v for k, v in RAW.items() if k in t}, [])
        fd.fetch_fx = lambda: 1.08
        fd.last_complete_day = lambda: pd.Timestamp("2026-10-30")
        fd.INDEXES = ["SP500", "NDX"]
        assert fd.main() == 0
        pr = json.load(open(os.path.join(tmp, "prices.json")))
        me = json.load(open(os.path.join(tmp, "meta.json")))
        assert pr["fx"] == 1.08 and me["stocks"] == len(pr["stocks"]) and me["signals_week"]["1L"] >= 2


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok   ", name)
            except Exception:
                fails += 1
                import traceback; print("FEHLER", name); traceback.print_exc()
    sys.exit(1 if fails else 0)
