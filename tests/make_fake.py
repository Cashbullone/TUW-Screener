"""Erzeugt ein künstliches Aktienuniversum zum Testen (ohne Netzwerk). Tagesdaten, die der Screener zu Wochenkerzen macht.

Neben Zufallsreihen enthält es benannte Fälle, die per Suche unter vielen Zufallsreihen gefunden werden:
  S1TK  System-1-Ausbruch in der letzten Woche, wird gehandelt     S1SK  System-1-Ausbruch in der letzten Woche, wird übersprungen (Skip-Regel)
  S2ON  55-Wochen-Ausbruch in der letzten Woche                     NEAR  knapp unter der 20-Wochen-Marke, S1 erlaubt
  SKIP  knapp unter der 20-Wochen-Marke, S1 würde übersprungen   SHRT  Short-Ausbruch in der letzten Woche
  RUN   langer Aufwärtstrend (System-1-Trade läuft)
Start: python tests/make_fake.py OUTDIR  schreibt OUTDIR/prices.json und meta.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import turtle_core as tc  # noqa: E402

N_DAYS = 900
IDX = pd.bdate_range(end="2026-10-30", periods=N_DAYS)
SECS = ["Information Technology", "Health Care", "Financials", "Consumer Discretionary", "Communication Services",
        "Industrials", "Consumer Staples", "Energy", "Utilities", "Real Estate", "Materials"]


def random_series(seed, drift=None, vol=None, start=None):
    r = np.random.default_rng(seed)
    drift = r.normal(0.0004, 0.0008) if drift is None else drift
    vol = r.uniform(0.01, 0.025) if vol is None else vol
    start = r.uniform(20, 300) if start is None else start
    c = start * np.exp(np.cumsum(drift + vol * r.standard_normal(N_DAYS)))
    gap = 1 + 0.4 * vol * r.standard_normal(N_DAYS)
    o = np.concatenate(([c[0]], c[:-1])) * gap
    h = np.maximum(o, c) * (1 + vol * r.uniform(0.1, 0.8, N_DAYS))
    l = np.minimum(o, c) * (1 - vol * r.uniform(0.1, 0.8, N_DAYS))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": float(r.integers(1, 20) * 1_000_000)}, index=IDX)


def classify(df):
    df = tc.to_weekly(df)
    res = tc.analyse(df["Open"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy())
    if res is None:
        return None, None
    n, p = res["n"], float(df["Close"].iloc[-1])
    f_l = res["f"][0]
    if "1L" in res["sg"] and "2L" not in res["sg"]:
        return "S1TK", res
    if "1L" in res["sk"] and "2L" not in res["sg"]:
        return "S1SK", res
    if "2L" in res["sg"]:
        return "S2ON", res
    if any(x.startswith("1S") for x in res["sg"]) or "2S" in res["sg"]:
        return "SHRT", res
    near = (res["hh"][1] - p) / n
    if f_l[1] == 0 and 0.15 < near < 0.7 and f_l[0] != "W":
        return "NEAR", res
    if f_l[1] == 0 and 0.15 < near < 0.7 and f_l[0] == "W":
        return "SKIP", res
    return None, res


def build_fake(n_random=330, seed=7):
    raw, ix, names, sectors = {}, {}, {}, {}
    want = {"S1TK": 2, "S1SK": 2, "S2ON": 2, "NEAR": 3, "SKIP": 3, "SHRT": 2}
    got = {k: 0 for k in want}
    k = 0
    while any(got[x] < want[x] for x in want) and k < 6000:
        k += 1
        df = random_series(10_000 + k)
        cat, _ = classify(df)
        if cat in want and got[cat] < want[cat]:
            got[cat] += 1
            name = "%s%s" % (cat, "ABC"[got[cat] - 1])
            raw[name] = df
    # langer Aufwärtstrend
    raw["RUN"] = random_series(1, drift=0.0030, vol=0.008, start=40)
    for i in range(n_random):
        raw["R%s" % "".join(chr(65 + int(d)) for d in "%04d" % i)] = random_series(seed * 1000 + i)
    for i, t in enumerate(sorted(raw)):
        ix[t] = {"SP500"} if i % 4 else {"NDX"}
        names[t] = "Firma %s" % t
        sectors[t] = SECS[i % len(SECS)]
    sectors["RUN"] = "Information Technology"
    return raw, ix, names, sectors, got


if __name__ == "__main__":
    import fetch_data as fd
    out = sys.argv[1] if len(sys.argv) > 1 else "site_test"
    os.makedirs(out, exist_ok=True)
    raw, ix, names, sectors, got = build_fake()
    payload = fd.assemble(raw, ix, names, sectors=sectors, min_price=5, min_avg_vol=0)
    payload["fx"] = 1.08
    payload["failed"] = []
    with open(os.path.join(out, "prices.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print("gefundene Fälle:", got, "Aktien:", len(payload["stocks"]), "Bytes:", os.path.getsize(os.path.join(out, "prices.json")))
