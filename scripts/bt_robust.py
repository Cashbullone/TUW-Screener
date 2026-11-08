#!/usr/bin/env python3
"""Robustheit der besten MN-Varianten: Startdatum, Rauschen in der Auswahl, Nachbarparameter.
Alle Läufe werden auf demselben Zeitfenster ausgewertet (ab dem spätesten Start), dazu Rückgänge 2020 und 2022."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import backtest as bt
import bt_screeners as bs
import bt_all as ba

OUT_DIR = os.environ.get("BT_OUT", "backtest_out")
YEARS = int(os.environ.get("BT_YEARS", "15"))
FAST = os.environ.get("BT_FAST", "") == "1"
STARTS = [260, 285, 310, 335, 360]
BEST = dict(np=6, risk=2.0, max_pos_pct=40.0, lev=1.5)
VARS = [("Beste (6 Pos., Risiko 2 %, Hebel 1,5x)", BEST), ("Beste ohne Hebel", dict(BEST, lev=1.0)),
        ("Standard", {}), ("Standard, Hebel 2x", dict(lev=2.0))]
NEIGH = [("Risiko 1,5 %", dict(risk=1.5)), ("Risiko 2,5 %", dict(risk=2.5)), ("5 Positionen", dict(np=5)), ("8 Positionen", dict(np=8)),
         ("Stop 6,5 %", dict(stop=6.5)), ("Stop 9 %", dict(stop=9.0))]


def window_stats(cal, y):
    s = bt.curve_stats(y, cal, start=0)
    out = dict(cagr=s["cagr"], maxdd=s["maxdd"], sharpe=s["sharpe"], total=s["total"])
    for yr in (2020, 2022):
        m = (cal.year == yr)
        idx = np.where(m)[0]
        if len(idx) < 5:
            continue
        a = max(idx[0] - 1, 0)
        seg = y[a:idx[-1] + 1]
        out["r%d" % yr] = float(seg[-1] / seg[0] - 1)
        out["dd%d" % yr] = float((seg / np.maximum.accumulate(seg) - 1).min())
    m = (cal >= pd.Timestamp("2020-02-14")) & (cal <= pd.Timestamp("2020-04-10"))
    seg = y[np.where(m)[0]]
    out["covid"] = float(seg.min() / seg[0] - 1) if len(seg) else None
    return out


def main():
    import fetch_data as fd
    os.makedirs(OUT_DIR, exist_ok=True)
    ix_map, status, names, sectors = fd.build_universe()
    sectors = dict(fd.load_sector_file(), **sectors)
    tickers = sorted(ix_map)
    start = (pd.Timestamp.now() - pd.DateOffset(years=YEARS + 2)).strftime("%Y-%m-%d")
    raw, failed = fd.download(tickers + ["SPY"], start=start)
    drop_after = fd.last_complete_day()
    daily = {}
    for t in tickers:
        d = fd.clean_frame(raw.get(t))
        if d is None:
            continue
        d = d[d.index <= drop_after]
        if len(d) >= 400:
            daily[t] = d
    fd.log("%d Aktien mit genug Kursen" % len(daily))
    P = bt.build_panel(daily, sectors)
    dates = P["dates"]
    rev = bs.review_days(dates)
    F = bs.mn_features(P, rev)
    runs = []

    def add(var, kind, kw):
        res = bs.run_mn(P, F, rev, **kw)
        runs.append((var, kind, ba.weekly_curve(res["eq"], dates), len(res["closed"])))
        fd.log("  %-42s %-16s %d Trades" % (var, kind, len(res["closed"])))

    vars_ = VARS[:1] if FAST else VARS
    for vn, vkw in vars_:
        for st in (STARTS[:2] if FAST else STARTS):
            add(vn, "Start %d" % st, dict(vkw, start=st))
        for sd in ((1,) if FAST else (1, 2, 3, 4, 5)):
            add(vn, "Rauschen %d" % sd, dict(vkw, jit=8.0, seed=sd))
    if not FAST:
        for nn, nkw in NEIGH:
            add(VARS[0][0], "Nachbar " + nn, dict(BEST, **nkw))
    bench = {}
    adj = raw["SPY"]["Adj Close"].dropna()
    adj.index = pd.DatetimeIndex([pd.Timestamp(i).tz_localize(None) if pd.Timestamp(i).tzinfo else pd.Timestamp(i) for i in adj.index])
    spy = adj.resample("W-FRI").last().dropna()
    bench["S&P 500 (SPY, mit Dividenden)"] = spy
    for L in (1.5, 2.0):
        r = spy.to_numpy()[1:] / spy.to_numpy()[:-1] - 1
        r = L * r - (L - 1) * 0.05 / 52
        bench["S&P 500, Hebel %sx (wöchentlich angepasst)" % str(L).replace(".", ",")] = pd.Series(np.concatenate(([1.0], np.cumprod(1 + r))), index=spy.index)
    last = min(min(x[2].index[-1] for x in runs), min(s.index[-1] for s in bench.values()))
    cal = pd.date_range(max(dates[max(STARTS)], pd.Timestamp("2012-01-06")), last, freq="W-FRI")
    rows = []
    for var, kind, c, nt in runs:
        x = c.reindex(cal.union(c.index)).ffill().reindex(cal)
        rows.append(dict(var=var, kind=kind, trades=nt, st=window_stats(cal, (x / x.iloc[0]).to_numpy())))
    brow = []
    for k, c in bench.items():
        x = c.reindex(cal.union(c.index)).ffill().reindex(cal)
        brow.append(dict(var=k, kind="", trades=0, st=window_stats(cal, (x / x.iloc[0]).to_numpy())))
    f = lambda v: "   –  " if v is None else ba.fmt(v).rjust(7)
    lines = ["Zeitfenster %s bis %s" % (cal[0].date(), cal[-1].date()),
             "%-42s %-24s %7s %7s %7s %7s %7s %7s" % ("Variante", "Lauf", "p.a.", "MaxDD", "Covid", "2020", "2022", "DD2022")]
    for x in rows + brow:
        s = x["st"]
        lines.append("%-42s %-24s %s %s %s %s %s %s" % (x["var"][:42], x["kind"], f(s["cagr"]), f(s["maxdd"]), f(s.get("covid")), f(s.get("r2020")), f(s.get("r2022")), f(s.get("dd2022"))))
    text = "\n".join(lines)
    print(text)
    out = dict(meta=dict(von=str(cal[0].date()), bis=str(cal[-1].date())), rows=rows, bench=brow, text=text)
    blob = json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o))
    with open(os.path.join(OUT_DIR, "result_robust.json"), "w", encoding="utf-8") as fh:
        fh.write(blob)
    print("RESULT_JSON_START %d" % len(blob))
    for k in range(0, len(blob), 800):
        print("BTJ %05d %s" % (k // 800, blob[k:k + 800]))
    print("RESULT_JSON_END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
