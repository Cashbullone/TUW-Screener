#!/usr/bin/env python3
"""Parameter-Test für MN und MO: Konzentration, Risiko je Trade, Hebel.
Jede Variante läuft einmal über den ganzen Zeitraum, ausgewertet wird getrennt für
Training (2012 bis 2019), Test (2020 bis 2026) und gesamt. Die Parameter werden nicht je Zeitraum neu eingestellt."""
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
SPLIT = pd.Timestamp("2019-12-27")


def mn_scen():
    out = [("MN Standard (10 Pos., Risiko 1,25 %)", {})]
    for n in (10, 6, 4):
        for r in (1.25, 2.0, 3.0):
            if n == 10 and r == 1.25:
                continue
            out.append(("MN %d Pos., Risiko %s %%" % (n, str(r).replace(".", ",")), dict(np=n, risk=r, max_pos_pct=25.0 if n == 10 else 40.0)))
    for L in (1.5, 2.0):
        out.append(("MN Standard, Hebel %sx" % str(L).replace(".", ","), dict(lev=L)))
        out.append(("MN 6 Pos., Risiko 2 %%, Hebel %sx" % str(L).replace(".", ","), dict(np=6, risk=2.0, max_pos_pct=40.0, lev=L)))
        out.append(("MN 4 Pos., Risiko 3 %%, Hebel %sx" % str(L).replace(".", ","), dict(np=4, risk=3.0, max_pos_pct=40.0, lev=L)))
    return out


def mo_scen():
    A = dict(risk=0.5, max_pos=6, cap=25.0)
    B = dict(risk=1.0, max_pos=5, cap=30.0)
    off = dict(market=False, brake=False)
    return [("MO Standard (10 Pos., Risiko 0,25 %)", {}), ("MO ohne Marktfilter und Bremse", off),
            ("MO 10 Pos., Risiko 0,5 %", dict(risk=0.5)), ("MO 6 Pos., Risiko 0,5 %, Cap 25 %", A),
            ("MO 5 Pos., Risiko 1,0 %, Cap 30 %", B), ("MO 6 Pos., Risiko 0,5 %, ohne Markt/Bremse", dict(A, **off)),
            ("MO 5 Pos., Risiko 1,0 %, ohne Markt/Bremse", dict(B, **off)),
            ("MO Standard, Hebel 1,5x", dict(lev=1.5)), ("MO Standard, Hebel 2x", dict(lev=2.0)),
            ("MO 6 Pos., Risiko 0,5 %, Hebel 1,5x", dict(A, lev=1.5)), ("MO 6 Pos., Risiko 0,5 %, Hebel 2x", dict(A, lev=2.0))]


def split_stats(cal, y):
    k = int(np.searchsorted(cal.values, np.datetime64(SPLIT), side="right")) - 1
    return dict(full=bt.curve_stats(y, cal, start=0), train=bt.curve_stats(y[:k + 1], cal[:k + 1], start=0),
                test=bt.curve_stats(y[k:], cal[k:], start=0))


def main():
    import fetch_data as fd
    os.makedirs(OUT_DIR, exist_ok=True)
    ix_map, status, names, sectors = fd.build_universe()
    sectors = dict(fd.load_sector_file(), **sectors)
    tickers = sorted(ix_map)
    start = (pd.Timestamp.now() - pd.DateOffset(years=YEARS + 2)).strftime("%Y-%m-%d")
    raw, failed = fd.download(tickers + ["SPY", "^GSPC", "^VIX"], start=start)
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

    def series(sym):
        c = raw[sym]["Close"].dropna()
        c.index = pd.DatetimeIndex([pd.Timestamp(i).tz_localize(None) if pd.Timestamp(i).tzinfo else pd.Timestamp(i) for i in c.index])
        return c[~c.index.duplicated(keep="last")].sort_index().reindex(dates).ffill().bfill().to_numpy()

    spx, vix = series("^GSPC"), series("^VIX")
    runs = []

    def add(group, name, res):
        runs.append((group, name, ba.weekly_curve(res["eq"], dates), len(res["closed"]), float(np.mean(res["npos"][bt.WARM:]))))
        fd.log("  %-4s %-48s %d Trades" % (group, name, len(res["closed"])))

    rev_f = bs.review_days(dates)
    Fm = bs.mn_features(P, rev_f)
    for name, kw in (mn_scen()[:2] if FAST else mn_scen()):
        add("MN", name, bs.run_mn(P, Fm, rev_f, **kw))
    del Fm
    rev_w = bs.review_days(dates, 2)
    Fo = bs.mo_features(P, rev_w)
    Mo = bs.market_state(P, spx, vix, rev_w, bs.MO_DEF)
    for name, kw in (mo_scen()[:2] if FAST else mo_scen()):
        add("MO", name, bs.run_mo(P, Fo, Mo, rev_w, **kw))

    bench = {}
    adj = raw["SPY"]["Adj Close"].dropna()
    adj.index = pd.DatetimeIndex([pd.Timestamp(i).tz_localize(None) if pd.Timestamp(i).tzinfo else pd.Timestamp(i) for i in adj.index])
    bench["S&P 500 (SPY, mit Dividenden)"] = adj.resample("W-FRI").last().dropna()
    r = P["C"][:, 1:] / P["C"][:, :-1] - 1
    r = np.where(np.isfinite(r) & (np.abs(r) < 1.0), r, np.nan)
    ew = np.concatenate(([1.0], np.cumprod(1 + np.nan_to_num(np.nanmean(r, axis=0)))))
    bench["Alle Aktien gleichgewichtet"] = ba.weekly_curve(ew, dates)

    last = min(min(x[2].index[-1] for x in runs), min(s.index[-1] for s in bench.values()))
    cal = pd.date_range(ba.STAT_START, last, freq="W-FRI")
    allc = {"%s|%s" % (g, n): c for g, n, c, _, _ in runs}
    allc.update({"BM|" + k: v for k, v in bench.items()})
    A = ba.align(allc, cal)
    rows = []
    for g, n, c, nt, npos in runs:
        y = A["%s|%s" % (g, n)]
        rows.append(dict(group=g, name=n, trades=nt, npos=npos, st=split_stats(cal, y), yearly=bt.yearly(y, cal, start=0),
                         curve=[round(float(v), 4) for v in y]))
    brow = [dict(name=k, st=split_stats(cal, A["BM|" + k]), yearly=bt.yearly(A["BM|" + k], cal, start=0),
                 curve=[round(float(v), 4) for v in A["BM|" + k]]) for k in bench]
    lines = ["%-4s %-50s %s" % ("", "Variante", "  Training p.a./DD     Test p.a./DD       Gesamt p.a./DD   Sharpe")]
    for x in rows + brow:
        s = x["st"]
        lines.append("%-4s %-50s %s  %s  %s  %5.2f" % (x.get("group", ""), x["name"],
                     *["%7s/%5s" % (ba.fmt(s[k].get("cagr")), ba.fmt(s[k].get("maxdd"), 0)) for k in ("train", "test", "full")], s["full"].get("sharpe", float("nan"))))
    text = "\n".join(lines)
    print(text)
    out = dict(meta=dict(von=str(cal[0].date()), bis=str(cal[-1].date()), split=str(SPLIT.date()), aktien=len(daily)),
               rows=rows, bench=brow, dates=[str(x.date()) for x in cal], text=text)
    blob = json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o))
    with open(os.path.join(OUT_DIR, "result_sweep.json"), "w", encoding="utf-8") as f:
        f.write(blob)
    print("RESULT_JSON_START %d" % len(blob))
    for k in range(0, len(blob), 800):
        print("BTJ %05d %s" % (k // 800, blob[k:k + 800]))
    print("RESULT_JSON_END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
