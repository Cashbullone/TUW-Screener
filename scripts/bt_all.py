#!/usr/bin/env python3
"""Backtest aller vier Screener (TU täglich, TUW wöchentlich, MN, MO) auf denselben Kursdaten und demselben Zeitraum.
Universum: heutige Mitglieder von S&P 500 und Nasdaq-100 (Überlebensverzerrung). Auswertung ab STAT_START."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import backtest as bt
import bt_screeners as bs
import turtle_core as tc

OUT_DIR = os.environ.get("BT_OUT", "backtest_out")
YEARS = int(os.environ.get("BT_YEARS", "15"))
STAT_START = pd.Timestamp(os.environ.get("BT_STAT_START", "2012-01-06"))
FAST = os.environ.get("BT_FAST", "") == "1"

TU_KW = dict(mom_weeks=130, dv_win=50, dv_div=1.0)
TU_SCEN = [("Standard (nur Long, Risiko 0,25 %)", {}), ("Long und Short", dict(short=True)),
           ("Nur System 1 (20/10 Tage)", dict(s2=False)), ("Nur System 2 (55/20 Tage)", dict(s1=False)),
           ("Risiko 0,5 % je Einheit", dict(risk=0.5)), ("Risiko 1,0 % je Einheit", dict(risk=1.0, cap_pos=50.0))] \
    + [("Zufallsauswahl bei Limits, Lauf %d" % k, dict(order="random", seed=k)) for k in (1, 2, 3)]
TUW_SCEN = [("Standard (nur Long, Risiko 0,25 %)", {}), ("Long und Short", dict(short=True)),
            ("Risiko 0,5 % je Einheit", dict(risk=0.5)), ("Risiko 1,0 % je Einheit", dict(risk=1.0, cap_pos=50.0))] \
    + [("Zufallsauswahl bei Limits, Lauf %d" % k, dict(order="random", seed=k)) for k in (1, 2, 3)]
MN_SCEN = [("Standard (Risiko 1,25 %, Stop 7,5 %)", {}), ("Ohne Tausch", dict(swap=False)),
           ("Template darf beliebig alt sein", dict(fresh=999)), ("RS-Rating mindestens 90", dict(rs_min=90)),
           ("Ohne Verkaufsstufen", dict(stages="")), ("Risiko 0,75 %", dict(risk=0.75)), ("Stop 10 %", dict(stop=10.0)),
           ("Kosten 0,3 % je Seite", dict(cost=0.003))]
MO_SCEN = [("Standard (10 Positionen, Risiko 0,25 %)", {}), ("Ohne Marktfilter und Bremse", dict(market=False, brake=False)),
           ("Ohne Volatilitätsbremse", dict(brake=False)), ("Ohne Frog in the Pan", dict(fip=False)),
           ("20 Positionen, 10 % je Position", dict(max_pos=20, cap=10.0)), ("Risiko 0,5 % (ATR)", dict(risk=0.5)),
           ("Kosten 0,3 % je Seite", dict(cost=0.003))]


def weekly_curve(eq, dates):
    s = pd.Series(np.asarray(eq, float), index=dates)
    return s.resample("W-FRI").last().dropna()


def align(curves, cal):
    out = {}
    for k, s in curves.items():
        x = s.reindex(cal.union(s.index)).ffill().reindex(cal)
        out[k] = (x / x.iloc[0]).to_numpy()
    return out


def stats_on(cal, y):
    return bt.curve_stats(y, cal, start=0)


def yearly_on(cal, y):
    return bt.yearly(y, cal, start=0)


def trade_info(closed, dates):
    if not closed:
        return dict(n=0)
    pnl = np.array([t["pnl"] for t in closed], float)
    ret = np.array([t["ret"] for t in closed], float)
    days = np.array([(dates[t["i_out"]] - dates[t["i_in"]]).days for t in closed], float)
    w, l = pnl[pnl > 0], pnl[pnl <= 0]
    return dict(n=len(closed), win=float((pnl > 0).mean()), avg_win=float(ret[pnl > 0].mean()) if len(w) else 0.0,
                avg_loss=float(ret[pnl <= 0].mean()) if len(l) else 0.0,
                pf=float(w.sum() / -l.sum()) if len(l) and l.sum() < 0 else float("nan"),
                hold_days=float(days.mean()), top5_share=float(np.sort(pnl)[::-1][:5].sum() / pnl.sum()) if pnl.sum() > 0 else float("nan"))


def fmt(x, d=1):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else ("%.*f %%" % (d, x * 100)).replace(".", ",")


def main():
    import fetch_data as fd
    os.makedirs(OUT_DIR, exist_ok=True)
    ix_map, status, names, sectors = fd.build_universe()
    sectors = dict(fd.load_sector_file(), **sectors)
    tickers = sorted(ix_map)
    start = (pd.Timestamp.now() - pd.DateOffset(years=YEARS + 2)).strftime("%Y-%m-%d")
    fd.log("%d Ticker, Kurse ab %s" % (len(tickers), start))
    raw, failed = fd.download(tickers + ["SPY", "^GSPC", "^VIX"], start=start)
    drop_after = fd.last_complete_day()
    daily, weekly = {}, {}
    for t in tickers:
        d = fd.clean_frame(raw.get(t))
        if d is None:
            continue
        d = d[d.index <= drop_after]
        if len(d) < 400:
            continue
        daily[t] = d
        w = tc.to_weekly(d)
        weekly[t] = w[w.index <= drop_after]
    fd.log("%d Aktien mit genug Kursen" % len(daily))
    Pd = bt.build_panel(daily, sectors)
    Pw = bt.build_panel(weekly, sectors)
    dates = Pd["dates"]

    def series(sym):
        df = raw.get(sym)
        if df is None or "Close" not in df.columns:
            fd.log("  %s nicht abrufbar" % sym)
            return np.full(len(dates), np.nan)
        c = df["Close"].dropna()
        c.index = pd.DatetimeIndex([pd.Timestamp(i).tz_localize(None) if pd.Timestamp(i).tzinfo else pd.Timestamp(i) for i in c.index])
        return c[~c.index.duplicated(keep="last")].sort_index().reindex(dates).ffill().bfill().to_numpy()

    spx, vix = series("^GSPC"), series("^VIX")
    curves, rows = {}, []

    def add(group, name, res, P):
        rows.append(dict(group=group, name=name, eq=weekly_curve(res["eq"], P["dates"]) if P is Pd else pd.Series(res["eq"], index=P["dates"]),
                         trades=trade_info(res["closed"], P["dates"]), npos=float(np.mean(res["npos"][bt.WARM:])) if "npos" in res else None,
                         gross=float(np.mean(res["gross"][bt.WARM:])) if "gross" in res else None,
                         best=sorted([(float(t["ret"]), t["t"] if isinstance(t["t"], str) else P["tick"][t["t"]], str(P["dates"][t["i_in"]].date())) for t in res["closed"]], reverse=True)[:5]))
        fd.log("  %-6s %-42s %d Trades" % (group, name, len(res["closed"])))

    # TU (täglich) und TUW (wöchentlich) mit derselben Simulation
    I, s1 = bt.indicators(Pd, dict(bt.DEFAULTS, **TU_KW))
    for name, kw in (TU_SCEN[:2] if FAST else TU_SCEN):
        add("TU", name, bt.run_backtest(Pd, I, s1, **dict(TU_KW, **kw)), Pd)
    Iw, s1w = bt.indicators(Pw, bt.DEFAULTS)
    for name, kw in (TUW_SCEN[:2] if FAST else TUW_SCEN):
        add("TUW", name, bt.run_backtest(Pw, Iw, s1w, **kw), Pw)
    # MN und MO
    rev_f = bs.review_days(dates)
    Fm = bs.mn_features(Pd, rev_f)
    for name, kw in (MN_SCEN[:2] if FAST else MN_SCEN):
        add("MN", name, bs.run_mn(Pd, Fm, rev_f, **kw), Pd)
    del Fm
    rev_w = bs.review_days(dates, 2)
    Fo = bs.mo_features(Pd, rev_w)
    Mo = bs.market_state(Pd, spx, vix, rev_w, bs.MO_DEF)
    for name, kw in (MO_SCEN[:2] if FAST else MO_SCEN):
        add("MO", name, bs.run_mo(Pd, Fo, Mo, rev_w, **kw), Pd)

    # Vergleichsreihen
    bench = {}
    if "SPY" in raw:
        try:
            adj = raw["SPY"]["Adj Close"].dropna()
            adj.index = pd.DatetimeIndex([pd.Timestamp(i).tz_localize(None) if pd.Timestamp(i).tzinfo else pd.Timestamp(i) for i in adj.index])
            bench["S&P 500 (SPY, mit Dividenden)"] = adj.resample("W-FRI").last().dropna()
            cl = raw["SPY"]["Close"].dropna()
            cl.index = pd.DatetimeIndex([pd.Timestamp(i).tz_localize(None) if pd.Timestamp(i).tzinfo else pd.Timestamp(i) for i in cl.index])
            bench["S&P 500 (SPY, nur Kurs)"] = cl.resample("W-FRI").last().dropna()
        except Exception as e:
            fd.log("SPY nicht auswertbar: %s" % e)
    r = Pd["C"][:, 1:] / Pd["C"][:, :-1] - 1
    r = np.where(np.isfinite(r) & (np.abs(r) < 1.0), r, np.nan)
    ew = np.concatenate(([1.0], np.cumprod(1 + np.nan_to_num(np.nanmean(r, axis=0)))))
    bench["Alle Aktien gleichgewichtet (Kauf und Halten)"] = weekly_curve(ew, dates)

    # gemeinsame Wochenreihe ab STAT_START
    last = min(min(x["eq"].index[-1] for x in rows), min(s.index[-1] for s in bench.values()))
    cal = pd.date_range(STAT_START, last, freq="W-FRI")
    allc = {("%s|%s" % (x["group"], x["name"])): x["eq"] for x in rows}
    allc.update({"BM|" + k: v for k, v in bench.items()})
    A = align(allc, cal)
    out_rows = []
    for x in rows:
        y = A["%s|%s" % (x["group"], x["name"])]
        out_rows.append(dict(group=x["group"], name=x["name"], stats=stats_on(cal, y), yearly=yearly_on(cal, y), trades=x["trades"],
                             npos=x["npos"], gross=x["gross"], best=x["best"]))
    out_b = []
    for k in bench:
        y = A["BM|" + k]
        out_b.append(dict(name=k, stats=stats_on(cal, y), yearly=yearly_on(cal, y)))
    # Kurven für die Hauptvarianten
    keep = {"TU": rows[0]["name"], "TUW": [x for x in rows if x["group"] == "TUW"][0]["name"],
            "MN": [x for x in rows if x["group"] == "MN"][0]["name"], "MO": [x for x in rows if x["group"] == "MO"][0]["name"]}
    curves = {g: [round(float(v), 4) for v in A["%s|%s" % (g, n)]] for g, n in keep.items()}
    for k in bench:
        curves[k] = [round(float(v), 4) for v in A["BM|" + k]]
    lines = ["Zeitraum %s bis %s, %d Aktien" % (cal[0].date(), cal[-1].date(), len(daily)), "",
             "%-6s %-44s %9s %9s %9s %7s %7s %7s %6s" % ("", "Variante", "Gesamt", "p. a.", "MaxDD", "Sharpe", "Calmar", "Trades", "Gewinn")]
    for x in out_rows:
        s, t = x["stats"], x["trades"]
        lines.append("%-6s %-44s %9s %9s %9s %7.2f %7.2f %7d %6s" % (x["group"], x["name"], fmt(s.get("total"), 0), fmt(s.get("cagr")), fmt(s.get("maxdd"), 0),
                     s.get("sharpe", float("nan")), s.get("calmar", float("nan")), t.get("n", 0), fmt(t.get("win", float("nan")), 0)))
    for x in out_b:
        s = x["stats"]
        lines.append("%-6s %-44s %9s %9s %9s %7.2f %7.2f" % ("", x["name"], fmt(s.get("total"), 0), fmt(s.get("cagr")), fmt(s.get("maxdd"), 0), s.get("sharpe", float("nan")), s.get("calmar", float("nan"))))
    lines += ["", "Kalenderjahre der vier Standardläufe und SPY (mit Dividenden)"]
    ys = sorted({y for x in out_rows[:1] for y in x["yearly"]})
    std = {g: [x for x in out_rows if x["group"] == g][0] for g in ("TU", "TUW", "MN", "MO")}
    spy = [x for x in out_b if "Dividenden" in x["name"]]
    for y in ys:
        lines.append("%d  %s" % (y, "  ".join("%s %8s" % (g, fmt(std[g]["yearly"].get(y))) for g in std) + ("  SPY %8s" % fmt(spy[0]["yearly"].get(y)) if spy else "")))
    for g in std:
        t = std[g]["trades"]
        lines.append("%s: Trades %s, Gewinner %s, Ø Gewinn %s, Ø Verlust %s, Haltedauer %.0f Tage, Top5-Anteil %s, Positionen im Mittel %s, beste: %s" % (
            g, t.get("n"), fmt(t.get("win"), 0), fmt(t.get("avg_win"), 0), fmt(t.get("avg_loss"), 0), t.get("hold_days", float("nan")), fmt(t.get("top5_share"), 0),
            ("%.1f" % std[g]["npos"]) if std[g]["npos"] is not None else "–", "; ".join("%s %+.0f%% (%s)" % (b[1], b[0] * 100, b[2]) for b in std[g]["best"][:3])))
    text = "\n".join(lines)
    print(text)
    out = dict(meta=dict(von=str(cal[0].date()), bis=str(cal[-1].date()), aktien=len(daily), erstellt=pd.Timestamp.now().isoformat()),
               rows=out_rows, bench=out_b, dates=[str(x.date()) for x in cal], curves=curves, text=text)
    blob = json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o))
    with open(os.path.join(OUT_DIR, "result_all.json"), "w", encoding="utf-8") as f:
        f.write(blob)
    print("RESULT_JSON_START %d" % len(blob))
    for k in range(0, len(blob), 800):
        print("BTJ %05d %s" % (k // 800, blob[k:k + 800]))
    print("RESULT_JSON_END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
