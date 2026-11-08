#!/usr/bin/env python3
"""Backtest der Turtle-Strategie auf Wochenkerzen (TUW-Screener) mit den Regeln und Standardwerten der App.

Ablauf je Woche (die Wochenkerze reicht von Montag bis Freitag, Entscheidungen fallen mit den Daten bis zur Vorwoche):
  1. Ausstiege: Stop (2 N unter dem letzten Kaufkurs) oder Ausstiegskanal (10 Wochen bei System 1, 20 bei System 2),
     je nachdem, was näher am Kurs liegt. Lücke durch die Marke: Verkauf zur Eröffnung der Woche.
  2. Zukäufe: alle ½ N über dem letzten Kaufkurs, bis 4 Einheiten je Aktie, nur wenn Limits und Depot es zulassen.
  3. Neue Einstiege: Bruch des 20-Wochen-Hochs (System 1, mit Skip-Regel) oder des 55-Wochen-Hochs (System 2).
     Bei mehreren Kandidaten in einer Woche entscheidet die Stärke der letzten 26 Wochen.
  Am Einstiegstag (hier: in der Einstiegswoche) wird nicht auf den Stop geprüft, wie im Rechenkern der App.
Stückzahl: Risiko % vom Depot ÷ N, begrenzt auf 25 % vom Depot je Aktie und 100 % Investitionsgrad (kein Hebel).
Kosten: Aufschlag je Kauf und Verkauf (Standard 0,10 % vom Umsatz). Keine Dividenden, keine Zinsen auf Cash.

Achtung Überlebensverzerrung: Das Universum ist die heutige Zusammensetzung von S&P 500 und Nasdaq-100. Aktien, die
früher im Index waren und ausgeschieden sind, fehlen. Das beschönigt das Ergebnis.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import turtle_core as tc

OUT_DIR = os.environ.get("BT_OUT", "backtest_out")
YEARS = int(os.environ.get("BT_YEARS", "14"))
WARM = 57          # Wochen, bis die 55-Wochen-Marke vorliegt

DEFAULTS = dict(equity=100000.0, risk=0.25, stop=2.0, add=0.5, max_units=4, max_sec=6, max_dir=12, cap_pos=25.0,
                max_inv=100.0, s1=True, s2=True, skip=True, short=False, min_price=10.0, min_dv=20e6, cost=0.001,
                mom_weeks=26)


# ----------------------------------------------------------------- Datenaufbereitung
def build_panel(weekly, sectors=None, vols=None):
    """weekly: {Ticker: Wochen-DataFrame (Open, High, Low, Close, Volume, Index = Freitag)}.
    Rückgabe: Panel mit Arrays [Aktie, Woche]; Lücken in der Reihe einer Aktie werden mit dem Vorschluss gefüllt,
    vor dem ersten und nach dem letzten Kurs steht NaN."""
    dates = sorted(set().union(*[set(d.index) for d in weekly.values()]))
    dates = pd.DatetimeIndex(dates)
    tick = sorted(weekly)
    S, T = len(tick), len(dates)
    A = {k: np.full((S, T), np.nan) for k in ("O", "H", "L", "C", "V")}
    first = np.zeros(S, int)
    last = np.zeros(S, int)
    for s, t in enumerate(tick):
        d = weekly[t].reindex(dates)
        ok = np.where(d["Close"].notna().to_numpy())[0]
        a, b = ok[0], ok[-1]
        first[s], last[s] = a, b
        c = d["Close"].ffill()
        for k, col in (("O", "Open"), ("H", "High"), ("L", "Low")):
            x = d[col].copy()
            x = x.where(x.notna(), c.shift(1).where(d["Close"].isna(), x))   # Lücke: Vorschluss
            x = x.where(x.notna(), c)
            A[k][s, a:b + 1] = x.to_numpy()[a:b + 1]
        A["C"][s, a:b + 1] = c.to_numpy()[a:b + 1]
        v = d["Volume"].fillna(0.0).to_numpy()
        A["V"][s, a:b + 1] = v[a:b + 1]
    return dict(dates=dates, tick=tick, sec=[(sectors or {}).get(t, "") for t in tick], first=first, last=last, **A)


def indicators(P, p):
    """Rechnet je Aktie N, die Marken der Vorwochen, Momentum, Liquidität und die gehandelten System-1-Ausbrüche."""
    S, T = len(P["tick"]), len(P["dates"])
    I = {k: np.full((S, T), np.nan) for k in ("N", "hh20", "hh55", "ll20", "ll55", "ll10", "hh10", "mom", "dv")}
    s1 = {1: [set() for _ in range(S)], -1: [set() for _ in range(S)]}
    for s in range(S):
        a, b = P["first"][s], P["last"][s]
        if b - a + 1 < tc.S2_IN + 6:
            continue
        o, h, l, c, v = (P[k][s, a:b + 1] for k in ("O", "H", "L", "C", "V"))
        n = tc.wilder_n(tc.true_range(h, l, c))
        sl = slice(a, b + 1)
        I["N"][s, sl] = n
        I["hh20"][s, sl] = tc.prev(tc.roll_max(h, tc.S1_IN))
        I["hh55"][s, sl] = tc.prev(tc.roll_max(h, tc.S2_IN))
        I["ll20"][s, sl] = tc.prev(tc.roll_min(l, tc.S2_OUT))
        I["ll55"][s, sl] = tc.prev(tc.roll_min(l, tc.S2_IN))
        I["mom"][s, sl] = tc.prev(c / np.concatenate((np.full(p["mom_weeks"], np.nan), c[:-p["mom_weeks"]])) - 1)
        I["dv"][s, sl] = tc.prev(pd.Series(v * c).rolling(10).mean().to_numpy() / 5)     # Umsatz je Handelstag, Mittel der letzten 10 Wochen
        for side in (1, -1):
            trades, pos, _ = tc.sim_side(o, h, l, n, tc.S1_IN, tc.S1_OUT, side, p["stop"])
            for t in trades + ([pos] if pos else []):
                if t["taken"] or not p["skip"]:
                    s1[side][s].add(a + t["entry_i"])
    return I, s1


# ----------------------------------------------------------------- Simulation
def run_backtest(P, I, s1, **kw):
    p = dict(DEFAULTS, **kw)
    S, T = len(P["tick"]), len(P["dates"])
    O, H, L, C = P["O"], P["H"], P["L"], P["C"]
    Cf = C.copy()                                           # Schluss, nach Ende der Reihe festgehalten
    for s in range(S):
        b = P["last"][s]
        Cf[s, b + 1:] = C[s, b]
    cash = p["equity"]
    pos, closed = {}, []
    eq = np.full(T, np.nan)
    gross_hist = np.zeros(T)
    npos_hist = np.zeros(T)
    c = p["cost"]
    sm, am = p["stop"], p["add"]
    dirs = (1, -1) if p["short"] else (1,)
    eq[:WARM] = p["equity"]

    def mark(s, i):
        x = Cf[s, i]
        return x if not math.isnan(x) else 0.0

    for i in range(WARM, T):
        eq_prev = cash + sum(P_["d"] * sum(f["q"] for f in P_["fills"]) * mark(s, i - 1) for s, P_ in pos.items())
        gross = sum(sum(f["q"] for f in P_["fills"]) * mark(s, i - 1) for s, P_ in pos.items())
        units_dir = {1: 0, -1: 0}
        units_sec = {}
        for s, P_ in pos.items():
            u = len(P_["fills"])
            units_dir[P_["d"]] += u
            k = (P["sec"][s], P_["d"])
            units_sec[k] = units_sec.get(k, 0) + u

        def close(s, px, i_):
            nonlocal cash, gross
            P_ = pos.pop(s)
            q = sum(f["q"] for f in P_["fills"])
            d = P_["d"]
            cash += d * q * px - q * px * c
            pnl = sum(d * f["q"] * (px - f["px"]) - f["q"] * f["px"] * c for f in P_["fills"]) - q * px * c
            cost_in = sum(f["q"] * f["px"] for f in P_["fills"])
            closed.append(dict(t=P["tick"][s], d=d, sys=P_["sys"], units=len(P_["fills"]), i_in=P_["i_in"], i_out=i_,
                               px_in=cost_in / q, px_out=px, pnl=pnl, ret=pnl / cost_in if cost_in else 0.0,
                               risk=sum(f["q"] * sm * f["N"] for f in P_["fills"])))
            gross -= q * mark(s, i - 1)
            units_dir[d] -= len(P_["fills"])
            k = (P["sec"][s], d)
            units_sec[k] = units_sec.get(k, 0) - len(P_["fills"])

        # 1. Ausstiege
        out_now = set()
        for s in list(pos):
            P_ = pos[s]
            d = P_["d"]
            if i > P["last"][s]:                              # Aktie ausgelistet oder Daten zu Ende: zum letzten Kurs zu
                close(s, C[s, P["last"][s]], i)
                out_now.add(s)
                continue
            lf = P_["fills"][-1]
            stop = lf["px"] - d * sm * lf["N"]
            k = P_["k"]
            if d > 0:
                ch = np.nanmin(L[s, i - k:i])
                lvl = stop if math.isnan(ch) else max(stop, ch)
                hit, px = L[s, i] <= lvl, min(O[s, i], lvl)
            else:
                ch = np.nanmax(H[s, i - k:i])
                lvl = stop if math.isnan(ch) else min(stop, ch)
                hit, px = H[s, i] >= lvl, max(O[s, i], lvl)
            if hit:
                close(s, px, i)
                out_now.add(s)

        def room_ok(s, d, price, N, q_units=1):
            if units_dir[d] + 1 > p["max_dir"]:
                return 0
            if P["sec"][s] and units_sec.get((P["sec"][s], d), 0) + 1 > p["max_sec"]:
                return 0
            q = math.floor(p["risk"] / 100 * eq_prev / N) if N > 0 else 0
            if q <= 0:
                return 0
            if p["cap_pos"] > 0:
                have = sum(f["q"] for f in pos[s]["fills"]) * price if s in pos else 0.0
                room = p["cap_pos"] / 100 * eq_prev - have
                q = min(q, math.floor(max(0.0, room) / price))
            if p["max_inv"] > 0:
                room = p["max_inv"] / 100 * eq_prev - gross
                q = min(q, math.floor(max(0.0, room) / price))
            return max(0, q)

        def fill(s, d, sys_, px, N, q, k_out):
            nonlocal cash, gross
            cash -= d * q * px + q * px * c
            gross += q * px
            if s in pos:
                pos[s]["fills"].append(dict(px=px, q=q, N=N, i=i))
            else:
                pos[s] = dict(d=d, sys=sys_, k=k_out, i_in=i, fills=[dict(px=px, q=q, N=N, i=i)])
            units_dir[d] += 1
            k_ = (P["sec"][s], d)
            units_sec[k_] = units_sec.get(k_, 0) + 1

        # 2. Zukäufe
        for s in list(pos):
            P_ = pos[s]
            d = P_["d"]
            if s in out_now or math.isnan(H[s, i]):
                continue
            while len(P_["fills"]) < p["max_units"]:
                lf = P_["fills"][-1]
                lvl = lf["px"] + d * am * lf["N"]
                if (H[s, i] >= lvl) if d > 0 else (L[s, i] <= lvl):
                    px = max(O[s, i], lvl) if d > 0 else min(O[s, i], lvl)
                    N = I["N"][s, i - 1]
                    q = room_ok(s, d, px, N) if not math.isnan(N) else 0
                    if q <= 0:
                        break
                    fill(s, d, P_["sys"], px, N, q, P_["k"])
                else:
                    break

        # 3. Neue Einstiege
        cands = []
        for s in range(S):
            if s in pos or s in out_now or i > P["last"][s] or i <= P["first"][s] or math.isnan(H[s, i]):
                continue
            N = I["N"][s, i - 1]
            if math.isnan(N) or N <= 0 or math.isnan(C[s, i - 1]) or C[s, i - 1] < p["min_price"]:
                continue
            if not (I["dv"][s, i] >= p["min_dv"]):
                continue
            for d in dirs:
                sys_, lvl = 0, None
                if p["s1"] and i in s1[d][s]:
                    sys_ = 1
                    lvl = I["hh20"][s, i] if d > 0 else I["ll20"][s, i]
                if not sys_ and p["s2"]:
                    lvl = I["hh55"][s, i] if d > 0 else I["ll55"][s, i]
                    if not math.isnan(lvl) and ((H[s, i] > lvl) if d > 0 else (L[s, i] < lvl)):
                        sys_ = 2
                if sys_ and lvl is not None and not math.isnan(lvl):
                    m = I["mom"][s, i]
                    cands.append((d, s, sys_, lvl, N, 0.0 if math.isnan(m) else m))
        cands.sort(key=lambda x: -x[0] * x[5])
        for d, s, sys_, lvl, N, m in cands:
            px = max(O[s, i], lvl) if d > 0 else min(O[s, i], lvl)
            q = room_ok(s, d, px, N)
            if q > 0:
                fill(s, d, sys_, px, N, q, tc.S1_OUT if sys_ == 1 else tc.S2_OUT)

        eq[i] = cash + sum(P_["d"] * sum(f["q"] for f in P_["fills"]) * mark(s, i) for s, P_ in pos.items())
        gross_hist[i] = sum(sum(f["q"] for f in P_["fills"]) * mark(s, i) for s, P_ in pos.items()) / eq[i] if eq[i] > 0 else 0
        npos_hist[i] = len(pos)

    open_pos = {P["tick"][s]: dict(d=P_["d"], units=len(P_["fills"]), q=sum(f["q"] for f in P_["fills"]),
                                   basis=sum(f["q"] * f["px"] for f in P_["fills"]), s=int(s)) for s, P_ in pos.items()}
    return dict(eq=eq, closed=closed, gross=gross_hist, npos=npos_hist, open=open_pos, params=p)


# ----------------------------------------------------------------- Kennzahlen
def curve_stats(eq, dates, start=WARM - 1):
    e = np.asarray(eq[start:], float)
    d = dates[start:]
    if len(e) < 10 or not np.isfinite(e).all():
        return {}
    yrs = (d[-1] - d[0]).days / 365.25
    r = e[1:] / e[:-1] - 1
    dd = e / np.maximum.accumulate(e) - 1
    cagr = (e[-1] / e[0]) ** (1 / yrs) - 1 if yrs > 0 and e[0] > 0 and e[-1] > 0 else float("nan")
    vol = r.std(ddof=1) * math.sqrt(52)
    neg = r[r < 0]
    return dict(total=e[-1] / e[0] - 1, cagr=cagr, vol=vol, maxdd=dd.min(), sharpe=(r.mean() * 52) / vol if vol else float("nan"),
                sortino=(r.mean() * 52) / (neg.std(ddof=1) * math.sqrt(52)) if len(neg) > 2 else float("nan"),
                calmar=cagr / abs(dd.min()) if dd.min() < 0 else float("nan"), years=yrs,
                best_week=r.max(), worst_week=r.min())


def trade_stats(closed):
    if not closed:
        return dict(n=0)
    pnl = np.array([t["pnl"] for t in closed])
    ret = np.array([t["ret"] for t in closed])
    w, l = pnl[pnl > 0], pnl[pnl <= 0]
    hold = np.array([t["i_out"] - t["i_in"] for t in closed])
    rr = np.array([t["pnl"] / t["risk"] for t in closed if t["risk"] > 0])
    return dict(n=len(closed), win=float((pnl > 0).mean()), avg_win=float(ret[pnl > 0].mean()) if len(w) else 0.0,
                avg_loss=float(ret[pnl <= 0].mean()) if len(l) else 0.0,
                pf=float(w.sum() / -l.sum()) if len(l) and l.sum() < 0 else float("nan"),
                hold=float(hold.mean()), r_mean=float(rr.mean()) if len(rr) else float("nan"),
                top5_share=float(np.sort(pnl)[::-1][:5].sum() / pnl.sum()) if pnl.sum() > 0 else float("nan"),
                units=float(np.mean([t["units"] for t in closed])),
                s1=int(sum(1 for t in closed if t["sys"] == 1)), s2=int(sum(1 for t in closed if t["sys"] == 2)),
                long=int(sum(1 for t in closed if t["d"] > 0)), short=int(sum(1 for t in closed if t["d"] < 0)))


def yearly(eq, dates, start=WARM - 1):
    s = pd.Series(np.asarray(eq, float)[start:], index=dates[start:])
    y = s.groupby(s.index.year).last()
    first = s.iloc[0]
    out, prev = {}, first
    for yr, v in y.items():
        out[int(yr)] = float(v / prev - 1)
        prev = v
    return out


def benchmark_curve(close, dates, start=WARM - 1):
    s = close.reindex(dates).ffill()
    s = s / s.iloc[start]
    return s.to_numpy()


def equal_weight_curve(P):
    C = P["C"]
    r = C[:, 1:] / C[:, :-1] - 1
    r = np.where(np.isfinite(r) & (np.abs(r) < 1.0), r, np.nan)      # Ausreißer (Datenfehler) ignorieren
    m = np.nanmean(r, axis=0)
    return np.concatenate(([1.0], np.cumprod(1 + np.nan_to_num(m))))


def summarize(name, res, P, start=WARM - 1):
    e = res["eq"]
    d = P["dates"]
    return dict(name=name, stats=curve_stats(e, d, start), trades=trade_stats(res["closed"]), yearly=yearly(e, d, start),
                avg_gross=float(np.mean(res["gross"][WARM:])), avg_pos=float(np.mean(res["npos"][WARM:])),
                open=res["open"], params={k: v for k, v in res["params"].items()})


SCENARIOS = [
    ("Standard (nur Long, Risiko 0,25 %)", {}),
    ("Long und Short", dict(short=True)),
    ("Nur System 1 (20/10 Wochen)", dict(s2=False)),
    ("Nur System 2 (55/20 Wochen)", dict(s1=False)),
    ("System 1 ohne Skip-Regel", dict(skip=False)),
    ("Risiko 0,5 % je Einheit", dict(risk=0.5)),
    ("Risiko 1,0 % je Einheit", dict(risk=1.0, cap_pos=50.0)),
    ("Ohne Handelskosten", dict(cost=0.0)),
    ("Kosten 0,3 % je Seite", dict(cost=0.003)),
]


def fmt_pct(x, d=1):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else ("%.*f %%" % (d, x * 100)).replace(".", ",")


def report_text(rows, bench, P):
    lines = ["Zeitraum %s bis %s (%d Wochen), %d Aktien im Universum" % (P["dates"][WARM - 1].date(), P["dates"][-1].date(), len(P["dates"]) - WARM + 1, len(P["tick"])), ""]
    lines.append("%-40s %9s %9s %9s %7s %7s %7s %6s %6s" % ("Variante", "Gesamt", "p. a.", "MaxDD", "Sharpe", "Calmar", "Trades", "Gewinn", "Invest"))
    for r in rows:
        s, t = r["stats"], r["trades"]
        lines.append("%-40s %9s %9s %9s %7.2f %7.2f %7d %6s %6s" % (r["name"], fmt_pct(s.get("total")), fmt_pct(s.get("cagr")), fmt_pct(s.get("maxdd")),
                     s.get("sharpe", float("nan")), s.get("calmar", float("nan")), t.get("n", 0), fmt_pct(t.get("win", float("nan")), 0), fmt_pct(r["avg_gross"], 0)))
    for k, b in bench.items():
        s = b["stats"]
        lines.append("%-40s %9s %9s %9s %7.2f %7.2f" % (k, fmt_pct(s.get("total")), fmt_pct(s.get("cagr")), fmt_pct(s.get("maxdd")), s.get("sharpe", float("nan")), s.get("calmar", float("nan"))))
    return "\n".join(lines)


def main():
    import fetch_data as fd
    os.makedirs(OUT_DIR, exist_ok=True)
    ix_map, status, names, sectors = fd.build_universe()
    sectors = dict(fd.load_sector_file(), **sectors)
    tickers = sorted(ix_map)
    fd.log("%d Ticker, lade %d Jahre Kurse" % (len(tickers), YEARS))
    start = (pd.Timestamp.now() - pd.DateOffset(years=YEARS + 2)).strftime("%Y-%m-%d")      # 2 Jahre Vorlauf für die 55-Wochen-Marke
    raw, failed = fd.download(tickers + ["SPY"], start=start)
    drop_after = fd.last_complete_day()
    weekly = {}
    for t, df in raw.items():
        d = fd.clean_frame(df)
        if d is None or len(d) < 300:
            continue
        w = tc.to_weekly(d[d.index <= drop_after])
        w = w[w.index <= drop_after]
        if len(w) >= 80:
            weekly[t] = w
    spy = weekly.pop("SPY", None)
    spy_tr = None
    if "SPY" in raw:
        a = raw["SPY"]
        try:
            adj = a["Adj Close"].dropna()
            adj.index = pd.DatetimeIndex([pd.Timestamp(x).tz_localize(None) if pd.Timestamp(x).tzinfo else pd.Timestamp(x) for x in adj.index])
            spy_tr = adj.resample("W-FRI").last().dropna()
        except Exception:
            spy_tr = None
    fd.log("%d Aktien mit genug Wochenkerzen, %d ohne Daten" % (len(weekly), len(failed)))
    P = build_panel(weekly, sectors)
    I, s1 = indicators(P, DEFAULTS)
    rows, curves = [], {}
    for name, kw in SCENARIOS:
        p = dict(DEFAULTS, **kw)
        I2, s12 = (indicators(P, p) if p["skip"] != DEFAULTS["skip"] else (I, s1))
        res = run_backtest(P, I2, s12, **kw)
        rows.append(summarize(name, res, P))
        curves[name] = res["eq"] / res["eq"][WARM - 1]
        fd.log("  %-40s %s" % (name, fmt_pct(rows[-1]["stats"].get("cagr"))))
        if name.startswith("Standard"):
            std_res = res
    bench = {}
    if spy_tr is not None:
        bc = benchmark_curve(spy_tr, P["dates"])
        bench["S&P 500 (SPY, mit Dividenden)"] = dict(stats=curve_stats(bc, P["dates"]), yearly=yearly(bc, P["dates"]))
        curves["S&P 500 (SPY, mit Dividenden)"] = bc
    if spy is not None:
        bc = benchmark_curve(spy["Close"], P["dates"])
        bench["S&P 500 (SPY, nur Kurs)"] = dict(stats=curve_stats(bc, P["dates"]), yearly=yearly(bc, P["dates"]))
        curves["S&P 500 (SPY, nur Kurs)"] = bc
    ew = equal_weight_curve(P)
    ew = ew / ew[WARM - 1]
    bench["Gleichgewichtetes Universum (Kauf und Halten, wöchentlich ausgeglichen)"] = dict(stats=curve_stats(ew, P["dates"]), yearly=yearly(ew, P["dates"]))
    curves["Gleichgewichtetes Universum"] = ew
    # beste und schlechteste Trades des Standardlaufs
    best = sorted(std_res["closed"], key=lambda t: -t["pnl"])[:10]
    worst = sorted(std_res["closed"], key=lambda t: t["pnl"])[:10]
    pack = lambda t: dict(t=t["t"], sys=t["sys"], von=str(P["dates"][t["i_in"]].date()), bis=str(P["dates"][t["i_out"]].date()), units=t["units"], ret=round(t["ret"], 3), pnl=round(t["pnl"]))
    text = report_text(rows, bench, P)
    print(text)
    out = dict(meta=dict(erstellt=pd.Timestamp.now().isoformat(), von=str(P["dates"][WARM - 1].date()), bis=str(P["dates"][-1].date()),
                         aktien=len(P["tick"]), fehlend=len(failed), kurs_von_spy=bool(spy is not None)),
               rows=rows, bench=bench, best=[pack(t) for t in best], worst=[pack(t) for t in worst],
               dates=[str(x.date()) for x in P["dates"][WARM - 1:]],
               curves={k: [round(float(x), 4) for x in v[WARM - 1:]] for k, v in curves.items()},
               exposure=[round(float(x), 3) for x in std_res["gross"][WARM - 1:]], text=text)
    with open(os.path.join(OUT_DIR, "result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o))
    print("RESULT_JSON_START")
    print(json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=str))
    print("RESULT_JSON_END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
