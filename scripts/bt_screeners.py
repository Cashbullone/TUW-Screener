"""Backtest-Simulationen für MN-Screener (Minervini) und MO-Screener (Momentum) auf Tageskerzen.

Beide arbeiten mit einer Überprüfung pro Woche. Die Entscheidung fällt mit den Schlusskursen des Prüftags, gehandelt wird zur
Eröffnung des nächsten Handelstags. Kosten je Kauf und Verkauf als Anteil vom Umsatz. Keine Dividenden, keine Zinsen.

MN: Prüftag ist der letzte Handelstag der Woche (Freitag), Orderzettel für Montag. Regeln wie im MN-Screener:
    Trend Template (7 Kursbedingungen) plus RS-Rating >= 80, Template höchstens 5 Handelstage alt, Stärke-Score, Stop 7,5 %
    unter dem Einstieg (Break-even ab +2R), Verkaufsstufen (25:10, 35:20, 45:30, 60:40, 100:100), Verkauf bei verletztem
    Template, Tausch schwacher gegen stärkere Aktien, höchstens 10 Positionen, Risiko 1,25 % je Trade, höchstens 25 % je Position.
    Nicht abbildbar: Fundamentaldaten (Yahoo liefert keine Historie).
MO: Prüftag ist der Mittwoch. Regeln wie im MO-Screener: Score aus 12-1-Rendite und Clenow-Wert (Prozentränge), Kauf aus den
    besten 20 %, über der 100-Tage-Linie, ohne Kurslücke über 15 %, Frog in the Pan, Marktfilter und Volatilitätsbremse,
    10 Positionen, 15 % je Position, 3 je Sektor, Größe nach ATR, Verkauf bei Linie, Rang, Lücke oder 9 % Verlust.
"""
import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- gemeinsame Hilfen
def review_days(dates, weekday=None):
    """Indizes der Prüftage: letzter Handelstag jeder Woche (weekday=None) oder der Handelstag mit diesem Wochentag
    (0 = Montag), ersatzweise der letzte frühere Handelstag derselben Woche."""
    wk = dates.to_period("W-SUN")
    out = []
    for w in pd.unique(wk):
        idx = np.where(wk == w)[0]
        if weekday is None:
            out.append(int(idx[-1]))
        else:
            ok = [i for i in idx if dates[i].weekday() <= weekday]
            if ok and dates[ok[-1]].weekday() == weekday:
                out.append(int(ok[-1]))
    return out


def frames(P):
    def df(k):
        return pd.DataFrame(P[k].T, index=P["dates"])
    return df


def rank_low(x):
    """Anzahl der Werte, die strikt kleiner sind (für Prozentränge)."""
    return pd.Series(x).rank(method="min").to_numpy() - 1


class Book:
    """Konto: Cash, Positionen, Handelsliste. Alle Preise in USD, Kosten als Anteil vom Umsatz."""

    def __init__(self, equity, cost):
        self.cash, self.cost, self.pos, self.closed = equity, cost, {}, []

    def buy(self, s, q, px, i):
        if q <= 0:
            return 0
        paid = q * px * (1 + self.cost)
        p = self.pos.get(s)
        if p is None:
            p = self.pos[s] = dict(q=0, cost=0.0, i_in=i, paid=0.0, pnl=0.0, q0=0, sold=0, peak=px, en=px)
        p["q"] += q
        p["q0"] += q
        p["cost"] += paid
        p["paid"] += paid
        p["en"] = p["cost"] / p["q"]
        self.cash -= paid
        return q

    def sell(self, s, n, px, i, tick=None):
        p = self.pos.get(s)
        if p is None or n <= 0:
            return
        n = min(n, p["q"])
        avg = p["cost"] / p["q"]
        proceeds = n * px * (1 - self.cost)
        self.cash += proceeds
        p["pnl"] += proceeds - n * avg
        p["cost"] -= n * avg
        p["q"] -= n
        p["sold"] += n
        if p["q"] <= 0:
            self.closed.append(dict(t=tick if tick is not None else s, d=1, sys=0, units=1, i_in=p["i_in"], i_out=i,
                                    pnl=p["pnl"], ret=p["pnl"] / p["paid"] if p["paid"] else 0.0, risk=0.0))
            del self.pos[s]

    def equity(self, Cf, i):
        return self.cash + sum(p["q"] * Cf[s, i] for s, p in self.pos.items())


def ffilled_close(P):
    Cf = P["C"].copy()
    for s in range(len(P["tick"])):
        Cf[s, P["last"][s] + 1:] = P["C"][s, P["last"][s]]
    return Cf


def liquid(P, minp, mindv):
    """Mindestkurs und Mindest-Tagesumsatz (Mittel der letzten 50 Tage), Arrays [Aktie, Tag]."""
    dv = pd.DataFrame((P["V"] * P["C"]).T).rolling(50, min_periods=40).mean().to_numpy().T
    return (P["C"] >= minp) & (dv >= mindv)


# ---------------------------------------------------------------- MO
MO_DEF = dict(equity=100000.0, cost=0.001, min_price=10.0, min_dv=20e6, max_pos=10, risk=0.25, cap=15.0, sec_max=3,
              buy_top=20.0, sell_top=30.0, fip_cut=50.0, stop=9.0, gap=0.15, market=True, brake=True, fip=True, wd=2,
              brake_dd=10.0, brake_vix=30.0, lev=1.0, borrow=0.05)


def mo_features(P, rev):
    C = pd.DataFrame(P["C"].T, index=P["dates"])
    H = pd.DataFrame(P["H"].T, index=P["dates"])
    L = pd.DataFrame(P["L"].T, index=P["dates"])
    S, T = len(P["tick"]), len(P["dates"])
    F = {}
    take = lambda df: df.to_numpy()[rev]
    F["sma100"] = take(C.rolling(100, min_periods=100).mean())
    F["m"] = take(C.shift(21) / C.shift(252) - 1)
    r = C.pct_change(fill_method=None)
    ok = r.notna().astype(float)
    up = (r > 0).astype(float).where(r.notna())
    dn = (r < 0).astype(float).where(r.notna())
    Tn = ok.rolling(231, min_periods=200).sum().shift(21)
    F["fip"] = take(np.sign(C.shift(21) / C.shift(252) - 1) * (dn.rolling(231, min_periods=200).sum().shift(21) / Tn - up.rolling(231, min_periods=200).sum().shift(21) / Tn))
    F["gap"] = take(r.abs().rolling(90, min_periods=60).max())
    pc = C.shift(1)
    tr = pd.DataFrame(np.fmax(np.fmax((H - L).to_numpy(), (H - pc).abs().to_numpy()), (L - pc).abs().to_numpy()), index=P["dates"])
    F["atr"] = take(tr.rolling(20, min_periods=17).mean())
    # Clenow: Steigung der logarithmierten Schlusskurse über 90 Tage, auf 250 Tage hochgerechnet, mal R2
    Y = np.log(P["C"].T)                                   # [Tag, Aktie]
    val = np.isfinite(Y)
    Y0 = np.where(val, Y, 0.0)
    k = np.arange(T, dtype=float)[:, None]
    z = np.zeros((1, S))
    cs1 = np.vstack([z, np.cumsum(Y0, 0)])
    cs2 = np.vstack([z, np.cumsum(k * Y0, 0)])
    csy = np.vstack([z, np.cumsum(Y0 * Y0, 0)])
    csv = np.vstack([z, np.cumsum(val.astype(float), 0)])
    N = 90
    x = np.arange(N, dtype=float)
    sx, sxx = x.sum(), (x * x).sum()
    cl = np.full((len(rev), S), np.nan)
    for a, i in enumerate(rev):
        if i < N:
            continue
        lo = i - N + 1
        sy = cs1[i + 1] - cs1[lo]
        sxy = (cs2[i + 1] - cs2[lo]) - lo * sy
        syy = csy[i + 1] - csy[lo]
        full = (csv[i + 1] - csv[lo]) == N
        dx = N * sxx - sx * sx
        dy = N * syy - sy * sy
        num = N * sxy - sx * sy
        b = num / dx
        r2 = np.where(dy > 1e-12, num * num / (dx * np.where(dy > 1e-12, dy, 1)), 0.0)
        c_ = (np.exp(b * 250) - 1) * 100 * r2
        cl[a] = np.where(full, c_, np.nan)
    F["cl"] = cl
    return F


def market_state(P, spx, vix, rev, p):
    """Marktfilter je Prüftag: sperrt Neukäufe (S&P 500 unter 200-Tage-Linie oder unter dem 10-Monats-Durchschnitt)
    und Investitionsgrad nach Volatilitätsbremse und 10-Monats-Linie."""
    d = P["dates"]
    c = pd.Series(spx, index=d).ffill()
    v = pd.Series(vix, index=d).ffill()
    sma200 = c.rolling(200, min_periods=200).mean()
    hi = c.rolling(252, min_periods=1).max()
    dd = (1 - c / hi) * 100
    brake = ((dd > p["brake_dd"]) & (v > p["brake_vix"])).to_numpy()
    month = d.year * 12 + d.month
    last_of_month = np.append(month[1:] != month[:-1], True)
    out = []
    for i in rev:
        above = bool(c.iloc[i] > sma200.iloc[i]) if not math.isnan(sma200.iloc[i]) else True
        idx = np.where(last_of_month[:i + 1])[0][-10:]      # abgeschlossene Monate; der laufende zählt erst am Monatsende
        below = None
        if len(idx) == 10:
            below = bool(c.iloc[idx[-1]] < c.iloc[idx].mean())
        f = 1.0
        if brake[i]:
            f = 0.5
        else:
            lo = max(0, i - 40)
            hits = np.where(brake[lo:i + 1])[0]
            if len(hits):
                j = lo + hits[-1]
                days = (d[i] - d[j]).days
                if days <= 28:
                    w = days // 7
                    if w < 3:
                        f = 0.5 + 0.5 * w / 3
        level = round((0.5 if below else 1.0) * f * 100)
        out.append(dict(block=(not above) or bool(below), below=bool(below), f=f, level=level))
    return out


def run_mo(P, F, M, rev, **kw):
    p = dict(MO_DEF, **kw)
    S, T = len(P["tick"]), len(P["dates"])
    O, C = P["O"], P["C"]
    Cf = ffilled_close(P)
    book = Book(p["equity"], p["cost"])
    elig = liquid(P, p["min_price"], p["min_dv"])
    eq = np.full(T, np.nan)
    nrev = {i: a for a, i in enumerate(rev)}
    pending = None
    npos = np.zeros(T)
    start = 260
    eq[:start] = p["equity"]
    for i in range(start, T):
        if book.cash < 0:
            book.cash *= 1 + p["borrow"] / 252
        # Orders von gestern zur Eröffnung
        for s in [s for s in book.pos if i > P["last"][s]]:
            book.sell(s, book.pos[s]["q"], C[s, P["last"][s]], i, P["tick"][s])
        if pending:
            sells, buys = pending
            for s in sells:
                if s in book.pos and not math.isnan(O[s, i]):
                    book.sell(s, book.pos[s]["q"], O[s, i], i, P["tick"][s])
            for s, q in buys:
                if s in book.pos or math.isnan(O[s, i]):
                    continue
                q = min(q, math.floor((book.cash + (p["lev"] - 1) * eq[i - 1]) / (O[s, i] * (1 + p["cost"]))))
                book.buy(s, q, O[s, i], i)
            pending = None
        eq[i] = book.equity(Cf, i)
        npos[i] = len(book.pos)
        a = nrev.get(i)
        if a is None or i >= T - 1:
            continue
        # --- Prüftag: Rangliste
        el = elig[:, i] & np.isfinite(F["m"][a]) & np.isfinite(F["cl"][a]) & np.isfinite(F["fip"][a]) & (i <= P["last"])
        idx = np.where(el)[0]
        if len(idx) < 50:
            continue
        m, cl = F["m"][a][idx], F["cl"][a][idx]
        n = len(idx)
        pm = 1 + 98 * rank_low(m) / (n - 1)
        pc = 1 + 98 * rank_low(cl) / (n - 1)
        sc = (pm + pc) / 2
        order = np.lexsort((np.array([P["tick"][s] for s in idx]), -cl, -sc))
        top = np.empty(n)
        top[order] = (np.arange(n) + 1) / n * 100
        TOP = dict(zip(idx.tolist(), top.tolist()))
        FIP = dict(zip(idx.tolist(), F["fip"][a][idx].tolist()))
        pool = sorted(FIP[s] for s in TOP if TOP[s] <= p["buy_top"])
        cut = pool[max(0, math.ceil(len(pool) * (100 - p["fip_cut"]) / 100) - 1)] if (pool and p["fip"] and p["fip_cut"] > 0) else math.inf
        eqv = eq[i]
        sells = []
        for s, ps in book.pos.items():
            px = Cf[s, i]
            sma = F["sma100"][a][s]
            bad = (not math.isnan(sma) and px < sma) or (s in TOP and TOP[s] > p["sell_top"]) \
                or (F["gap"][a][s] > p["gap"]) or (p["stop"] > 0 and px <= ps["en"] * (1 - p["stop"] / 100))
            if bad:
                sells.append(s)
        keep = [s for s in book.pos if s not in sells]
        sec = {}
        for s in keep:
            sec[P["sec"][s]] = sec.get(P["sec"][s], 0) + 1
        mk = M[a]
        invested = sum(book.pos[s]["q"] * Cf[s, i] for s in keep)
        blocked = p["market"] and mk["block"]
        level = round((0.5 if (p["market"] and mk["below"]) else 1.0) * (mk["f"] if p["brake"] else 1.0) * 100)
        room = max(0.0, level / 100 * p["lev"] * eqv - invested)
        slots = p["max_pos"] - len(keep)
        buys = []
        if not blocked:
            for k in order:
                if slots <= 0:
                    break
                s = int(idx[k])
                if s in book.pos or TOP[s] > p["buy_top"]:
                    continue
                px = Cf[s, i]
                sma = F["sma100"][a][s]
                if math.isnan(sma) or px < sma or F["gap"][a][s] > p["gap"] or FIP[s] > cut:
                    continue
                if p["sec_max"] > 0 and P["sec"][s] and sec.get(P["sec"][s], 0) >= p["sec_max"]:
                    continue
                atr = F["atr"][a][s]
                if not atr > 0:
                    continue
                q = max(0, min(math.floor(eqv * p["risk"] / 100 / atr), math.floor(eqv * p["cap"] / 100 / px)))
                if q * px > room:
                    q = math.floor(room / px)
                if q < 1:
                    continue
                buys.append((s, q))
                slots -= 1
                room -= q * px
                sec[P["sec"][s]] = sec.get(P["sec"][s], 0) + 1
        pending = (sells, buys)
    return dict(eq=eq, closed=book.closed, npos=npos, params=p, open=len(book.pos))


# ---------------------------------------------------------------- MN
MN_DEF = dict(equity=100000.0, cost=0.001, min_price=10.0, min_dv=20e6, risk=1.25, stop=7.5, max_pos_pct=25.0, np=10,
              rs_min=80, fresh=5, sw=15, sk=20.0, sn=1, sh=4, be=True, stages="25:10,35:20,45:30,60:40,100:100",
              template_exit=True, swap=True, lev=1.0, borrow=0.05, start=260, jit=0.0, seed=0)


def mn_features(P, rev):
    C = pd.DataFrame(P["C"].T, index=P["dates"])
    H = pd.DataFrame(P["H"].T, index=P["dates"])
    L = pd.DataFrame(P["L"].T, index=P["dates"])
    s50, s150, s200 = (C.rolling(n, min_periods=n).mean() for n in (50, 150, 200))
    s200p = s200.shift(22)
    hi = H.rolling(252, min_periods=252).max()
    lo = L.rolling(252, min_periods=252).min()
    k1 = (C > s150) & (C > s200)
    k2 = s150 > s200
    k3 = s200 > s200p
    k4 = (s50 > s150) & (s50 > s200)
    k5 = C > s50
    k6 = C >= lo * 1.3
    k7 = C >= hi * 0.75
    ok7 = (k1 & k2 & k3 & k4 & k5 & k6 & k7).fillna(False).to_numpy()
    B = ok7
    Tn = B.shape[0]
    idxs = np.arange(Tn)[:, None]
    lastf = np.maximum.accumulate(np.where(~B, idxs, -1), axis=0)
    age = np.where(B, idxs - lastf - 1, 999)
    rs_raw = (.4 * C / C.shift(63) + .2 * C / C.shift(126) + .2 * C / C.shift(189) + .2 * C / C.shift(252)).to_numpy()
    dist = ((hi - C) / hi * 100).to_numpy()
    sel = lambda a: np.asarray(a)[rev]
    F = dict(ok7=sel(ok7), age=sel(age), rs=sel(rs_raw), dist=sel(dist),
             s50=sel(s50.to_numpy()), a200=None)
    F["k"] = [sel(x.fillna(False).to_numpy()) for x in (k1, k2, k3, k4, k5, k6, k7)]
    # Marktbreite wird nur angezeigt (Ampel ohne Wirkung im Standard) und hier nicht gebraucht
    return F


def stage_list(txt):
    out = []
    for part in txt.split(","):
        a, b = part.split(":")
        out.append((float(a), float(b)))
    return sorted(out)


def run_mn(P, F, rev, **kw):
    p = dict(MN_DEF, **kw)
    S, T = len(P["tick"]), len(P["dates"])
    O, L, C = P["O"], P["L"], P["C"]
    Cf = ffilled_close(P)
    book = Book(p["equity"], p["cost"])
    elig = liquid(P, p["min_price"], p["min_dv"])
    stages = stage_list(p["stages"]) if p["stages"] else []
    eq = np.full(T, np.nan)
    nrev = {i: a for a, i in enumerate(rev)}
    pending = None
    npos = np.zeros(T)
    start = int(p["start"])
    rng = np.random.default_rng(int(p["seed"]))
    eq[:start] = p["equity"]
    sp = p["stop"] / 100
    for i in range(start, T):
        if book.cash < 0:
            book.cash *= 1 + p["borrow"] / 252
        for s in [s for s in book.pos if i > P["last"][s]]:
            book.sell(s, book.pos[s]["q"], C[s, P["last"][s]], i, P["tick"][s])
        if pending:
            sells, parts, swaps, buys = pending
            for s in sells:
                if s in book.pos and not math.isnan(O[s, i]):
                    book.sell(s, book.pos[s]["q"], O[s, i], i, P["tick"][s])
            for s, n in parts:
                if s in book.pos and not math.isnan(O[s, i]):
                    book.sell(s, n, O[s, i], i, P["tick"][s])
            for s_out, s_in, q in swaps:
                if s_out in book.pos and not math.isnan(O[s_out, i]):
                    book.sell(s_out, book.pos[s_out]["q"], O[s_out, i], i, P["tick"][s_out])
            for s, q in [(b[0], b[1]) for b in buys] + [(x[1], x[2]) for x in swaps]:
                if s in book.pos or math.isnan(O[s, i]):
                    continue
                q = min(q, math.floor((book.cash + (p["lev"] - 1) * eq[i - 1]) / (O[s, i] * (1 + p["cost"]))))
                if q >= 1:
                    book.buy(s, q, O[s, i], i)
            pending = None
        # Stop-Order im Markt (nicht am Einstiegstag)
        for s in list(book.pos):
            ps = book.pos[s]
            if ps["i_in"] == i or math.isnan(L[s, i]):
                continue
            en = ps["en"]
            lvl = en * (1 - sp)
            if p["be"] and ps["peak"] >= en * (1 + 2 * sp):
                lvl = max(lvl, en)
            if L[s, i] <= lvl:
                book.sell(s, ps["q"], min(O[s, i], lvl), i, P["tick"][s])
            else:
                ps["peak"] = max(ps["peak"], Cf[s, i])
        eq[i] = book.equity(Cf, i)
        npos[i] = len(book.pos)
        a = nrev.get(i)
        if a is None or i >= T - 1:
            continue
        # --- Prüftag: RS, Template, Score
        raw = F["rs"][a]
        el = elig[:, i] & np.isfinite(raw) & (i <= P["last"])
        idx = np.where(el)[0]
        if len(idx) < 50:
            continue
        sorted_raw = np.sort(raw[idx])
        noise = rng.uniform(-p["jit"], p["jit"], S) if p["jit"] > 0 else np.zeros(S)
        n = len(idx)

        def rs_of(s):
            if not np.isfinite(raw[s]):
                return None
            below = np.searchsorted(sorted_raw, raw[s], side="left")
            return max(1, min(99, round(1 + 98 * below / (n - 1))))

        def score(s):
            r = rs_of(s)
            d = F["dist"][a][s]
            return None if r is None or not np.isfinite(d) else round(.6 * r + .4 * max(0.0, 100 - 4 * d)) + noise[s]

        eqv = eq[i]
        sells, parts, hold = [], [], []
        for s, ps in book.pos.items():
            px = Cf[s, i]
            k_ok = all(bool(F["k"][j][a][s]) for j in range(7))
            g = (px / ps["en"] - 1) * 100
            # Verkaufsstufen: Teilverkäufe aus der Ursprungsmenge, Rest bei 100 %
            rem, hitN = ps["q0"], 0
            full = False
            for lv, pc in stages:
                nn = rem if pc >= 100 else rem * pc / 100
                nn = min(rem, round(nn))
                rem -= nn
                if g >= lv:
                    hitN += nn
                    if pc >= 100:
                        full = True
            todo = max(0, hitN - ps["sold"])
            if (p["template_exit"] and not k_ok) or (full and todo > 0):
                sells.append(s)
            else:
                if todo > 0:
                    parts.append((s, todo))
                hold.append(s)
        part_val = {s: n_ * Cf[s, i] for s, n_ in parts}
        heldAll = sum(book.pos[s]["q"] * Cf[s, i] for s in book.pos) - sum(book.pos[s]["q"] * Cf[s, i] for s in sells) - sum(part_val.values())
        # Kandidaten: Template voll, RS >= Mindest, nicht älter als `fresh` Tage
        cands = []
        for s in idx:
            s = int(s)
            if s in book.pos or not F["ok7"][a][s] or F["age"][a][s] > p["fresh"]:
                continue
            r = rs_of(s)
            if r is None or r < p["rs_min"]:
                continue
            sc_ = score(s)
            if sc_ is None:
                continue
            cands.append((s, sc_, r, F["dist"][a][s]))
        cands.sort(key=lambda x: (-x[1], -x[2], x[3], P["tick"][x[0]]))

        def size(s, px):
            rps = px * sp
            by_risk = math.floor(eqv * p["risk"] / 100 / rps)
            by_max = math.floor(eqv * p["max_pos_pct"] / 100 / px)
            return max(0, min(by_risk, by_max))

        # Tausch
        swaps = []
        if p["swap"] and p["sn"] >= 1:
            hs = []
            for s in hold:
                sc_ = score(s)
                g = (Cf[s, i] / book.pos[s]["en"] - 1) * 100
                if sc_ is not None and g < p["sk"]:
                    hs.append((s, sc_, rs_of(s) or 0, g))
            hs.sort(key=lambda x: x[1])
            used, out = set(), {}
            for c_s, c_sc, c_rs, _ in cands:
                for h_s, h_sc, h_rs, g in hs:
                    if h_s in used:
                        continue
                    if c_sc - h_sc >= p["sw"] and c_rs - h_rs >= 10:
                        used.add(h_s)
                        out[h_s] = (c_s, c_sc, h_sc)
                        break
            n_sw = 0
            for h_s, (c_s, c_sc, h_sc) in sorted(out.items(), key=lambda kv: -(kv[1][1] - kv[1][2])):
                days = (P["dates"][i] - P["dates"][book.pos[h_s]["i_in"]]).days
                if days < p["sh"] * 7 or n_sw >= p["sn"]:
                    continue
                q = size(c_s, Cf[c_s, i])
                if q < 1:
                    continue
                val = book.pos[h_s]["q"] * Cf[h_s, i]
                if sum(x[2] * Cf[x[1], i] for x in swaps) + q * Cf[c_s, i] > p["lev"] * eqv - heldAll + sum(book.pos[x[0]]["q"] * Cf[x[0], i] for x in swaps) + val:
                    continue
                swaps.append((h_s, c_s, q))
                n_sw += 1
        swapped = {x[0] for x in swaps}
        swbuy = {x[1] for x in swaps}
        keep = [s for s in hold if s not in swapped]
        held = heldAll - sum(book.pos[s]["q"] * Cf[s, i] for s in swapped)
        slots = max(0, p["np"] - len(keep))
        room = max(0.0, p["lev"] * eqv - held)
        tp = sum(x[2] * Cf[x[1], i] for x in swaps)
        buys = []
        for c_s, c_sc, c_rs, d_ in cands:
            if c_s in swbuy:
                continue
            q = size(c_s, Cf[c_s, i])
            if q < 1:
                continue
            if len(swaps) + len(buys) >= slots:
                break
            if tp + q * Cf[c_s, i] > room:
                break
            buys.append((c_s, q))
            tp += q * Cf[c_s, i]
        pending = (sells, parts, swaps, buys)
    return dict(eq=eq, closed=book.closed, npos=npos, params=p, open=len(book.pos))
