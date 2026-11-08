"""Rechenkern des TUW-Screeners (Turtle-Strategie auf Wochenkerzen, ohne Netzwerk, damit er sich testen lässt).

Regeln (Richard Dennis / William Eckhardt, so beschrieben bei Curtis Faith, "Way of the Turtle"), hier auf
Wochenkerzen angewendet. Eine Wochenkerze reicht von Montag bis Freitag: Eröffnung am ersten Handelstag, Hoch und Tief
der Woche, Schluss am letzten Handelstag. "Tage" in den Originalregeln werden zu Wochen:
  N          Wilder-Durchschnitt der True Range über 20 Wochen: N = (19 * N_alt + TR) / 20
  System 1   Einstieg beim Bruch des 20-Wochen-Hochs (Short: -Tiefs), Ausstieg beim 10-Wochen-Tief (Short: -Hoch).
             Überspringen: War der letzte System-1-Trade (auch ein nur gedachter) ein Gewinner, wird der nächste
             System-1-Ausbruch übersprungen.
  System 2   Einstieg beim Bruch des 55-Wochen-Hochs, Ausstieg beim 20-Wochen-Tief, ohne Überspringen (Absicherung).
  Stop       2 N vom Einstieg.

Alle Preise sind Wochenwerte. Ein Ausbruch gilt, wenn das Wochenhoch das Hoch der Vorwochen übersteigt. Bei einer
Kurslücke über die Marke zum Wochenbeginn wird zur Eröffnung der Woche eingestiegen, sonst genau an der Marke.
"""
import math

import numpy as np
import pandas as pd

N_PERIOD = 20
S1_IN, S1_OUT = 20, 10
S2_IN, S2_OUT = 55, 20
STOP_MULT = 2.0
NH = 104                # Wochen, die als Kursreihe an die App gehen


def to_weekly(d):
    """Wochenkerzen aus Tagesdaten (Spalten Open, High, Low, Close, Volume, Zeitindex). Die Woche endet am Freitag und
    trägt dessen Datum. Eine laufende Woche ist dabei noch unvollständig, der Aufrufer schneidet sie ab."""
    w = d.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return w.dropna(subset=["Open", "High", "Low", "Close"])


def true_range(h, l, c):
    """True Range = max(H-L, |H-Vortagesschluss|, |L-Vortagesschluss|). Am ersten Tag nur H-L."""
    pc = np.concatenate(([np.nan], c[:-1]))
    return np.fmax(np.fmax(h - l, np.abs(h - pc)), np.abs(l - pc))


def wilder_n(tr, period=N_PERIOD):
    """N: erster Wert = einfacher Mittelwert der ersten 20 TR, danach (19 * N_alt + TR) / 20."""
    n = np.full(len(tr), np.nan)
    if len(tr) < period:
        return n
    n[period - 1] = tr[:period].mean()
    for i in range(period, len(tr)):
        n[i] = ((period - 1) * n[i - 1] + tr[i]) / period
    return n


def roll_max(x, k):
    """Höchster Wert der letzten k Wochen einschließlich der aktuellen."""
    return pd.Series(x).rolling(k).max().to_numpy()


def roll_min(x, k):
    return pd.Series(x).rolling(k).min().to_numpy()


def prev(x):
    """Reihe um eine Woche nach hinten schieben: Wert der Vorwoche."""
    return np.concatenate(([np.nan], x[:-1]))


def sim_long(o, h, l, n, k_in, k_out, stop_mult=STOP_MULT):
    """Spielt die Einheit eines Systems durch (Long-Seite, eine Einheit, kein Zukauf).

    Liefert (Trades, offener Trade oder None, Ergebnis des letzten abgeschlossenen Trades 'W'/'L'/'').
    Jeder Ausbruch bei flacher Position beginnt einen Trade, auch wenn er nach der Skip-Regel nicht gehandelt würde
    (taken=False). Er wird trotzdem verfolgt, weil sein Ausgang die nächste Entscheidung bestimmt.
    Am Einstiegstag wird nicht auf Stop geprüft (die Reihenfolge von Hoch und Tief im Tag ist unbekannt)."""
    hh = prev(roll_max(h, k_in))            # Hoch der k_in Tage vor dem jeweiligen Tag
    ll = prev(roll_min(l, k_out))
    trades, pos, last = [], None, ''
    for i in range(len(h)):
        if pos is not None:
            lvl = pos["stop"] if math.isnan(ll[i]) else max(pos["stop"], ll[i])
            if l[i] <= lvl:
                px = min(o[i], lvl)         # Lücke unter die Marke: Verkauf zur Eröffnung
                res = "W" if px > pos["entry"] else "L"
                trades.append(dict(pos, exit_i=i, exit=px, res=res))
                last, pos = res, None
                continue                    # am Ausstiegstag kein neuer Einstieg
        if pos is None and i > 0 and not math.isnan(hh[i]) and not math.isnan(n[i - 1]) and h[i] > hh[i]:
            ep = max(o[i], hh[i])
            pos = {"entry_i": i, "entry": float(ep), "stop": float(ep - stop_mult * n[i - 1]),
                   "n": float(n[i - 1]), "taken": last != "W"}
    return trades, pos, last


def sim_side(o, h, l, n, k_in, k_out, side, stop_mult=STOP_MULT):
    """Long (side=+1) oder Short (side=-1). Short wird als gespiegelter Long-Fall gerechnet."""
    if side > 0:
        return sim_long(o, h, l, n, k_in, k_out, stop_mult)
    trades, pos, last = sim_long(-o, -l, -h, n, k_in, k_out, stop_mult)
    flip = lambda d: None if d is None else dict(d, entry=-d["entry"], stop=-d["stop"],
                                                 **({"exit": -d["exit"]} if "exit" in d else {}))
    return [flip(t) for t in trades], flip(pos), last


def analyse(o, h, l, c, days=None):
    """Alle Kennzahlen einer Aktie für die letzte Woche. Arrays gleicher Länge, aufsteigend nach Datum, ohne Lücken (NaN).
    days: optional, Wochen als Unix-Tage (Freitag, für das Datum eines laufenden gedachten System-1-Trades).

    Rückgabe (dict) oder None, wenn die Reihe zu kurz ist."""
    o, h, l, c = (np.asarray(x, dtype=float) for x in (o, h, l, c))
    m = len(c)
    if m < S2_IN + 6:
        return None
    n = wilder_n(true_range(h, l, c))
    if math.isnan(n[-1]) or n[-1] <= 0:
        return None
    last = m - 1
    out = {
        "n": float(n[-1]),
        "hh": [float(roll_max(h, k)[-1]) for k in (S1_OUT, S1_IN, S2_IN)],     # Marken für die nächste Woche, einschließlich der letzten
        "ll": [float(roll_min(l, k)[-1]) for k in (S1_OUT, S1_IN, S2_IN)],
        "f": [], "sg": [], "sk": [], "trades": {},
    }
    hh55, ll55 = prev(roll_max(h, S2_IN))[last], prev(roll_min(l, S2_IN))[last]
    for side, tag in ((1, "L"), (-1, "S")):
        trades, pos, lastres = sim_side(o, h, l, n, S1_IN, S1_OUT, side)
        out["trades"][tag] = trades
        day = 0
        if pos:
            day = int(days[pos["entry_i"]]) if days is not None else int(pos["entry_i"])
        # f je Richtung: [Ergebnis des letzten abgeschlossenen System-1-Trades 'W'/'L'/'', Starttag des laufenden
        # gedachten Trades oder 0, 1 wenn dieser Trade nach der Skip-Regel gehandelt würde]
        out["f"].append([lastres, day, 1 if (pos and pos["taken"]) else 0])
        if pos and pos["entry_i"] == last:
            (out["sg"] if pos["taken"] else out["sk"]).append("1" + tag)
        # System 2: 55-Wochen-Ausbruch in der letzten Woche (gilt immer, auch wenn System 1 übersprungen wurde)
        if side > 0 and not math.isnan(hh55) and h[last] > hh55:
            out["sg"].append("2L")
        if side < 0 and not math.isnan(ll55) and l[last] < ll55:
            out["sg"].append("2S")
    out["o_last"] = float(o[last])
    out["n_all"] = n                                            # N für jeden Tag (für die Kursreihe der App, je Woche)
    return out


def r2(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 2)
