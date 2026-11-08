"""Tests für den Rechenkern. Start: python tests/test_core.py (oder pytest)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import turtle_core as tc  # noqa: E402


def build(closes, spread=1.0, overrides=None):
    """Tagesreihe aus Schlusskursen: Eröffnung = Vortagesschluss, Hoch = Schluss + spread, Tief = Schluss - spread.
    overrides: {Tag: dict(o=, h=, l=, c=)}"""
    c = list(map(float, closes))
    o = [c[0]] + c[:-1]
    h = [x + spread for x in c]
    l = [x - spread for x in c]
    for i, d in (overrides or {}).items():
        if "c" in d: c[i] = float(d["c"])
        if "o" in d: o[i] = float(d["o"])
        if "h" in d: h[i] = float(d["h"])
        if "l" in d: l[i] = float(d["l"])
    return np.array(o), np.array(h), np.array(l), np.array(c)


def ref_n(h, l, c, period=20):
    """Unabhängige Referenz in reinem Python."""
    tr = []
    for i in range(len(c)):
        if i == 0:
            tr.append(h[i] - l[i])
        else:
            tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    out = [None] * len(c)
    out[period - 1] = sum(tr[:period]) / period
    for i in range(period, len(c)):
        out[i] = ((period - 1) * out[i - 1] + tr[i]) / period
    return out


def test_true_range_and_n():
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(120)))
    h = c * (1 + 0.01 * rng.random(120))
    l = c * (1 - 0.01 * rng.random(120))
    c[30] *= 1.08                                   # Lücke: True Range muss den Vortagesschluss sehen
    h[30], l[30] = max(h[30], c[30]), min(l[30], c[30])
    n = tc.wilder_n(tc.true_range(h, l, c))
    ref = ref_n(h, l, c)
    assert all(math.isnan(x) for x in n[:19])
    for i in range(19, 120):
        assert abs(n[i] - ref[i]) < 1e-9, i
    tr = tc.true_range(h, l, c)
    assert abs(tr[30] - max(h[30] - l[30], abs(h[30] - c[29]), abs(l[30] - c[29]))) < 1e-12
    assert tr[30] > h[30] - l[30]                   # die Lücke zählt mit


def test_constant_range_gives_n_equal_range():
    o, h, l, c = build([100] * 40)
    n = tc.wilder_n(tc.true_range(h, l, c))
    assert abs(n[-1] - 2.0) < 1e-12 and abs(n[19] - 2.0) < 1e-12


def scenario():
    """Drei System-1-Trades hintereinander: Gewinner (gehandelt), Verlierer (übersprungen), Verlierer (gehandelt)."""
    closes = [100 + t for t in range(41)]                       # Tag 0..40 steigt um 1 pro Tag
    closes += [105] * 21                                        # Tag 41..61: Absturz, danach seitwärts
    closes += [107, 62]                                         # Tag 62: Ausbruch, Tag 63: Absturz
    closes += [62] * 21                                         # Tag 64..84
    ov = {
        41: dict(o=140, h=140.5, l=100, c=105),                 # Ausstieg über das 10-Tage-Tief, Gewinner
        62: dict(o=106, h=108, l=105.5, c=107),                 # Ausbruch über 106, System 1 wird übersprungen
        63: dict(o=107, h=107.5, l=60, c=62),                   # Verlierer
        84: dict(o=62, h=65, l=61.5, c=64),                     # nächster Ausbruch, wird gehandelt
    }
    return build(closes, 1.0, ov)


def test_s1_sequence_win_skip_loss_take():
    o, h, l, c = scenario()
    n = tc.wilder_n(tc.true_range(h, l, c))
    tr, pos, last = tc.sim_long(o, h, l, n, 20, 10)
    assert len(tr) == 2, tr
    t1, t2 = tr
    assert t1["entry_i"] == 20 and t1["entry"] == 120 and abs(t1["stop"] - 116) < 1e-9 and t1["taken"] is True
    assert t1["exit_i"] == 41 and t1["exit"] == 130 and t1["res"] == "W"
    assert t2["entry_i"] == 62 and t2["entry"] == 106 and t2["taken"] is False       # nach Gewinner übersprungen
    assert t2["exit_i"] == 63 and t2["res"] == "L" and t2["exit"] == 104
    assert pos is not None and pos["entry_i"] == 84 and pos["taken"] is True and last == "L"


def test_stop_two_n_exit_and_gap():
    closes = [100 + t for t in range(25)] + [111]
    o, h, l, c = build(closes, 1.0, {25: dict(o=124, h=124.5, l=110, c=111)})
    n = tc.wilder_n(tc.true_range(h, l, c))
    tr, pos, last = tc.sim_long(o, h, l, n, 20, 10)
    assert tr[0]["exit"] == 116 and tr[0]["res"] == "L" and pos is None and last == "L"      # Stop 120 - 2*2
    o2 = o.copy(); o2[25] = 100                                                              # Lücke unter den Stop
    tr, pos, last = tc.sim_long(o2, h, l, n, 20, 10)
    assert tr[0]["exit"] == 100 and tr[0]["res"] == "L"                                      # Verkauf zur Eröffnung


def test_gap_entry_uses_open():
    closes = [100] * 25 + [110]
    o, h, l, c = build(closes, 1.0, {25: dict(o=104, h=111, l=103.5, c=110)})
    n = tc.wilder_n(tc.true_range(h, l, c))
    tr, pos, last = tc.sim_long(o, h, l, n, 20, 10)
    assert pos["entry"] == 104 and pos["entry_i"] == 25                                      # Lücke: Eröffnung statt Marke 101


def test_no_entry_on_exit_day_and_no_entry_without_history():
    closes = [100] * 15 + [120]                      # Ausbruch vor Tag 20: Marke gibt es noch nicht
    o, h, l, c = build(closes, 1.0)
    n = tc.wilder_n(tc.true_range(h, l, c))
    tr, pos, last = tc.sim_long(o, h, l, n, 20, 10)
    assert pos is None and not tr


def test_short_is_mirror_of_long():
    o, h, l, c = scenario()
    K = 300.0
    om, hm, lm, cm = K - o, K - l, K - h, K - c
    n = tc.wilder_n(tc.true_range(h, l, c))
    nm = tc.wilder_n(tc.true_range(hm, lm, cm))
    assert np.allclose(n[19:], nm[19:])
    trl, posl, lastl = tc.sim_side(o, h, l, n, 20, 10, +1)
    trs, poss, lasts = tc.sim_side(om, hm, lm, nm, 20, 10, -1)
    assert len(trl) == len(trs) and lastl == lasts
    for a, b in zip(trl, trs):
        assert a["entry_i"] == b["entry_i"] and a["exit_i"] == b["exit_i"] and a["res"] == b["res"]
        assert abs((K - a["entry"]) - b["entry"]) < 1e-9 and abs((K - a["exit"]) - b["exit"]) < 1e-9
        assert abs((K - a["stop"]) - b["stop"]) < 1e-9 and a["taken"] == b["taken"]
    assert poss["entry_i"] == posl["entry_i"] and abs((K - posl["entry"]) - poss["entry"]) < 1e-9


def test_analyse_signals_and_state():
    o, h, l, c = scenario()
    # letzter Tag = Tag 62: System-1-Ausbruch, aber übersprungen; kein 55-Tage-Ausbruch (Hoch 140,5 in 55 Tagen)
    k = 63
    r = tc.analyse(o[:k], h[:k], l[:k], c[:k], days=np.arange(k) + 20000)
    assert r["sg"] == [] and r["sk"] == ["1L"]
    assert r["f"][0] == ["W", 20062, 0]                         # gedachter Trade läuft seit Tag 62, wird nicht gehandelt
    assert r["f"][1] == ["W", 0, 0]                             # Short: Trade Tag 41 bis 52 war ein Gewinner, jetzt flach
    # letzter Tag = Tag 84: gehandelter System-1-Ausbruch, Vorgänger war Verlierer
    k = 85
    r = tc.analyse(o[:k], h[:k], l[:k], c[:k], days=np.arange(k) + 20000)
    assert r["sg"] == ["1L"] and r["sk"] == []
    assert r["f"][0] == ["L", 20084, 1]
    assert r["hh"][1] == 65 and r["hh"][0] == 65 and r["f"][1] == ["W", 0, 0]
    # letzter Tag = Tag 63: Absturz. Short-System-1 wird übersprungen (letzter Short war Gewinner),
    # die 55-Tage-Marke löst aber trotzdem System 2 aus (Absicherung)
    k = 64
    r = tc.analyse(o[:k], h[:k], l[:k], c[:k], days=np.arange(k) + 20000)
    assert r["sk"] == ["1S"] and "2S" in r["sg"] and "1S" not in r["sg"]
    assert r["f"][1] == ["W", 20063, 0] and r["f"][0][0] == "L"


def test_analyse_s2_signal_while_s1_open():
    closes = [100 + t for t in range(61)]                       # stetig steigend, System-1-Trade seit Tag 20
    o, h, l, c = build(closes)
    r = tc.analyse(o, h, l, c, days=np.arange(len(c)) + 20000)
    assert "2L" in r["sg"] and "1L" not in r["sg"]               # 55-Tage-Hoch heute, System 1 läuft schon
    assert r["f"][0] == ["", 20020, 1]
    assert r["hh"] == [161.0, 161.0, 161.0]                      # Marken für morgen: heutiges Hoch
    assert r["ll"][0] == 150.0 and r["ll"][1] == 140.0 and r["ll"][2] == 105.0


def test_analyse_short_signal():
    closes = [200 - t for t in range(61)]
    o, h, l, c = build(closes)
    r = tc.analyse(o, h, l, c)
    assert "2S" in r["sg"] and "2L" not in r["sg"]
    assert r["f"][1][2] == 1 and r["f"][0][0] == ""


def test_too_short_returns_none():
    o, h, l, c = build([100] * 60)
    assert tc.analyse(o, h, l, c) is None


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                print("FEHLER", name)
                traceback.print_exc()
    sys.exit(1 if fails else 0)
