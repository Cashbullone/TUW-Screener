"""Tests der Backtest-Simulation (ohne Netzwerk)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import backtest as bt  # noqa: E402
import make_fake  # noqa: E402
import turtle_core as tc  # noqa: E402


def weekly_frame(closes, rng=1.0, vol=5_000_000, end="2026-10-30"):
    idx = pd.date_range(end=end, periods=len(closes), freq="W-FRI")
    c = np.asarray(closes, float)
    o = np.concatenate(([c[0]], c[:-1]))
    h = np.maximum(o, c) + rng
    l = np.minimum(o, c) - rng
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": float(vol)}, index=idx)


def trend_then_crash():
    flat = [100.0 + (0.5 if i % 2 else -0.5) for i in range(80)]
    up = [100.0 * 1.03 ** k for k in range(1, 31)]
    dn = [up[-1] * 0.93 ** k for k in range(1, 15)]
    return flat + up + dn


def panel_for(frames):
    P = bt.build_panel(frames, {t: "Information Technology" for t in frames})
    I, s1 = bt.indicators(P, bt.DEFAULTS)
    return P, I, s1


def test_trend_is_bought_and_stopped_out():
    P, I, s1 = panel_for({"UP": weekly_frame(trend_then_crash()), "FLAT": weekly_frame([100 + (i % 2) for i in range(124)])})
    res = bt.run_backtest(P, I, s1)
    tr = [t for t in res["closed"] if t["t"] == "UP"]
    assert tr, "Der Trend muss gehandelt werden"
    t = tr[0]
    assert t["d"] == 1 and t["sys"] == 1 and 1 <= t["units"] <= 4
    assert t["px_out"] > t["px_in"] and t["pnl"] > 0, "Trendgewinn erwartet: %s" % t
    assert res["eq"][-1] > bt.DEFAULTS["equity"]
    assert not [x for x in res["closed"] if x["t"] == "FLAT"]
    print("ok    test_trend_is_bought_and_stopped_out")


def test_position_size_follows_risk_over_n():
    P, I, s1 = panel_for({"UP": weekly_frame(trend_then_crash()), "FLAT": weekly_frame([100 + (i % 2) for i in range(124)])})
    res = bt.run_backtest(P, I, s1, cost=0.0)
    t = [x for x in res["closed"] if x["t"] == "UP"][0]
    # eine Einheit riskiert 0,25 % von 100.000 je N -> Stück = 250 / N, das Risiko (2 N je Stück) also 2 x 250 = 500 je Einheit
    assert abs(t["risk"] / t["units"] - 500) < 60, t
    print("ok    test_position_size_follows_risk_over_n")


def test_equity_identity_without_costs():
    RAW, IX, NAMES, SECS, GOT = make_fake.build_fake(n_random=80)
    weekly = {t: tc.to_weekly(df) for t, df in RAW.items()}
    P = bt.build_panel(weekly, SECS)
    I, s1 = bt.indicators(P, bt.DEFAULTS)
    for kw in (dict(), dict(short=True), dict(risk=1.0, cap_pos=50.0)):
        res = bt.run_backtest(P, I, s1, cost=0.0, min_dv=0.0, min_price=1.0, **kw)
        eq0 = res["params"]["equity"]
        unreal = sum(o["basis"] * 0 + o["d"] * (o["q"] * P["C"][o["s"], -1] - o["basis"]) for o in res["open"].values())
        real = sum(t["pnl"] for t in res["closed"])
        assert abs(eq0 + real + unreal - res["eq"][-1]) < 1e-3 * eq0, (kw, eq0 + real + unreal, res["eq"][-1])
        assert np.isfinite(res["eq"]).all() and (res["eq"] > 0).all()
        assert all(1 <= t["units"] <= 4 for t in res["closed"])
        assert res["gross"].max() < 1.6
        assert len(res["closed"]) > 0, kw
    print("ok    test_equity_identity_without_costs")


def test_limits_hold():
    RAW, IX, NAMES, SECS, GOT = make_fake.build_fake(n_random=80)
    weekly = {t: tc.to_weekly(df) for t, df in RAW.items()}
    P = bt.build_panel(weekly, SECS)
    I, s1 = bt.indicators(P, bt.DEFAULTS)
    res = bt.run_backtest(P, I, s1, min_dv=0.0, min_price=1.0, max_dir=3, max_sec=2)
    # offene Einheiten nach dem Lauf dürfen die Grenzen nicht überschreiten
    units = sum(o["units"] for o in res["open"].values())
    assert units <= 3
    print("ok    test_limits_hold")


def test_costs_reduce_result():
    P, I, s1 = panel_for({"UP": weekly_frame(trend_then_crash()), "FLAT": weekly_frame([100 + (i % 2) for i in range(124)])})
    a = bt.run_backtest(P, I, s1, cost=0.0)["eq"][-1]
    b = bt.run_backtest(P, I, s1, cost=0.01)["eq"][-1]
    assert b < a
    print("ok    test_costs_reduce_result")


def test_stats():
    eq = np.array([100.0, 110.0, 99.0, 120.0] * 20)
    d = pd.date_range("2020-01-03", periods=len(eq), freq="W-FRI")
    s = bt.curve_stats(eq, d, start=0)
    assert s["maxdd"] < 0 and s["total"] == eq[-1] / eq[0] - 1
    print("ok    test_stats")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
