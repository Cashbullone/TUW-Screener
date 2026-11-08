"""Tests der MN- und MO-Simulation sowie der Tages-Turtle-Läufe mit künstlichen Daten (ohne Netzwerk)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import backtest as bt  # noqa: E402
import bt_screeners as bs  # noqa: E402
import make_fake  # noqa: E402

RAW, IX, NAMES, SECS, GOT = make_fake.build_fake(n_random=120)


def daily_panel():
    frames = {}
    for t, df in RAW.items():
        d = df.copy()
        frames[t] = d
    return bt.build_panel(frames, SECS)


P = daily_panel()
SPX = np.nanmean(P["C"], axis=0)
VIX = np.full(len(P["dates"]), 18.0)


def test_review_days():
    r = bs.review_days(P["dates"])
    assert all(P["dates"][i].weekday() <= 4 for i in r)
    w = bs.review_days(P["dates"], 2)
    assert all(P["dates"][i].weekday() == 2 for i in w) and len(w) > 100
    print("ok    test_review_days")


def test_mo_runs_and_accounts():
    rev = bs.review_days(P["dates"], 2)
    F = bs.mo_features(P, rev)
    M = bs.market_state(P, SPX, VIX, rev, bs.MO_DEF)
    r = bs.run_mo(P, F, M, rev, min_dv=0.0, min_price=1.0)
    assert np.isfinite(r["eq"]).all() and (r["eq"] > 0).all()
    assert r["npos"].max() <= 10
    assert len(r["closed"]) + r["open"] > 0, "es muss gehandelt werden"
    # Variante ohne Markt hat einen Marktfilter weniger, läuft ebenfalls durch
    r2 = bs.run_mo(P, F, M, rev, min_dv=0.0, min_price=1.0, market=False, brake=False)
    assert np.isfinite(r2["eq"]).all()
    print("ok    test_mo_runs_and_accounts  trades=%d offen=%d" % (len(r["closed"]), r["open"]))


def test_mn_runs_and_accounts():
    rev = bs.review_days(P["dates"])
    F = bs.mn_features(P, rev)
    r = bs.run_mn(P, F, rev, min_dv=0.0, min_price=1.0)
    assert np.isfinite(r["eq"]).all() and (r["eq"] > 0).all()
    assert r["npos"].max() <= 10 + 1
    assert len(r["closed"]) + r["open"] > 0
    for kw in (dict(swap=False), dict(stages=""), dict(fresh=999)):
        r2 = bs.run_mn(P, F, rev, min_dv=0.0, min_price=1.0, **kw)
        assert np.isfinite(r2["eq"]).all(), kw
    print("ok    test_mn_runs_and_accounts  trades=%d offen=%d" % (len(r["closed"]), r["open"]))


def test_leverage():
    rev = bs.review_days(P["dates"])
    F = bs.mn_features(P, rev)
    r1 = bs.run_mn(P, F, rev, min_dv=0.0, min_price=1.0, risk=3.0, np=4, max_pos_pct=40.0)
    r2 = bs.run_mn(P, F, rev, min_dv=0.0, min_price=1.0, risk=3.0, np=4, max_pos_pct=40.0, lev=2.0)
    assert np.isfinite(r2["eq"]).all() and (r2["eq"] > 0).all()
    assert not np.allclose(r1["eq"], r2["eq"]), "Hebel muss wirken"
    # lev=1 ändert nichts gegenüber Standard
    r0 = bs.run_mn(P, F, rev, min_dv=0.0, min_price=1.0, risk=3.0, np=4, max_pos_pct=40.0, lev=1.0, borrow=0.2)
    assert np.allclose(r0["eq"], r1["eq"])
    rev_w = bs.review_days(P["dates"], 2)
    Fo = bs.mo_features(P, rev_w)
    M = bs.market_state(P, SPX, VIX, rev_w, bs.MO_DEF)
    a = bs.run_mo(P, Fo, M, rev_w, min_dv=0.0, min_price=1.0)
    b = bs.run_mo(P, Fo, M, rev_w, min_dv=0.0, min_price=1.0, lev=2.0)
    assert np.isfinite(b["eq"]).all() and not np.allclose(a["eq"], b["eq"])
    print("ok    test_leverage")


def test_clenow_matches_direct_regression():
    rev = bs.review_days(P["dates"], 2)
    F = bs.mo_features(P, rev)
    a = len(rev) - 3
    i = rev[a]
    s = 5
    y = np.log(P["C"][s, i - 89:i + 1])
    x = np.arange(90)
    b, _ = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    want = (np.exp(b * 250) - 1) * 100 * r2
    assert abs(F["cl"][a][s] - want) < 1e-6 * max(1, abs(want)), (F["cl"][a][s], want)
    print("ok    test_clenow_matches_direct_regression")


def test_tu_daily_engine():
    I, s1 = bt.indicators(P, dict(bt.DEFAULTS, mom_weeks=130, dv_win=50, dv_div=1.0))
    r = bt.run_backtest(P, I, s1, mom_weeks=130, dv_win=50, dv_div=1.0, min_dv=0.0, min_price=1.0)
    assert np.isfinite(r["eq"]).all() and len(r["closed"]) > 0
    print("ok    test_tu_daily_engine  trades=%d" % len(r["closed"]))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
