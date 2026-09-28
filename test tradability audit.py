#!/usr/bin/env python3
"""
tests/test_tradability_audit.py - prints VERIFIED on success.

1. Flags on REAL bars copied from edge_anatomy's printed examples (ELIN, TTML,
   TANLA, LLOYDSENGG, TVSELECT lock; RPOWER, EXICOM, PCJEWELLER do not).
2. Hand-built flag cases (close at band vs close at high off-band, etc.).
3. ETF name rules; --etf-list loading.
4. Substitution and the cash variant on hand-built days.
5. Version A's exit session/outcome equals entry_timing's ft_A on random paths.
6. A stop exit on a session frozen at the lower band is flagged as unfillable.
7. Three planted worlds, end to end, next to entry_timing and edge_anatomy on
   the same fixture (the audit must reproduce both to machine precision):
     circuit: ranks 1-3 are locked at the upper band at the close and open
              frozen at the band -> A looks great, A_sub FAILS
     clean:   ranks 1-3 gap +2.3% overnight with no lock -> nothing flagged,
              A_sub = A_orig, PASSES
     etf:     same as clean but ranks 1-3 are ETF-named -> A_sub FAILS,
              A_sub_circuit PASSES
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import tradability_audit as TA  # noqa: E402
import entry_timing as ET  # noqa: E402
import edge_anatomy as EA  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def two_bars(prev_bar, next_bar):
    """prev_bar/next_bar = (o, h, l, c); returns flags for bar 0 (signal) and bar 1."""
    o, h, l, c = (np.array([prev_bar[k], next_bar[k]], float) for k in range(4))
    return TA.symbol_flags(o, h, l, c)


# ----------------------------------------------------------------------
def test_real_bars():
    print("1. flags on real bars from edge_anatomy's examples")
    cases = [  # (name, close_T, T+1 OHLC, open_uc, frozen1)
        ("ELIN 2025-07-10", 177.46, (181.0, 181.0, 181.0, 181.0), True, True),
        ("TVSELECT 2026-01-01", 437.05, (445.75, 445.75, 445.75, 445.75), True, True),
        ("TTML 2022-02-04", 164.0, (172.2, 172.2, 165.1, 172.2), True, False),
        ("TANLA 2020-03-27", 44.55, (46.75, 46.75, 42.35, 46.15), True, False),
        ("LLOYDSENGG 2022-02-04", 15.74, (16.49, 16.49, 15.74, 16.49), True, False),
        ("RPOWER 2021-05-10", 6.65, (7.0, 7.3, 6.65, 7.3), False, False),
        ("EXICOM 2025-11-25", 111.76, (114.0, 117.88, 110.49, 116.48), False, False),
        ("PCJEWELLER 2022-09-20", 7.76, (7.86, 8.14, 7.80, 8.14), False, False),
        ("PNBGILTS 2022-04-26", 61.70, (61.46, 61.65, 60.97, 61.26), False, False),
    ]
    for name, cT, nb, want_uc, want_fz in cases:
        f = two_bars((cT, cT * 1.01, cT * 0.99, cT), nb)
        check(bool(f["open_uc"][0]) == want_uc and bool(f["frozen1"][0]) == want_fz,
              f"{name}: open_uc={want_uc}, frozen={want_fz}")
    # TTML T+4 -> T+5: frozen at the lower 5% band
    f = two_bars((163.25, 163.25, 163.25, 163.25), (155.1, 155.1, 155.1, 155.1))
    check(bool(f["frozen_lower"][1]), "TTML 2022-02-11: frozen at the lower band")
    # ELIN T+2 -> T+3 is a frozen UPPER day, not lower
    f = two_bars((184.62, 184.62, 184.62, 184.62), (188.31, 188.31, 188.31, 188.31))
    check(not f["frozen_lower"][1], "ELIN frozen UPPER day is not a lower-band lock")


def test_hand_flags():
    print("2. hand-built close-lock cases")
    f = two_bars((160, 165, 159, 164.0), (165, 172.2, 164.5, 172.2))
    check(bool(f["uc_close"][1]), "prev 164.00 -> close = high = 172.20 (+5%): locked at the close")
    f = two_bars((99, 101, 98, 100.0), (101, 103.4, 100.5, 103.4))
    check(not f["uc_close"][1] and bool(f["uc_close_loose"][1]),
          "close = high at +3.4% (no band): strict no, loose yes")
    f = two_bars((99, 101, 98, 100.0), (101, 106.0, 100.5, 105.0))
    check(not f["uc_close"][1] and not f["uc_close_loose"][1], "+5% close below the high: not locked")
    f = two_bars((99, 101, 98, 100.0), (100.5, 101.5, 100.2, 101.5))
    check(not f["uc_close"][1] and not f["uc_close_loose"][1], "close = high at +1.5%: neither")
    f = two_bars((6.5, 6.7, 6.4, 6.65), (6.8, 7.0, 6.7, 7.0))
    check(bool(f["uc_close"][1]), "Rs 6.65 -> 7.00 (+5.26%, tick rounding): locked")
    f = two_bars((99, 101, 98, 100.0), (101, 120.0, 100.5, 120.0))
    check(bool(f["uc_close"][1]), "+20% band: locked")
    f = two_bars((np.nan, np.nan, np.nan, np.nan), (101, 105.0, 100.5, 105.0))
    check(not f["uc_close"][1], "no previous close: never flagged")


def test_etf_names():
    print("3. ETF rules")
    for s in ("MON100", "MAFANG", "NIFTYBEES", "GOLDBEES", "SETFNIF50", "CPSEETF", "LIQUIDETF",
              "HNGSNGBEES", "MASPTOP50"):
        check(TA.is_etf(s), f"{s} treated as ETF")
    for s in ("KOTAKBANK", "HDFCBANK", "PNBGILTS", "TTML", "BEML", "RPOWER", "ELIN"):
        check(not TA.is_etf(s), f"{s} is an equity")
    d = Path(tempfile.mkdtemp())
    try:
        (d / "x.txt").write_text("# mine\nmyetf1\n\nABC\n", encoding="utf-8")
        check(TA._load_etf_list(d / "x.txt") == {"MYETF1", "ABC"}, "text list loaded, comments skipped")
        pd.DataFrame({"Symbol ": ["QQQX", "ZZZ"]}).to_csv(d / "y.csv", index=False)
        check(TA._load_etf_list(d / "y.csv") == {"QQQX", "ZZZ"}, "CSV list loaded by its SYMBOL column")
        check(TA.is_etf("QQQX", {"QQQX"}), "extra list applies")
    finally:
        shutil.rmtree(d)


def test_substitution():
    print("4. substitution and the cash variant")
    t1, t2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    rows = []
    for t in (t1, t2):
        for r in range(1, 6):
            rows.append({"timestamp": t, "symbol": f"S{r}", "rk": float(r), "xr_A": r / 100, "xr_C": r / 100,
                         "uc_close": (t == t1 and r == 1), "uc_close_loose": False,
                         "open_uc": (t == t2 and r == 2), "frozen1": False, "etf": (r == 3)})
    V = pd.DataFrame(rows)
    top = TA.select_top(V, 3, "uc_close|etf")
    got = {t: list(g["symbol"]) for t, g in top.groupby("timestamp")}
    check(got[t1] == ["S2", "S4", "S5"] and got[t2] == ["S1", "S2", "S4"],
          "locked rank 1 and ETF rank 3 skipped; next ranks move up")
    t, uni, _ = TA.variant_days(V, V, "C_cash", 3, cost=0.0)
    # day 2: ETF rank 3 substituted by rank 4; rank 2 opens locked -> slot = 0
    check(math.isclose(t.loc[t2], (0.01 + 0.0 + 0.04) / 3), "cash variant: locked slot earns 0, ETF still substituted")
    t, uni, _ = TA.variant_days(V, V, "C_sub", 3, cost=0.0)
    check(math.isclose(t.loc[t2], (0.01 + 0.04 + 0.05) / 3), "sub variant: locked slot replaced by the next rank")
    check(math.isclose(uni.loc[t1], (0.01 + 0.02 + 0.04 + 0.05) / 4), "buy-everything drops the same flagged rows")


def _walk(n, rng, start=100.0):
    o, h, l, c = (np.zeros(n) for _ in range(4))
    p = start
    for i in range(n):
        o[i] = p * (1 + rng.normal(0, 0.003))
        c[i] = o[i] * (1 + rng.normal(0, 0.02))
        h[i] = max(o[i], c[i]) * (1 + rng.uniform(0.002, 0.012))
        l[i] = min(o[i], c[i]) * (1 - rng.uniform(0.002, 0.012))
        p = c[i]
    return [np.round(x, 2) for x in (o, h, l, c)]


def test_exit_day():
    print("5. version A exit session/outcome = entry_timing's ft_A")
    rng = np.random.default_rng(3)
    mism, bad_day, n_all = 0, 0, 0
    for _ in range(20):
        o, h, l, c = _walk(400, rng)
        day, res = TA.exit_day_A(o, h, l, c)
        lab = ET.symbol_labels(o, h, l, c)
        m = np.isfinite(lab["ft_A"]) | np.isfinite(res)
        mism += int((np.nan_to_num(res[m], nan=9) != np.nan_to_num(lab["ft_A"][m], nan=9)).sum())
        both = np.isfinite(res)
        bad_day += int(((day[both] == 0) != (res[both] == 0)).sum())
        n_all += int(m.sum())
    check(mism == 0 and n_all > 7000, f"outcome: 0 mismatches on {n_all:,} labels")
    check(bad_day == 0, "timeout <-> session 0, barrier hit <-> session 1..5")


def _write_symbol(root, sym, dates, o, h, l, c):
    pd.DataFrame({"timestamp": dates, "open": o, "high": h, "low": l, "close": c,
                  "volume": 1e6}).to_parquet(root / f"{sym}_daily.parquet", index=False)


def test_exit_lock():
    print("6. a stop on a session frozen at the lower band is unfillable")
    d = Path(tempfile.mkdtemp())
    try:
        rng = np.random.default_rng(5)
        o, h, l, c = _walk(40, rng)
        i = 30                                   # signal day
        # T+1 normal, T+2 frozen at the lower 5% band, then normal
        o[i + 1], h[i + 1], l[i + 1], c[i + 1] = c[i], c[i] * 1.004, c[i] * 0.996, c[i]
        lk = math.ceil(c[i + 1] * 0.95 / 0.05) * 0.05
        o[i + 2] = h[i + 2] = l[i + 2] = c[i + 2] = round(lk, 2)
        for k in range(i + 3, 40):
            o[k], h[k], l[k], c[k] = c[k - 1], c[k - 1] * 1.004, c[k - 1] * 0.996, c[k - 1]
        dates = pd.bdate_range("2024-01-01", periods=40)
        _write_symbol(d, "LOCKD", dates, o, h, l, c)
        S = TA.collect(d, pd.DataFrame({"timestamp": [dates[i]], "symbol": ["LOCKD"]}))
        r = S.iloc[0]
        check(r["ft_C"] == -1 and r["day_C"] == 2, "C stops out on T+2 (gap through the stop)")
        check(bool(r["exit_locked_C"]), "that stop is flagged unfillable (frozen at the lower band)")
        check(r["ft_A"] == -1 and bool(r["exit_locked_A"]) == (r["day_A"] == 2),
              "A's stop is flagged only if it also lands on the frozen session")
    finally:
        shutil.rmtree(d)


# ----------------------------------------------------------------------
def floor_tick(x, t=0.05):
    return round(math.floor(x / t + 1e-9) * t, 2)


def make_world(kind, root: Path, n_days=320, n_event=39, n_fill=30, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    wf = list(range(40, n_days - 8))
    ev_name = (lambda j: f"E{j}BEES") if kind == "etf" else (lambda j: f"EV{j}")
    events = {j: set() for j in range(n_event)}
    for d in wf:
        for j in range(3):
            events[(3 * d + j) % n_event].add(d)
    syms = [ev_name(j) for j in range(n_event)] + [f"F{j}" for j in range(n_fill)]
    bars = {}
    for s_i, sym in enumerate(syms):
        ev = events.get(s_i, set()) if s_i < n_event else set()
        o, h, l, c = (np.zeros(n_days) for _ in range(4))
        p, force_next = 100.0 + 5 * s_i, None
        for i in range(n_days):
            if force_next is not None:
                o[i], h[i], l[i], c[i] = force_next
                force_next = None
            elif i in ev and kind == "circuit":
                o[i] = p * 1.01
                c[i] = h[i] = floor_tick(p * 1.05)
                l[i] = min(o[i], c[i]) * 0.995
                lk = floor_tick(c[i] * 1.05)
                force_next = (lk, lk, lk, lk)
            elif i in ev:                       # clean / etf: genuine overnight gap, no lock
                o[i] = p * 1.003
                c[i] = o[i] * 1.005
                h[i], l[i] = c[i] * 1.004, o[i] * 0.996
                o1 = c[i] * 1.023
                c1 = o1 * (1 + rng.normal(0, 0.01))
                force_next = (o1, max(o1, c1) * 1.006, min(o1, c1) * 0.994, c1)
            else:
                o[i] = p * (1 + float(np.clip(rng.normal(0, 0.003), -0.012, 0.012)))
                c[i] = o[i] * (1 + rng.normal(0, 0.015))
                h[i] = max(o[i], c[i]) * (1 + rng.uniform(0.002, 0.01))
                l[i] = min(o[i], c[i]) * (1 - rng.uniform(0.002, 0.01))
            o[i], h[i], l[i], c[i] = (round(float(x), 2) for x in (o[i], h[i], l[i], c[i]))
            p = c[i]
        bars[sym] = (o, h, l, c)
        _write_symbol(root, sym, dates, o, h, l, c)
    # panel with the real labels (so the built-in panel checks run)
    prow = []
    for sym, (o, h, l, c) in bars.items():
        lab = ET.symbol_labels(o, h, l, c)
        prow.append(pd.DataFrame({"timestamp": dates, "symbol": sym, "label_exit_ret": lab["xr_A"],
                                  "label_exit_ret_o1": lab["xr_C"]}))
    (root / "panel" / "engine").mkdir(parents=True, exist_ok=True)
    pd.concat(prow, ignore_index=True).to_parquet(root / "panel" / "panel.parquet", index=False)
    vrow = []
    for d in wf:
        evs = [ev_name(j) for j in range(n_event) if d in events[j]]
        fill = list(rng.choice([f"F{j}" for j in range(n_fill)], 17, replace=False))
        for r, s in enumerate(evs + fill, start=1):
            vrow.append({"timestamp": dates[d], "symbol": s, "rk": float(r)})
    pd.DataFrame(vrow).to_parquet(root / "panel" / "engine" / "validation_picks.parquet", index=False)


def run_world(kind):
    d = Path(tempfile.mkdtemp())
    try:
        make_world(kind, d)
        ET.run(d, verbose=False)
        EA.run(d, verbose=False)
        res = TA.run(d, verbose=False)
        et = json.loads((d / "panel" / "entry_timing" / "entry_timing.json").read_text())
        ea = json.loads((d / "panel" / "entry_timing" / "edge_anatomy.json").read_text())
        led = (d / "panel" / "research_ledger.jsonl")
        led_ok = any(json.loads(x).get("tool") == TA.CODE_VERSION
                     for x in led.read_text().splitlines()) if led.exists() else None
        return res, et, ea, led_ok
    finally:
        shutil.rmtree(d)


def _lo(res, v):
    return res["variants"][v]["net"]["lo"]


def common_checks(kind, res, et, ea, led_ok):
    check(res["check_A_vs_panel"] == 0.0 and res["check_C_vs_panel"] == 0.0,
          f"{kind}: recomputed A and C equal the panel labels exactly")
    check(res["check_exit_day_A_mismatches"] == 0, f"{kind}: A exit outcome = ft_A on every row")
    a_et = et["versions"]["A"]["topn"]["3"]["net"]["mean"]
    c_et = et["versions"]["C"]["topn"]["3"]["net"]["mean"]
    check(abs(res["variants"]["A_orig"]["net"]["mean"] - a_et) < 1e-12 and
          abs(res["variants"]["C_orig"]["net"]["mean"] - c_et) < 1e-12,
          f"{kind}: A_orig / C_orig reproduce entry_timing's top-3 net exactly")
    check(abs(res["ON1_excess_bp"] - ea["segments"]["ON1"]["excess"]["mean"] * 1e4) < 1e-9,
          f"{kind}: first-overnight excess reproduces edge_anatomy exactly")
    check(led_ok is True, f"{kind}: decision logged in the research ledger")


def test_worlds():
    print("7a. circuit world (the edge is locked - not buyable)")
    res, et, ea, led = run_world("circuit")
    common_checks("circuit", res, et, ea, led)
    fs = res["flag_share"]
    # market: 3 of 69 names lock at the close each day, and the frozen next day is
    # itself a close-lock -> 2 x 3/69 = 8.70%
    check(fs["uc_close"]["top3"] == 1.0 and abs(fs["uc_close"]["market"] - 6 / 69) < 0.003,
          f"every top-3 pick locked at the close; market {fs['uc_close']['market']:.2%} (planted 8.70%)")
    check(fs["open_uc"]["top3"] == 1.0 and fs["frozen1"]["top3"] == 1.0, "every top-3 pick opens frozen at the band")
    check(res["ON1_by_flag"]["uc_close"]["share_of_ON1_excess"] > 0.97,
          f"locked picks carry {res['ON1_by_flag']['uc_close']['share_of_ON1_excess']:.1%} of the ON1 excess")
    check(_lo(res, "A_orig") > 0, f"A as tested looks great: {res['variants']['A_orig']['net']['mean']*1e4:+.0f} bp")
    check(not res["alive"] and res["variants"]["A_sub"]["slots_replaced"] == 1.0,
          f"A_sub {res['variants']['A_sub']['net']['mean']*1e4:+.0f} bp -> DEAD (all slots replaced)")
    check(res["variants"]["A_sub_etf"]["slots_replaced"] == 0.0, "no ETFs here: ETF-only skip changes nothing")

    print("7b. clean world (a genuine, buyable overnight edge)")
    res, et, ea, led = run_world("clean")
    common_checks("clean", res, et, ea, led)
    fs = res["flag_share"]
    check(all(fs[k]["top3"] == 0.0 for k in fs) and all(fs[k]["market"] == 0.0 for k in fs),
          "nothing flagged anywhere")
    check(res["variants"]["A_sub"]["net"] == res["variants"]["A_orig"]["net"] and
          res["variants"]["C_sub"]["net"] == res["variants"]["C_orig"]["net"],
          "A_sub = A_orig and C_sub = C_orig exactly")
    check(res["alive"], f"A_sub {res['variants']['A_sub']['net']['mean']*1e4:+.0f} bp -> ALIVE")
    check(res["exit_locked"]["A_locked"] == 0 and res["exit_locked"]["C_locked"] == 0, "no locked exits")

    print("7c. ETF world (the edge sits in ETFs)")
    res, et, ea, led = run_world("etf")
    common_checks("etf", res, et, ea, led)
    check(res["flag_share"]["etf"]["top3"] == 1.0 and res["flag_share"]["uc_close"]["top3"] == 0.0,
          "every top-3 pick is an ETF, none locked")
    check(not res["alive"], f"A_sub {res['variants']['A_sub']['net']['mean']*1e4:+.0f} bp -> DEAD")
    check(_lo(res, "A_sub_circuit") > 0, "circuit-only skip keeps the edge (it was ETFs, not locks)")
    check(len(res["etf_symbols_picked"]) == 39, "all 39 ETF symbols listed for review")


def main():
    test_real_bars()
    test_hand_flags()
    test_etf_names()
    test_substitution()
    test_exit_day()
    test_exit_lock()
    test_worlds()
    print("\nVERIFIED  tradability_audit v1")


if __name__ == "__main__":
    main()
