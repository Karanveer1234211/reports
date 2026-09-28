#!/usr/bin/env python3
"""
edge_anatomy.py - WHERE in the five days does the picks' return happen?

    python edge_anatomy.py --root %CACHE_DAILY_ROOT%
    python edge_anatomy.py --root %CACHE_DAILY_ROOT% --examples 20

The entry-timing test compared entry at the signal close (A) with entry at the
next open (C). That difference mixes TWO things: the first overnight move, and
how the rest of the path interacts with a bracket re-centred on the open (a
stock that gaps up and then fades can hit A's target before the fade but C's
stop after it). This report MEASURES instead of inferring.

1. SEGMENTS, NO BRACKETS. For the engine's walk-forward top-N picks and for
   every stock ("market"), the average return of each piece of the week:
      ON1  close T  -> open T+1        (the first overnight move)
      ID1  open T+1 -> close T+1       (session 1, open to close)
      ONk  close T+k-1 -> open T+k     (later overnight moves, k = 2..5)
      IDk  open T+k -> close T+k       (later sessions)
   Picks minus market per segment, with 95% block-bootstrap intervals, and the
   cumulative curve. Segments are log returns, so they ADD UP to close T ->
   close T+5 exactly.
2. REAL PICKS, PRINTED IN FULL: close, ATR, target/stop under A and C, every
   session's OHLC, which barrier hit when, both exit returns. Every printed
   outcome is recomputed here and must equal the panel's own labels.
3. A minus C, decomposed: how much tracks the first gap, and how often a pick
   reached A's target on session 1 while C stopped out.
4. OPEN-PRICE DATA CHECK: missing opens; opens exactly equal to the prior
   close (a possible data artefact).

Descriptive only. Reads the engine's validation_picks.parquet (every pick made
before its day). Writes panel/entry_timing/edge_anatomy.json and a CSV of the
printed example paths.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import research_common as RC  # noqa: E402
import entry_timing as ET  # noqa: E402

CODE_VERSION = "edge_anatomy v1"
H = 5
SEGS = ["ON1", "ID1"] + [f"{p}{k}" for k in range(2, H + 1) for p in ("ON", "ID")]


def _log(msg):
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def symbol_segments(o, h, l, c):
    """Log return of every segment of the five days after each bar."""
    o, c = np.asarray(o, float), np.asarray(c, float)
    n = len(c)
    out = {s: np.full(n, np.nan) for s in SEGS}
    out["total"] = np.full(n, np.nan)
    idx = np.arange(0, n - H)
    with np.errstate(invalid="ignore", divide="ignore"):
        lo, lc = np.log(o), np.log(c)
        out["ON1"][idx] = lo[idx + 1] - lc[idx]
        out["ID1"][idx] = lc[idx + 1] - lo[idx + 1]
        for k in range(2, H + 1):
            out[f"ON{k}"][idx] = lo[idx + k] - lc[idx + k - 1]
            out[f"ID{k}"][idx] = lc[idx + k] - lo[idx + k]
        out["total"][idx] = lc[idx + H] - lc[idx]
    return out


def collect(root: Path, keys: pd.DataFrame, log=None):
    """Segments + entry-timing labels + raw paths for the given (timestamp, symbol) rows."""
    from data_quality import _paths
    parts, paths = [], {}
    syms = sorted(keys["symbol"].unique())
    for j, s in enumerate(syms):
        fp, _ = _paths(root, s)
        if not Path(fp).exists():
            continue
        d = pd.read_parquet(fp, columns=["timestamp", "open", "high", "low", "close"])
        d["timestamp"] = ET._naive(d["timestamp"])
        d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        seg = symbol_segments(d["open"], d["high"], d["low"], d["close"])
        lab = ET.symbol_labels(d["open"], d["high"], d["low"], d["close"])
        prev = d["close"].shift(1)
        f = pd.DataFrame({"timestamp": d["timestamp"], "symbol": s, **seg,
                          "xr_A": lab["xr_A"], "ft_A": lab["ft_A"], "xr_C": lab["xr_C"], "ft_C": lab["ft_C"],
                          "open_missing": d["open"].isna().to_numpy(),
                          "open_eq_prev_close": np.isclose(d["open"].to_numpy(float), prev.to_numpy(float))})
        want = set(keys.loc[keys["symbol"] == s, "timestamp"])
        parts.append(f[f["timestamp"].isin(want)])
        paths[s] = d
        if log and (j + 1) % 250 == 0:
            log(f"    {j+1}/{len(syms)} symbols")
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), paths


def path_table(d: pd.DataFrame, t: pd.Timestamp) -> dict:
    """One real pick, every number needed to check its label by hand."""
    i = int(np.searchsorted(d["timestamp"].to_numpy(), np.datetime64(t)))
    if i + H >= len(d) or d["timestamp"].iloc[i] != t:
        return {}
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    tr = np.full(len(c), np.nan)
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    a = atr[i] / c[i]
    rows = [{"session": f"T+{k}", "date": str(d['timestamp'].iloc[i + k].date()), "open": o[i + k],
             "high": h[i + k], "low": l[i + k], "close": c[i + k]} for k in range(1, H + 1)]
    return {"close_T": c[i], "atr_pct": 100 * a, "tp_pct": 150 * a, "sl_pct": 100 * a,
            "A_target": c[i] * (1 + 1.5 * a), "A_stop": c[i] * (1 - a),
            "C_entry": o[i + 1], "C_target": o[i + 1] * (1 + 1.5 * a), "C_stop": o[i + 1] * (1 - a),
            "sessions": rows}


def run(root: Path, top_n: int = 3, examples: int = 12, cost_bps: float = 35.0, seed: int = 7,
        verbose: bool = True) -> dict:
    vp = root / "panel" / "engine" / "validation_picks.parquet"
    if not vp.exists():
        raise SystemExit("no engine walk-forward picks - run: python signal_engine.py train")
    V = pd.read_parquet(vp, columns=["timestamp", "symbol", "rk"])
    V["timestamp"] = ET._naive(V["timestamp"])
    days = sorted(V["timestamp"].unique())
    P = pd.read_parquet(root / "panel" / "panel.parquet",
                        columns=["timestamp", "symbol", "label_exit_ret"] +
                        (["label_exit_ret_o1"] if "label_exit_ret_o1" in
                         __import__("pyarrow.parquet", fromlist=["x"]).ParquetFile(
                             root / "panel" / "panel.parquet").schema_arrow.names else []))
    P["timestamp"] = ET._naive(P["timestamp"])
    P = P[P["timestamp"].isin(set(days))]
    if verbose:
        _log(f"{len(days):,} walk-forward days | collecting segments for {len(P):,} stock-days")
    S, paths = collect(root, P[["timestamp", "symbol"]], _log if verbose else None)
    S = S.merge(P, on=["timestamp", "symbol"], how="left")
    picks = V[V["rk"] <= top_n].merge(S, on=["timestamp", "symbol"], how="inner")
    res = {"code": CODE_VERSION, "built_at": dt.datetime.now().isoformat(), "top_n": top_n,
           "days": len(days), "picks": int(len(picks))}

    # 0. the recomputed labels must equal the panel's
    m = picks["label_exit_ret"].notna() & picks["xr_A"].notna()
    res["check_A_vs_panel"] = float(np.abs(picks.loc[m, "label_exit_ret"] - picks.loc[m, "xr_A"]).max())
    if "label_exit_ret_o1" in picks:
        m2 = picks["label_exit_ret_o1"].notna() & picks["xr_C"].notna()
        res["check_C_vs_panel"] = float(np.abs(picks.loc[m2, "label_exit_ret_o1"] - picks.loc[m2, "xr_C"]).max())

    # 1. segments: picks, market, picks minus market (per day, then bootstrap)
    uni = S.groupby("timestamp")[SEGS + ["total"]].mean()
    pk = picks.groupby("timestamp")[SEGS + ["total"]].mean()
    seg = {}
    for s_ in SEGS + ["total"]:
        ex = (pk[s_] - uni[s_].reindex(pk.index)).dropna()
        seg[s_] = {"picks_bp": float(pk[s_].mean() * 1e4), "market_bp": float(uni[s_].mean() * 1e4),
                   "excess": RC.block_bootstrap_mean(ex.to_numpy(), seed=seed)}
    res["segments"] = seg
    cum_p = np.cumsum([seg[s_]["picks_bp"] for s_ in SEGS])
    cum_x = np.cumsum([seg[s_]["excess"]["mean"] * 1e4 for s_ in SEGS])
    res["cumulative_picks_bp"] = dict(zip(SEGS, map(float, cum_p)))
    res["cumulative_excess_bp"] = dict(zip(SEGS, map(float, cum_x)))
    tot_ex = seg["total"]["excess"]["mean"]
    res["share_of_week_excess_in_first_overnight"] = (float(seg["ON1"]["excess"]["mean"] / tot_ex)
                                                      if tot_ex else float("nan"))
    later = [s_ for s_ in SEGS if s_ != "ON1"]
    res["excess_after_first_open_bp"] = float(sum(seg[s_]["excess"]["mean"] for s_ in later) * 1e4)
    yr = picks["timestamp"].dt.year
    res["by_year"] = {}
    for y, g in picks.groupby(yr):
        u = uni[uni.index.year == y]
        gp = g.groupby("timestamp")[SEGS + ["total"]].mean()
        res["by_year"][int(y)] = {
            "ON1_excess_bp": float((gp["ON1"] - u["ON1"].reindex(gp.index)).mean() * 1e4),
            "after_open_excess_bp": float(sum((gp[s_] - u[s_].reindex(gp.index)).mean() for s_ in later) * 1e4),
            "week_excess_bp": float((gp["total"] - u["total"].reindex(gp.index)).mean() * 1e4)}

    # 3. A minus C
    d_ac = picks["xr_A"] - picks["xr_C"]
    ok = d_ac.notna() & picks["ON1"].notna()
    res["A_minus_C_bp"] = float(d_ac[ok].mean() * 1e4)
    res["A_minus_C_corr_with_first_gap"] = float(np.corrcoef(d_ac[ok], picks.loc[ok, "ON1"])[0, 1]) if ok.sum() > 10 else float("nan")
    res["A_minus_C_quantiles_bp"] = {f"p{q}": float(np.percentile(d_ac[ok], q) * 1e4) for q in (5, 25, 50, 75, 95)}
    both = picks[ok]
    res["A_target_C_stop_share"] = float(((both["ft_A"] == 1) & (both["ft_C"] == -1)).mean())
    res["A_stop_C_target_share"] = float(((both["ft_A"] == -1) & (both["ft_C"] == 1)).mean())
    res["same_outcome_share"] = float((both["ft_A"] == both["ft_C"]).mean())

    # 4. open-price data check
    res["open_missing_share"] = float(S["open_missing"].mean())
    res["open_eq_prev_close_share_all"] = float(S["open_eq_prev_close"].mean())
    res["open_eq_prev_close_share_picks"] = float(picks["open_eq_prev_close"].mean())

    # 2. example paths: random picks + the largest A-C differences
    rng = np.random.default_rng(seed)
    ex_rows = pd.concat([picks.loc[ok].sample(min(examples, int(ok.sum())), random_state=seed),
                         picks.loc[ok].assign(_d=d_ac[ok]).nlargest(3, "_d")]).drop_duplicates(["timestamp", "symbol"])
    exs = []
    for _, r in ex_rows.iterrows():
        pt = path_table(paths[r["symbol"]], r["timestamp"])
        if not pt:
            continue
        pt.update({"symbol": r["symbol"], "signal_date": str(r["timestamp"].date()), "rank": int(r["rk"]),
                   "first_gap_pct": 100 * (np.exp(r["ON1"]) - 1),
                   "A_outcome": {1: "TARGET", -1: "STOP", 0: "TIMEOUT"}.get(r["ft_A"], "-"),
                   "A_return_pct": 100 * r["xr_A"],
                   "C_outcome": {1: "TARGET", -1: "STOP", 0: "TIMEOUT"}.get(r["ft_C"], "-"),
                   "C_return_pct": 100 * r["xr_C"],
                   "panel_A_pct": 100 * r["label_exit_ret"] if pd.notna(r["label_exit_ret"]) else None,
                   "panel_C_pct": (100 * r["label_exit_ret_o1"] if "label_exit_ret_o1" in r and
                                   pd.notna(r["label_exit_ret_o1"]) else None)})
        exs.append(pt)
    res["examples"] = exs
    out = root / "panel" / "entry_timing"
    out.mkdir(parents=True, exist_ok=True)
    (out / "edge_anatomy.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    flat = [{"symbol": e["symbol"], "signal_date": e["signal_date"], **s_, "A_target": e["A_target"],
             "A_stop": e["A_stop"], "C_entry": e["C_entry"], "C_target": e["C_target"], "C_stop": e["C_stop"]}
            for e in exs for s_ in e["sessions"]]
    pd.DataFrame(flat).to_csv(out / "edge_anatomy_examples.csv", index=False)
    if verbose:
        _print(res)
    return res


def _print(res):
    print("\n" + "=" * 92)
    print(f"  EDGE ANATOMY - engine walk-forward top-{res['top_n']} picks, {res['days']:,} days, "
          f"{res['picks']:,} picks (no brackets, no costs)")
    print("=" * 92)
    print(f"  check: recomputed A matches panel to {res['check_A_vs_panel']:.1e}" +
          (f", C to {res['check_C_vs_panel']:.1e}" if "check_C_vs_panel" in res else ""))
    print(f"\n  {'segment':<34}{'picks bp':>10}{'market bp':>11}{'picks - market bp [95%]':>28}")
    names = {"ON1": "ON1  close T -> open T+1", "ID1": "ID1  open T+1 -> close T+1"}
    for s_ in SEGS:
        r = res["segments"][s_]
        e = r["excess"]
        lab = names.get(s_, f"{s_}  {'overnight' if s_.startswith('ON') else 'session'} {s_[2:]}")
        print(f"  {lab:<34}{r['picks_bp']:>+10.1f}{r['market_bp']:>+11.1f}"
              f"{e['mean']*1e4:>+12.1f} [{e['lo']*1e4:+6.1f},{e['hi']*1e4:+6.1f}]")
    t = res["segments"]["total"]
    print(f"  {'WEEK  close T -> close T+5':<34}{t['picks_bp']:>+10.1f}{t['market_bp']:>+11.1f}"
          f"{t['excess']['mean']*1e4:>+12.1f} [{t['excess']['lo']*1e4:+6.1f},{t['excess']['hi']*1e4:+6.1f}]")
    wk = res["segments"]["total"]["excess"]["mean"] * 1e4
    share = (f"{res['share_of_week_excess_in_first_overnight']:.0%}" if abs(wk) >= 10
             else "n/a (the week's excess is ~0)")
    print(f"\n  share of the week's excess earned in the FIRST overnight: {share} | "
          f"excess after the first open: {res['excess_after_first_open_bp']:+.1f} bp")
    print("  cumulative excess bp: " + " ".join(f"{k} {v:+.0f}" for k, v in res["cumulative_excess_bp"].items()))
    print("\n  by year (excess bp): " + " | ".join(
        f"{y}: gap {d['ON1_excess_bp']:+.0f}, after open {d['after_open_excess_bp']:+.0f}"
        for y, d in res["by_year"].items()))
    q = res["A_minus_C_quantiles_bp"]
    print(f"\n  A minus C per pick: mean {res['A_minus_C_bp']:+.0f} bp | median {q['p50']:+.0f} | "
          f"p5 {q['p5']:+.0f} / p95 {q['p95']:+.0f} | correlation with the first gap "
          f"{res['A_minus_C_corr_with_first_gap']:+.2f}")
    print(f"  same outcome under A and C: {res['same_outcome_share']:.0%} | A target but C stop: "
          f"{res['A_target_C_stop_share']:.0%} | A stop but C target: {res['A_stop_C_target_share']:.0%}")
    print(f"\n  open-price data: missing {res['open_missing_share']:.2%} | open exactly = previous close: "
          f"all stocks {res['open_eq_prev_close_share_all']:.2%}, picks {res['open_eq_prev_close_share_picks']:.2%}")
    for e in res["examples"][:4]:
        print(f"\n  EXAMPLE {e['symbol']} signal {e['signal_date']} (rank {e['rank']}): close {e['close_T']:.2f}, "
              f"ATR {e['atr_pct']:.2f}% -> A target {e['A_target']:.2f} / stop {e['A_stop']:.2f} | "
              f"next open {e['C_entry']:.2f} ({e['first_gap_pct']:+.2f}%) -> C target {e['C_target']:.2f} / "
              f"stop {e['C_stop']:.2f}")
        for s_ in e["sessions"]:
            print(f"      {s_['session']} {s_['date']}  O {s_['open']:.2f}  H {s_['high']:.2f}  "
                  f"L {s_['low']:.2f}  C {s_['close']:.2f}")
        print(f"      A: {e['A_outcome']} {e['A_return_pct']:+.2f}% (panel {e['panel_A_pct']:+.2f}%) | "
              f"C: {e['C_outcome']} {e['C_return_pct']:+.2f}%"
              + (f" (panel {e['panel_C_pct']:+.2f}%)" if e.get("panel_C_pct") is not None else ""))
    print(f"\n  all {len(res['examples'])} example paths: panel/entry_timing/edge_anatomy_examples.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT") or "")
    if not str(root):
        raise SystemExit("CACHE_DAILY_ROOT not set and --root not given")
    run(root, top_n=a.top, examples=a.examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
