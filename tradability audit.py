#!/usr/bin/env python3
"""
tradability_audit.py - could the picks actually have been BOUGHT (and sold)?

    python tradability_audit.py --root %CACHE_DAILY_ROOT%
    python tradability_audit.py --root %CACHE_DAILY_ROOT% --etf-list my_etfs.txt

WHY
===
edge_anatomy showed the picks earn +136 bp over the market in the first
overnight (close T -> open T+1). Its printed examples had first gaps of exactly
2% and 5% - NSE price-band sizes - and several frozen sessions (open = high =
low = close). A stock that CLOSES locked at its upper circuit has buyers and no
sellers: nobody could have bought it at that close, which is exactly the price
version A assumes. Version C has the same problem when the next session OPENS
at the upper band. And a stop "filled" on a session frozen at the LOWER band
could not have been sold. Two picks were also overseas-index ETFs (MON100,
MAFANG), whose overnight move is the US session, not stock selection.

FLAGS (circuit bands are not in the cache, so these are proxies from daily bars;
bands tested: 2 / 5 / 10 / 20 %; tolerance = max(0.15 pp, 1.02 ticks of 0.05 at
the reference price) to allow for the exchange rounding band prices to a tick)
  uc_close      signal day closes AT its high AND at an upper band vs the prior
                close. The close is the average of the last 30 minutes, so close
                = high means every late trade printed at the high: locked.
                -> UNBUYABLE AT THE CLOSE (version A).        [PRIMARY]
  uc_close_loose  close = high and the day's move >= +1.9%, any size (sensitivity)
  open_uc       T+1 opens AT its high AND at an upper band vs close T
                -> no sellers in the opening auction (version C).
  frozen1       open_uc and the whole of T+1 at one price (subset of open_uc)
  etf           ETF / non-equity by symbol pattern (+ optional --etf-list file)
  exit_locked   a stop exit (A or C) on a session frozen at the LOWER band:
                that fill could not happen. Descriptive only.

WHAT IS RECOMPUTED
  Every variant uses the engine's own walk-forward picks (validation_picks,
  ranks 1-20) and the same outcome code as entry_timing (A = close entry,
  barrier fills; C = next open, gap-aware fills), 35 bp cost.
  SUBSTITUTION: a live run sees the lock before ordering (at 15:30 for a close
  run, in the pre-open for a next-open run), so an unbuyable name is skipped
  and the next rank is taken. "Buy everything" drops the same flagged
  stock-days, because those could not be bought either.

THE RULE, FIXED BEFORE IT RUNS (declared 28 Sep 2026, before any audit result)
------------------------------
Close execution stays alive ONLY IF version A's top-3 net, with stocks locked
at the upper circuit at the signal close AND ETFs skipped and replaced by the
next rank (variant A_sub), has a 95% block-bootstrap interval above zero.
Everything else printed is descriptive.
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
import edge_anatomy as EA  # noqa: E402

CODE_VERSION = "tradability_audit v1"
BANDS = (0.02, 0.05, 0.10, 0.20)
TOL_PP = 0.0015
TICK = 0.05
TICK_SLACK = 1.02
H = ET.CFG["horizon"]
RULE = ("A_sub (skip upper-circuit-locked-at-close + ETFs, take next rank): "
        "top-3 net 95% interval above zero")

# ETF / non-equity symbols. Patterns catch the systematic names; the explicit
# list holds overseas and factor ETFs whose symbols carry no marker. The run
# prints the most-picked symbols so the list can be checked by eye; add more
# with --etf-list (one symbol per line, or a CSV with a SYMBOL column).
ETF_SUFFIXES = ("BEES", "ETF")
ETF_CONTAINS = ("ETF",)
ETF_PREFIXES = ("SETF",)
ETF_EXPLICIT = frozenset({
    "MON100", "N100", "MAFANG", "MASPTOP50", "MAHKTECH", "MONQ50", "MOM100", "MOM50",
    "ICICIB22", "NV20", "GOLDSHARE",
})

VARIANTS = {
    # name: (world, skip rule, unfilled slots)
    "A_orig": ("A", None, "sub"),
    "A_sub": ("A", "uc_close|etf", "sub"),                 # THE RULE
    "A_sub_circuit": ("A", "uc_close", "sub"),
    "A_sub_etf": ("A", "etf", "sub"),
    "A_sub_loose": ("A", "uc_close_loose|etf", "sub"),
    "C_orig": ("C", None, "sub"),
    "C_sub": ("C", "open_uc|etf", "sub"),
    "C_cash": ("C", "open_uc|etf", "cash"),
}
LABELS = {
    "A_orig": "A close entry, as tested",
    "A_sub": "A skip locked-at-close + ETFs  [RULE]",
    "A_sub_circuit": "A skip locked-at-close only",
    "A_sub_etf": "A skip ETFs only",
    "A_sub_loose": "A skip close=high>=+1.9% + ETFs",
    "C_orig": "C next open, as tested",
    "C_sub": "C skip opens-at-upper-band + ETFs",
    "C_cash": "C same, unfilled slot = cash",
}


def _log(msg):
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


# ----------------------------------------------------------------------
# flags
# ----------------------------------------------------------------------
def _eq(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(invalid="ignore"):
        return np.abs(a - b) <= 1e-6 * np.maximum(np.abs(a), np.abs(b))


def at_band(move, ref):
    """True where `move` (a fraction, signed toward the band) sits on a 2/5/10/20% band."""
    move, ref = np.asarray(move, float), np.asarray(ref, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tol = np.maximum(TOL_PP, TICK_SLACK * TICK / ref)
        hit = np.zeros(move.shape, dtype=bool)
        for b in BANDS:
            hit |= np.abs(move - b) <= tol
    return hit & np.isfinite(move) & np.isfinite(ref) & (ref > 0)


def symbol_flags(o, h, l, c) -> dict:
    """Per bar i (the signal day): locks at the close of i, at the open of i+1,
    and whether bar i itself is frozen at the lower band."""
    o, h, l, c = (np.asarray(x, dtype="float64") for x in (o, h, l, c))
    n = len(c)
    prev = np.full(n, np.nan)
    prev[1:] = c[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mv = c / prev - 1
    at_high = _eq(c, h)
    out = {"uc_close": at_high & at_band(mv, prev),
           "uc_close_loose": at_high & (np.nan_to_num(mv, nan=-1.0) >= 0.019),
           "frozen_lower": _eq(h, l) & at_band(-mv, prev)}
    o1, h1, l1 = (np.full(n, np.nan) for _ in range(3))
    o1[:-1], h1[:-1], l1[:-1] = o[1:], h[1:], l[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        g1 = o1 / c - 1
    out["open_uc"] = _eq(o1, h1) & at_band(g1, c)
    out["frozen1"] = out["open_uc"] & _eq(h1, l1)
    return out


def exit_day_A(o, h, l, c, cfg=ET.CFG):
    """Version A's exit: (session 1..H of the barrier hit, 0 = timeout; outcome
    +1 target / -1 stop / 0 timeout). Same rule as entry_timing.symbol_labels
    (tie -> stop); the outcome is checked against its ft_A on every row."""
    h, l, c = (np.asarray(x, dtype="float64") for x in (h, l, c))
    n = len(c)
    day, res = np.full(n, np.nan), np.full(n, np.nan)
    if n <= H + cfg["atr_n"]:
        return day, res
    tr = np.full(n, np.nan)
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    atr = pd.Series(tr).rolling(cfg["atr_n"], min_periods=cfg["atr_n"]).mean().to_numpy()
    idx = np.arange(0, n - H)
    a = atr[idx] / c[idx]
    tp, sl = cfg["tp_mult"] * a, cfg["sl_mult"] * a
    K = idx[:, None] + np.arange(1, H + 1)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        up = h[K] / c[idx][:, None] - 1 >= tp[:, None]
        dn = l[K] / c[idx][:, None] - 1 <= -sl[:, None]
    k_up = np.where(up.any(1), up.argmax(1), H)
    k_dn = np.where(dn.any(1), dn.argmax(1), H)
    win = (k_up < H) & (k_up < k_dn)
    d_ = np.where(win, k_up + 1, np.where(k_dn < H, k_dn + 1, 0))
    r_ = np.where(win, 1.0, np.where(k_dn < H, -1.0, 0.0))
    ok = np.isfinite(tp) & np.isfinite(sl) & (c[idx] > 0)
    day[idx], res[idx] = np.where(ok, d_, np.nan), np.where(ok, r_, np.nan)
    return day, res


def _load_etf_list(path) -> set:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--etf-list file not found: {p}")
    txt = p.read_text(encoding="utf-8-sig", errors="ignore")
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        col = next((c for c in df.columns if c.strip().upper() == "SYMBOL"), None)
        if col is None:
            raise SystemExit(f"{p} has no SYMBOL column")
        return {str(s).strip().upper() for s in df[col].dropna()}
    return {ln.strip().upper() for ln in txt.splitlines() if ln.strip() and not ln.startswith("#")}


def is_etf(sym: str, extra: set = frozenset()) -> bool:
    s = str(sym).upper()
    return (s in ETF_EXPLICIT or s in extra or s.endswith(ETF_SUFFIXES) or
            s.startswith(ETF_PREFIXES) or any(k in s for k in ETF_CONTAINS))


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def collect(root: Path, keys: pd.DataFrame, log=None) -> pd.DataFrame:
    """Segments, entry-timing outcomes, exit sessions and lock flags for the
    given (timestamp, symbol) rows, straight from the raw cache."""
    from data_quality import _paths
    parts = []
    syms = sorted(keys["symbol"].unique())
    for j, s in enumerate(syms):
        fp, _ = _paths(root, s)
        if not Path(fp).exists():
            continue
        d = pd.read_parquet(fp, columns=["timestamp", "open", "high", "low", "close"])
        d["timestamp"] = ET._naive(d["timestamp"])
        d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        seg = EA.symbol_segments(o, h, l, c)
        lab = ET.symbol_labels(o, h, l, c)
        fl = symbol_flags(o, h, l, c)
        kA, rA = exit_day_A(o, h, l, c)
        n = len(c)

        def locked_on(day):                      # frozen at the lower band on session i+day
            out = np.zeros(n, dtype=bool)
            ok = np.isfinite(day) & (day >= 1)
            i = np.where(ok)[0]
            tgt = i + day[ok].astype(int)
            inside = tgt < n
            out[i[inside]] = fl["frozen_lower"][tgt[inside]]
            return out

        f = pd.DataFrame({"timestamp": d["timestamp"], "symbol": s,
                          "ON1": seg["ON1"], "total": seg["total"],
                          "xr_A": lab["xr_A"], "ft_A": lab["ft_A"], "xr_C": lab["xr_C"], "ft_C": lab["ft_C"],
                          "day_A": kA, "res_A": rA, "day_C": lab["day_C"],
                          "uc_close": fl["uc_close"], "uc_close_loose": fl["uc_close_loose"],
                          "open_uc": fl["open_uc"], "frozen1": fl["frozen1"]})
        f["exit_locked_A"] = (f["ft_A"] == -1).to_numpy() & locked_on(kA)
        f["exit_locked_C"] = (f["ft_C"] == -1).to_numpy() & locked_on(lab["day_C"])
        want = set(keys.loc[keys["symbol"] == s, "timestamp"])
        parts.append(f[f["timestamp"].isin(want)])
        if log and (j + 1) % 250 == 0:
            log(f"    {j+1}/{len(syms)} symbols")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _skip_mask(df: pd.DataFrame, rule) -> pd.Series:
    if not rule:
        return pd.Series(False, index=df.index)
    m = pd.Series(False, index=df.index)
    for k in rule.split("|"):
        m |= df[k].astype(bool)
    return m


def select_top(V: pd.DataFrame, n: int, rule=None) -> pd.DataFrame:
    """Top-n per day after skipping flagged names (the next rank moves up)."""
    keep = V[~_skip_mask(V, rule)]
    return keep.sort_values(["timestamp", "rk"]).groupby("timestamp", sort=True).head(n)


def variant_days(V, S, variant, n=3, cost=0.0035):
    """Per-day net of the variant's top-n, and of buy-everything under the same skip."""
    world, rule, fill = VARIANTS[variant]
    x = f"xr_{world}"
    if fill == "cash":
        # names are substituted for the non-lock part of the rule; the lock itself
        # leaves the slot unfilled (0 return, no cost) instead of moving up a rank
        lock = rule.split("|")[0]
        other = "|".join(rule.split("|")[1:]) or None
        top = select_top(V, n, other)
        val = np.where(top[lock].astype(bool), 0.0, top[x] - cost)
        t = pd.Series(val, index=top.index).groupby(top["timestamp"]).mean()
    else:
        top = select_top(V, n, rule)
        t = (top[x] - cost).groupby(top["timestamp"]).mean()
    Su = S[~_skip_mask(S, rule)]
    uni = (Su[x] - cost).groupby(Su["timestamp"]).mean()
    return t, uni, top


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------
def run(root: Path, etf_list=None, cost_bps: float = 35.0, seed: int = 7, verbose: bool = True,
        write_ledger: bool = True) -> dict:
    extra = _load_etf_list(etf_list)
    vp = root / "panel" / "engine" / "validation_picks.parquet"
    if not vp.exists():
        raise SystemExit("no engine walk-forward picks - run: python signal_engine.py train")
    V0 = pd.read_parquet(vp, columns=["timestamp", "symbol", "rk"])
    V0["timestamp"] = ET._naive(V0["timestamp"])
    days = set(V0["timestamp"].unique())
    import pyarrow.parquet as pq
    names = pq.ParquetFile(root / "panel" / "panel.parquet").schema_arrow.names
    pcols = ["timestamp", "symbol"] + [c for c in ("label_exit_ret", "label_exit_ret_o1") if c in names]
    P = pd.read_parquet(root / "panel" / "panel.parquet", columns=pcols)
    P["timestamp"] = ET._naive(P["timestamp"])
    P = P[P["timestamp"].isin(days)]
    if verbose:
        _log(f"{len(days):,} walk-forward days | reading {P['symbol'].nunique():,} symbols from the cache")
    S = collect(root, P[["timestamp", "symbol"]], _log if verbose else None)
    S = S.merge(P, on=["timestamp", "symbol"], how="left")
    S["etf"] = S["symbol"].map(lambda s: is_etf(s, extra)).astype(bool)
    V = V0.merge(S, on=["timestamp", "symbol"], how="inner")
    cost = cost_bps / 1e4
    res = {"code": CODE_VERSION, "built_at": dt.datetime.now().isoformat(), "rule": RULE,
           "cost_bps": cost_bps, "days": len(days), "universe_rows": int(len(S)),
           "bands": list(BANDS), "tol_pp": TOL_PP, "tick": TICK,
           "etf_list_file": str(etf_list) if etf_list else None, "etf_extra": sorted(extra)}

    # 0. checks: recomputed labels vs panel; A exit session vs entry_timing's ft_A
    m = S["label_exit_ret"].notna() & S["xr_A"].notna() if "label_exit_ret" in S else None
    res["check_A_vs_panel"] = float(np.abs(S.loc[m, "label_exit_ret"] - S.loc[m, "xr_A"]).max()) if m is not None and m.any() else None
    if "label_exit_ret_o1" in S:
        m2 = S["label_exit_ret_o1"].notna() & S["xr_C"].notna()
        res["check_C_vs_panel"] = float(np.abs(S.loc[m2, "label_exit_ret_o1"] - S.loc[m2, "xr_C"]).max()) if m2.any() else None
    ok = S["ft_A"].notna() | S["res_A"].notna()
    res["check_exit_day_A_mismatches"] = int((S.loc[ok, "res_A"].fillna(9) != S.loc[ok, "ft_A"].fillna(9)).sum())

    top3 = V[V["rk"] <= 3]
    # 1. how many picks carry each flag, versus the market
    flags = ["uc_close", "uc_close_loose", "open_uc", "frozen1", "etf"]
    res["flag_share"] = {k: {"top3": float(top3[k].mean()), "market": float(S[k].mean())} for k in flags}
    st = top3[top3["ft_A"] == -1]
    sc = top3[top3["ft_C"] == -1]
    res["exit_locked"] = {"A_stop_exits": int(len(st)), "A_locked": int(st["exit_locked_A"].sum()),
                          "C_stop_exits": int(len(sc)), "C_locked": int(sc["exit_locked_C"].sum())}

    # 2. where does the first-overnight excess come from?
    mk = S.groupby("timestamp")[["ON1"]].mean().rename(columns={"ON1": "mk_ON1"})
    t3 = top3.merge(mk, left_on="timestamp", right_index=True, how="left")
    t3["ex_ON1"] = t3["ON1"] - t3["mk_ON1"]
    daily_ex = t3.groupby("timestamp")["ex_ON1"].mean()
    res["ON1_excess_bp"] = float(daily_ex.mean() * 1e4)
    tot = t3["ex_ON1"].sum()
    on1 = {}
    for k in ("uc_close", "open_uc", "etf", "any"):
        f_ = (t3["uc_close"] | t3["open_uc"] | t3["etf"]) if k == "any" else t3[k].astype(bool)
        on1[k] = {"picks": int(f_.sum()),
                  "share_of_ON1_excess": float(t3.loc[f_, "ex_ON1"].sum() / tot) if tot else float("nan"),
                  "flagged_ON1_bp": float(t3.loc[f_, "ON1"].mean() * 1e4) if f_.any() else float("nan"),
                  "unflagged_ON1_bp": float(t3.loc[~f_, "ON1"].mean() * 1e4) if (~f_).any() else float("nan")}
    res["ON1_by_flag"] = on1
    d_ac = (t3["xr_A"] - t3["xr_C"])
    okac = d_ac.notna()
    anyf = (t3["uc_close"] | t3["open_uc"] | t3["etf"])
    res["A_minus_C"] = {"mean_bp": float(d_ac[okac].mean() * 1e4),
                        "share_from_flagged": float(d_ac[okac & anyf].sum() / d_ac[okac].sum()) if d_ac[okac].sum() else float("nan"),
                        "top5pct_flagged_share": float(anyf[okac & (d_ac >= d_ac[okac].quantile(0.95))].mean())}

    # 3. the variants
    res["variants"] = {}
    for vname in VARIANTS:
        t, uni, top = variant_days(V, S, vname, 3, cost)
        ex = (t - uni.reindex(t.index)).dropna()
        base = top3.set_index(["timestamp", "symbol"]).index
        changed = float(1 - top.set_index(["timestamp", "symbol"]).index.isin(base).mean())
        short = int((top.groupby("timestamp").size() < 3).sum())
        yr = t.groupby(t.index.year).mean() * 1e4
        res["variants"][vname] = {
            "net": RC.block_bootstrap_mean(t.to_numpy(), seed=seed),
            "excess": RC.block_bootstrap_mean(ex.to_numpy(), seed=seed),
            "universe": RC.block_bootstrap_mean(uni.to_numpy(), seed=seed),
            "slots_replaced": changed, "days_short_of_3": short,
            "by_year_net_bp": {int(y): float(v) for y, v in yr.items()}}
    a = res["variants"]["A_sub"]["net"]
    res["alive"] = bool(np.isfinite(a["lo"]) and a["lo"] > 0)

    # 4. symbols to check by eye
    cnt = top3["symbol"].value_counts()
    res["most_picked"] = [{"symbol": s, "picks": int(k), "etf": bool(is_etf(s, extra))} for s, k in cnt.head(25).items()]
    res["etf_symbols_picked"] = {s: int(k) for s, k in cnt.items() if is_etf(s, extra)}

    # 5. reproduce earlier reports if they are on disk
    et_fp = root / "panel" / "entry_timing" / "entry_timing.json"
    ea_fp = root / "panel" / "entry_timing" / "edge_anatomy.json"
    rep = {}
    if et_fp.exists():
        e = json.loads(et_fp.read_text(encoding="utf-8"))
        for w in ("A", "C"):
            try:
                rep[f"{w}_top3_net_bp_entry_timing"] = e["versions"][w]["topn"]["3"]["net"]["mean"] * 1e4
            except (KeyError, TypeError):
                pass
    if ea_fp.exists():
        e = json.loads(ea_fp.read_text(encoding="utf-8"))
        try:
            rep["ON1_excess_bp_edge_anatomy"] = e["segments"]["ON1"]["excess"]["mean"] * 1e4
        except (KeyError, TypeError):
            pass
    rep["A_top3_net_bp_here"] = res["variants"]["A_orig"]["net"]["mean"] * 1e4
    rep["C_top3_net_bp_here"] = res["variants"]["C_orig"]["net"]["mean"] * 1e4
    rep["ON1_excess_bp_here"] = res["ON1_excess_bp"]
    res["reproduction"] = rep

    out = root / "panel" / "entry_timing"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tradability_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    cols = ["timestamp", "symbol", "rk", "uc_close", "uc_close_loose", "open_uc", "frozen1", "etf",
            "exit_locked_A", "exit_locked_C", "ON1", "xr_A", "ft_A", "xr_C", "ft_C"]
    fl = top3[top3[["uc_close", "uc_close_loose", "open_uc", "etf", "exit_locked_A", "exit_locked_C"]].any(axis=1)]
    fl[cols].sort_values(["timestamp", "rk"]).to_csv(out / "tradability_flagged_picks.csv", index=False)
    if write_ledger:
        RC.ledger_append(root / "panel" / "panel.parquet",
                         {"kind": "decision", "tool": CODE_VERSION, "rule": RULE,
                          "A_sub_top3_net": res["variants"]["A_sub"]["net"], "alive": res["alive"]})
    if verbose:
        _print(res)
    return res


def _f(e):
    return f"{e['mean']*1e4:+6.1f} [{e['lo']*1e4:+6.1f},{e['hi']*1e4:+6.1f}]"


def _print(res):
    print("\n" + "=" * 96)
    print(f"  TRADABILITY AUDIT - engine walk-forward picks, {res['days']:,} days, net of {res['cost_bps']:.0f} bp")
    print("=" * 96)
    ck = [f"A vs panel {res['check_A_vs_panel']:.1e}" if res.get("check_A_vs_panel") is not None else "A vs panel n/a"]
    if res.get("check_C_vs_panel") is not None:
        ck.append(f"C vs panel {res['check_C_vs_panel']:.1e}")
    ck.append(f"A exit-session mismatches {res['check_exit_day_A_mismatches']}")
    print("  checks: " + " | ".join(ck))
    r = res["reproduction"]
    print("  reproduces earlier runs: " + " | ".join(
        f"{k} {v:+.2f}" for k, v in r.items()))
    print(f"\n  {'flag':<52}{'top-3 picks':>14}{'market':>12}")
    desc = {"uc_close": "locked at upper band at the signal CLOSE (A can't buy)",
            "uc_close_loose": "close = high and move >= +1.9% (loose)",
            "open_uc": "T+1 opens at upper band (C can't buy)",
            "frozen1": "  of which T+1 frozen all day",
            "etf": "ETF / non-equity"}
    for k, v in res["flag_share"].items():
        print(f"  {desc[k]:<52}{v['top3']:>14.2%}{v['market']:>12.2%}")
    e = res["exit_locked"]
    print(f"  stop exits on a session frozen at the LOWER band (unfillable): "
          f"A {e['A_locked']}/{e['A_stop_exits']} | C {e['C_locked']}/{e['C_stop_exits']}")
    print(f"\n  first-overnight excess of the top-3: {res['ON1_excess_bp']:+.1f} bp. Where it comes from:")
    for k, v in res["ON1_by_flag"].items():
        print(f"    {k:<10} {v['picks']:>6,} picks | share of the ON1 excess {v['share_of_ON1_excess']:>7.1%} | "
              f"ON1 flagged {v['flagged_ON1_bp']:+8.1f} bp vs unflagged {v['unflagged_ON1_bp']:+7.1f} bp")
    a = res["A_minus_C"]
    print(f"  A minus C: mean {a['mean_bp']:+.1f} bp | share from flagged picks {a['share_from_flagged']:.1%} | "
          f"flagged among the largest 5% {a['top5pct_flagged_share']:.1%}")
    print(f"\n  {'variant':<40}{'top-3 net [95%]':>26}{'top-3 excess [95%]':>26}{'replaced':>10}")
    for k, v in res["variants"].items():
        print(f"  {LABELS[k]:<40}{_f(v['net']):>26}{_f(v['excess']):>26}{v['slots_replaced']:>10.1%}")
    print("  buy everything (same skip): " + " | ".join(
        f"{k} {v['universe']['mean']*1e4:+.1f}" for k, v in res["variants"].items()))
    print("\n  by year, top-3 net bp:")
    for k in ("A_orig", "A_sub", "C_orig", "C_sub"):
        print(f"    {k:<8} " + " | ".join(f"{y} {b:+.0f}" for y, b in res["variants"][k]["by_year_net_bp"].items()))
    print("\n  most-picked symbols (check for missed ETFs / non-equities; E = treated as ETF):")
    mp = res["most_picked"]
    print("    " + ", ".join(f"{x['symbol']}{' E' if x['etf'] else ''} {x['picks']}" for x in mp))
    if res["etf_symbols_picked"]:
        print("  ETF symbols among the top-3: " + ", ".join(f"{s} {k}" for s, k in res["etf_symbols_picked"].items()))
    print("\n" + "-" * 96)
    print(f"  RULE: {res['rule']}")
    print(f"  RESULT: A_sub top-3 net {_f(res['variants']['A_sub']['net'])} -> " +
          ("CLOSE EXECUTION STAYS ALIVE" if res["alive"]
           else "CLOSE EXECUTION IS DEAD: the overnight edge was not buyable"))
    print("-" * 96)
    print("  files: panel/entry_timing/tradability_audit.json, tradability_flagged_picks.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--etf-list", default=None, help="extra ETF symbols: one per line, or CSV with SYMBOL")
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT") or "")
    if not str(root) or str(root) == ".":
        raise SystemExit("CACHE_DAILY_ROOT not set and --root not given")
    run(root, etf_list=a.etf_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
