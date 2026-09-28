#!/usr/bin/env python3
"""
feature_atlas.py - which signals work, in which regime. Exploratory only.

    python feature_atlas.py run    --root <cache_root>
    python feature_atlas.py run    --root <cache_root> --resume ATLAS_20260925_001
    python feature_atlas.py report --root <cache_root> --run ATLAS_20260925_001
    python feature_atlas.py list   --root <cache_root>

WHAT THIS IS - AND IS NOT
=========================
An ATLAS. It generates a broad feature library, assigns every stock-day a
regime, and measures how well each feature ranks stocks against the target
inside each regime. Nothing here feeds a model. That is what makes a large
pool acceptable: the risk of a big pool is false discoveries leaking into a
model, and here there is no model to leak into - provided the statistics stay
honest about how many things were tried.

THE MEASURE: PER-DATE RANK IC
-----------------------------
For each date, rank the feature across stocks and correlate with the target.
One number per date; the date is the unit of observation. This is the right
metric because your system picks top names EACH DAY - a cross-sectional
decision - and because pooling rows would count 1,500 stocks on one day as
1,500 independent observations when they share a market move.

With a binary target, Spearman within a date equals Pearson between rank(f)
and y, since ranking a binary variable is an affine transform of it. So only
the feature is ranked - roughly half the compute.

HONESTY ABOUT HOW MANY THINGS WERE TRIED
----------------------------------------
    * The t-statistic divides by sqrt(n_dates / horizon): 5-session labels
      overlap, so consecutive daily ICs are not independent.
    * Benjamini-Hochberg false-discovery control across EVERY
      (feature x layer x regime) cell tested. With ~400 features x ~15
      cells, thousands of tests run; q-values say how many "discoveries"
      are expected to be noise.
    * Sign stability across walk-forward folds: significant AND the same sign
      in at least 80% of folds with data.
    * Rare regimes get this automatically - fewer dates means a larger
      standard error means a harder bar.
    * Working features are clustered into families, so forty representations
      of one volatility signal are reported as one finding, not forty.

REGIMES
-------
Two layers, both discovered and never predefined:
    STOCK   Gaussian mixture on per-date ranks of stock state
    MARKET  Gaussian mixture on the per-date market state vector
K is chosen by BIC on the first fold's training window, then HELD FIXED so a
regime ID means the same thing across folds; later folds refit and are
aligned to the reference by Hungarian matching on centroids. Alignment drift
is reported. Every row receives a regime: rows before the first test window
get the first engine's label and are flagged in-sample; only test-window rows
enter the IC statistics.

WHY NO STOCK x MARKET INTERACTIONS
----------------------------------
Within a single date a market variable is a constant, so momentum x
market_trend just rescales - or flips - the momentum ranking; its per-date IC
is plus or minus momentum's IC. The question it asks, "does momentum work
differently in different market states", is answered directly by the MARKET
layer. Stock x stock interactions are kept.

MEMORY
------
Features are written to column-major memory-mapped files and read back one
column at a time, so peak memory is set by the row count, not the feature
count. A previous design widened a pandas frame to ~600 columns x 1.95M rows
in float64 - ~9 GB - which is the allocation this replaces.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CODE_VERSION = "feature_atlas v5"

import research_common as RC

DEFAULTS = {
    "target": "label_tp_before_sl",
    "horizon": 5,
    "n_splits": 5,
    "k_range_stock": [3, 4, 5, 6, 7, 8, 9, 10],
    "k_range_market": [2, 3, 4, 5, 6],
    "gmm_rows": 150_000,
    "gmm_select_rows": 40_000,
    "min_names": 15,           # stocks per (date, regime) for a daily IC
    "min_dates": 60,           # dates per cell to report it
    "fdr_q": 0.05,
    "min_fold_sign": 0.80,
    "min_abs_ic": 0.010,
    "family_corr": 0.80,
    "all_representations": False,
    "seed": 0,
    # SHARED with regime_research via research_common: the atlas must never
    # look at the lockbox either, or regime_research's lockbox is no longer
    # untouched - its statistics would already have been seen here.
    "lockbox_fraction": RC.DEFAULT_LOCKBOX_FRACTION,
}

# Signals that get the full set of alternative representations. Whichever
# exist in the panel are used; absent ones are skipped and logged.
CORE_SIGNALS = [
    "D_ema20_angle_deg", "D_adx14", "D_rsi14", "D_rsi7", "D_macd_hist",
    "D_atr_pct", "D_realvol_20", "D_realvol_ratio_20_60", "D_bb_bw_20",
    "D_atr_ratio_14_30", "D_dvol_z20", "D_pos_in_52w_range",
    "D_dist_from_20h", "D_donch_pos_20", "D_bb_pctB_20",
    "D_intraday_ret_pct", "D_gap_pct", "D_obv_slope", "D_cmf20",
    "D_WQ_38", "D_compress_state", "D_drawdown_252", "D_macd_hist_rstd10",
]

# Canonical state lives in research_common; this alias is read-only.
STOCK_STATE = {n: c for n, c, _ in RC.STOCK_STATE_SPEC}


# ======================================================================
# RUN MANAGEMENT (shared with regime_research)
# ======================================================================
def _run_cls():
    import regime_research as RR
    return RR.Run, RR.RunConflict, RR.fingerprint


def new_run_id(base: Path) -> str:
    base.mkdir(parents=True, exist_ok=True)
    day = dt.date.today().strftime("%Y%m%d")
    n = 1 + max([int(p.name.rsplit("_", 1)[-1]) for p in base.glob(f"ATLAS_{day}_*")
                 if p.name.rsplit("_", 1)[-1].isdigit()] or [0])
    return f"ATLAS_{day}_{n:03d}"


# ======================================================================
# CORE FRAME - the only thing held in memory for the whole run
# ======================================================================
class Core:
    """
    Row order, keys and target. Everything else is read from disk on demand.

    Canonical order is (timestamp, symbol). Within one symbol that is also
    time order, which is what the per-symbol rolling operations need; within
    one date it is contiguous, which is what the per-date IC needs.
    """

    def __init__(self, panel_path: Path, target: str):
        self.path = panel_path
        keys = pd.read_parquet(panel_path, columns=["timestamp", "symbol"])
        ts = pd.to_datetime(keys["timestamp"]).dt.tz_localize(None).to_numpy()
        sym = keys["symbol"].astype(str).to_numpy()
        self.perm = np.lexsort((sym, ts))
        self.ts = ts[self.perm]
        self.dates, self.date_code = np.unique(self.ts, return_inverse=True)
        self.symbols, self.sym_code = np.unique(sym[self.perm], return_inverse=True)
        self._seg = None
        self.n = len(self.perm)
        self.y = self.read(target)
        self._cols = set(self._schema())

    def _schema(self) -> List[str]:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(self.path).schema_arrow.names)

    def has(self, c: str) -> bool:
        return c in self._cols

    def read(self, c: str) -> np.ndarray:
        """One column, float32, canonical order."""
        v = pd.read_parquet(self.path, columns=[c])[c]
        return pd.to_numeric(v, errors="coerce").to_numpy(dtype="float32")[self.perm]


# ======================================================================
# FEATURE PRIMITIVES (all trailing or same-date: point-in-time)
# ======================================================================
def _s(v) -> pd.Series:
    return pd.Series(np.asarray(v, dtype="float64"))


def _seg(core: Core) -> np.ndarray:
    """Unbroken session runs: no per-stock window may cross a liquidity gap."""
    if core._seg is None:
        core._seg = RC.session_segments(core.date_code, core.sym_code)
    return core._seg


def by_sym_shift(core: Core, v, k: int) -> np.ndarray:
    return _s(v).groupby(_seg(core)).shift(k).to_numpy()


def by_sym_roll(core: Core, v, w: int, fn: str, minp: Optional[int] = None) -> np.ndarray:
    r = getattr(_s(v).groupby(_seg(core)).rolling(w, min_periods=minp or max(3, w // 2)), fn)()
    return r.reset_index(level=0, drop=True).sort_index().to_numpy()


def by_sym_roll_corr(core: Core, a, b, w: int) -> np.ndarray:
    a = np.asarray(a, "float64"); b = np.asarray(b, "float64")
    ma = by_sym_roll(core, a, w, "mean"); mb = by_sym_roll(core, b, w, "mean")
    mab = by_sym_roll(core, a * b, w, "mean")
    va = by_sym_roll(core, a * a, w, "mean") - ma * ma
    vb = by_sym_roll(core, b * b, w, "mean") - mb * mb
    return (mab - ma * mb) / np.sqrt(np.clip(va * vb, 1e-18, None))


def by_date_rank(core: Core, v) -> np.ndarray:
    return _s(v).groupby(core.date_code).rank(pct=True).to_numpy() - 0.5


def by_date_z(core: Core, v) -> np.ndarray:
    s = _s(v); g = s.groupby(core.date_code)
    return ((s - g.transform("mean")) / g.transform("std")).to_numpy()


def per_date(core: Core, series_by_date: np.ndarray) -> np.ndarray:
    return series_by_date[core.date_code]


# ======================================================================
# FEATURE LIBRARY - a PLAN of specs, allocated then filled column by column
# ======================================================================
class Spec:
    def __init__(self, name, group, category, fn: Callable[[], np.ndarray], note=""):
        self.name, self.group, self.category, self.fn, self.note = \
            name, group, category, fn, note


def build_plan(core: Core, cfg: dict, cache: dict) -> List[Spec]:
    """
    Every feature the atlas will generate, as a lazy spec.

    Specs are evaluated one at a time and written straight to disk, so the
    library can be as large as wanted without growing memory.
    """
    plan: List[Spec] = []
    def C(c):
        # NOT cache.setdefault(c, core.read(c)): Python evaluates the default
        # argument BEFORE setdefault looks in the cache, so that form re-reads
        # the parquet column on every call and the cache never saves anything.
        if c not in cache:
            cache[c] = core.read(c)
        return cache[c]

    # --- 1. every existing panel feature, as-is
    import panel_build as PB
    schema = core._schema()
    skip = set(PB.PANEL_LABELS) | {"timestamp", "symbol", "X_turnover_med"}
    raw_cols = {"open", "high", "low", "close", "volume"}
    panel_feats = [c for c in schema
                   if c not in skip and c not in raw_cols
                   and not RC.is_forbidden_feature(c)]
    for c in panel_feats:
        plan.append(Spec(c, "panel", "existing", (lambda c=c: core.read(c))))

    # --- 2. alternative REPRESENTATIONS of core signals
    reps_for = panel_feats if cfg["all_representations"] else \
        [c for c in CORE_SIGNALS if c in schema]
    for c in reps_for:
        base = c.replace("D_", "")
        plan += [
            Spec(f"R_{base}__csrank", "repr", "representation",
                 (lambda c=c: by_date_rank(core, C(c))), "per-date rank"),
            Spec(f"R_{base}__csz", "repr", "representation",
                 (lambda c=c: by_date_z(core, C(c))), "per-date z-score"),
            Spec(f"R_{base}__tsz60", "repr", "representation",
                 (lambda c=c: (C(c) - by_sym_roll(core, C(c), 60, "mean"))
                  / by_sym_roll(core, C(c), 60, "std")), "vs own 60d"),
            Spec(f"R_{base}__d5", "repr", "representation",
                 (lambda c=c: C(c) - by_sym_shift(core, C(c), 5)), "5-session change"),
            Spec(f"R_{base}__d20", "repr", "representation",
                 (lambda c=c: C(c) - by_sym_shift(core, C(c), 20)), "20-session change"),
            Spec(f"R_{base}__accel", "repr", "representation",
                 (lambda c=c: (C(c) - by_sym_shift(core, C(c), 5))
                  - (by_sym_shift(core, C(c), 5) - by_sym_shift(core, C(c), 10))),
                 "change of change"),
        ]

    # --- 3. NEW structural / distributional features from raw OHLCV
    if all(core.has(c) for c in ("open", "high", "low", "close", "volume")):
        def ret1():
            if "ret1" not in cache:
                cl = C("close")
                cache["ret1"] = cl / by_sym_shift(core, cl, 1) - 1
            return cache["ret1"]

        def rng_():
            return C("high") - C("low")

        S = [
            ("N_skew20", "distribution", lambda: by_sym_roll(core, ret1(), 20, "skew")),
            ("N_kurt20", "distribution", lambda: by_sym_roll(core, ret1(), 20, "kurt")),
            ("N_max_ret20", "distribution", lambda: by_sym_roll(core, ret1(), 20, "max")),
            ("N_min_ret20", "distribution", lambda: by_sym_roll(core, ret1(), 20, "min")),
            ("N_autocorr20", "distribution",
             lambda: by_sym_roll_corr(core, ret1(), by_sym_shift(core, ret1(), 1), 20)),
            ("N_volofvol20", "volatility",
             lambda: by_sym_roll(core, by_sym_roll(core, ret1(), 20, "std"), 20, "std")),
            ("N_up_vol_ratio20", "volume",
             lambda: by_sym_roll(core, np.where(ret1() > 0, C("volume"), 0.0), 20, "sum")
             / by_sym_roll(core, C("volume"), 20, "sum")),
            ("N_pv_corr20", "volume",
             lambda: by_sym_roll_corr(core, ret1(),
                                      C("volume") / by_sym_shift(core, C("volume"), 1) - 1, 20)),
            ("N_clv", "structure",
             lambda: ((C("close") - C("low")) - (C("high") - C("close")))
             / np.where(rng_() > 0, rng_(), np.nan)),
            ("N_range_expansion", "volatility",
             lambda: rng_() / by_sym_roll(core, rng_(), 20, "mean")),
            ("N_body_range", "structure",
             lambda: (C("close") - C("open")) / np.where(rng_() > 0, rng_(), np.nan)),
            ("N_gap_filled", "structure", lambda: _gap_filled(core, C)),
            ("N_streak", "momentum", lambda: _streak(core, ret1())),
        ]
        for n, w in (("10", 10), ("50", 50), ("100", 100), ("250", 250)):
            S.append((f"N_dist_hi{n}", "location",
                      (lambda w=w: C("close") / by_sym_roll(core, C("high"), w, "max") - 1)))
            S.append((f"N_dist_lo{n}", "location",
                      (lambda w=w: C("close") / by_sym_roll(core, C("low"), w, "min") - 1)))
        for h in (3, 10, 60, 120):
            S.append((f"N_ret{h}", "momentum",
                      (lambda h=h: C("close") / by_sym_shift(core, C("close"), h) - 1)))
        for n, cat, fn in S:
            plan.append(Spec(n, "structure", cat, fn))

        # --- 4. market-relative (stock minus market; varies across stocks)
        def mkt():
            if "mkt" not in cache:
                cache["mkt"] = RC.market_state(core.read("close"), core.date_code,
                                               _seg(core), len(core.dates))
            return cache["mkt"]

        def mret_row():
            return per_date(core, mkt()["mk_ret1"])

        def beta_ivol():
            # beta and idiosyncratic vol share five rolling regressions;
            # compute them once for both features
            if "beta_ivol" not in cache:
                cache["beta_ivol"] = _beta(core, ret1(), mret_row(), 60)
            return cache["beta_ivol"]

        M = [
            ("M_rel_ret5", lambda: _rel(core, ret1(), mret_row(), 5)),
            ("M_rel_ret20", lambda: _rel(core, ret1(), mret_row(), 20)),
            ("M_beta60", lambda: beta_ivol()[0]),
            ("M_ivol60", lambda: beta_ivol()[1]),
            ("M_corr60", lambda: by_sym_roll_corr(core, ret1(), mret_row(), 60)),
        ]
        for n, fn in M:
            plan.append(Spec(n, "market_rel", "market-relative", fn))

    # --- 5. controlled stock x stock interactions (per-date ranks, PIT)
    def rk(c):
        return by_date_rank(core, C(c)) if core.has(c) else None

    pairs = [
        ("D_rsi14", "D_realvol_20", "momentum x volatility"),
        ("D_dvol_z20", "D_ema20_angle_deg", "volume x trend"),
        ("D_compress_state", "D_dist_from_20h", "compression x breakout proximity"),
        ("D_obv_slope", "D_ema20_angle_deg", "OBV slope x price slope"),
        ("D_rsi14", "D_adx14", "momentum x trend strength"),
        ("D_realvol_ratio_20_60", "D_pos_in_52w_range", "vol change x location"),
        ("D_dvol_z20", "D_realvol_ratio_20_60", "volume x vol change"),
        ("D_macd_hist", "D_atr_pct", "MACD x volatility"),
        ("D_intraday_ret_pct", "D_dvol_z20", "day move x volume"),
        ("D_bb_pctB_20", "D_adx14", "band position x trend strength"),
    ]
    for a, b, note in pairs:
        if core.has(a) and core.has(b):
            plan.append(Spec(f"I_{a[2:]}__x__{b[2:]}", "interaction", "interaction",
                             (lambda a=a, b=b: rk(a) * rk(b)), note))
    RC.assert_no_label_leak([sp.name for sp in plan], "feature_atlas.build_plan")
    return plan


def _gap_filled(core, C):
    pc = by_sym_shift(core, C("close"), 1)
    gap = C("open") / pc - 1
    up = (gap > 0) & (C("low") <= pc)
    dn = (gap < 0) & (C("high") >= pc)
    return np.where(np.isnan(gap), np.nan, (up | dn).astype("float64"))


def _streak(core, r):
    """Signed run length of consecutive up/down closes, per symbol."""
    s = np.sign(np.nan_to_num(r, nan=0.0))
    # runs break wherever the sign or the symbol changes
    df = pd.DataFrame({"c": _seg(core), "s": s})
    brk = (df["s"] != df.groupby("c")["s"].shift()).cumsum()
    run = df.groupby(brk).cumcount() + 1
    return (run * df["s"]).to_numpy().astype("float64")


def _rel(core, r, m, w):
    return (by_sym_roll(core, r, w, "sum") - by_sym_roll(core, m, w, "sum"))


def _beta(core, r, m, w):
    er = by_sym_roll(core, r, w, "mean"); em = by_sym_roll(core, m, w, "mean")
    erm = by_sym_roll(core, r * m, w, "mean")
    vm = by_sym_roll(core, m * m, w, "mean") - em * em
    vr = by_sym_roll(core, r * r, w, "mean") - er * er
    beta = (erm - er * em) / np.where(vm > 1e-12, vm, np.nan)
    ivol = np.sqrt(np.clip(vr - beta * beta * vm, 0, None))
    return beta, ivol


# ======================================================================
# FEATURE STORE - column-major memmaps on disk
# ======================================================================
class Store:
    def __init__(self, run, n_rows: int, research: Optional[np.ndarray] = None):
        self.run, self.n = run, n_rows
        self.research = research if research is not None else np.ones(n_rows, bool)
        self.reg_path = run.path("features", "registry.json")
        self.registry = json.loads(self.reg_path.read_text(encoding="utf-8")) \
            if self.reg_path.exists() else {}

    def write_group(self, group: str, specs: List[Spec], log) -> None:
        path = self.run.path("features", f"{group}.npy")
        # Fortran order: each COLUMN is contiguous on disk, so reading one
        # feature back is a single sequential read rather than a stride
        # across every row's page.
        mm = np.lib.format.open_memmap(path, mode="w+", dtype="float32",
                                       shape=(self.n, len(specs)),
                                       fortran_order=True)
        t0 = time.perf_counter()
        for j, sp in enumerate(specs):
            try:
                v = np.asarray(sp.fn(), dtype="float64")
                v[~np.isfinite(v)] = np.nan
                mm[:, j] = v.astype("float32")
                # coverage decides which features ENTER - measure it on
                # research rows only, never the lockbox
                cov = float(np.isfinite(v[self.research]).mean())
                err = None
            except Exception as e:
                mm[:, j] = np.nan
                cov, err = 0.0, f"{type(e).__name__}: {e}"[:160]
            self.registry[sp.name] = {"group": group, "col": j,
                                      "category": sp.category, "note": sp.note,
                                      "coverage": cov, "error": err}
            if (j + 1) % 25 == 0:
                log(f"      {group}: {j+1}/{len(specs)} "
                    f"({time.perf_counter()-t0:.0f}s)")
        mm.flush()
        del mm
        self.reg_path.write_text(json.dumps(self.registry, indent=1),
                                 encoding="utf-8")

    def read(self, name: str) -> np.ndarray:
        r = self.registry[name]
        mm = np.load(self.run.path("features", f"{r['group']}.npy"), mmap_mode="r")
        return np.asarray(mm[:, r["col"]])


# ======================================================================
# REGIMES - discovered, fixed K, aligned across folds
# ======================================================================
def _subsample(X, rows, seed):
    ok = np.where(np.isfinite(X).all(axis=1))[0]
    rng = np.random.default_rng(seed)
    return rng.choice(ok, min(rows, len(ok)), replace=False)


def _gmm(X, k, rows, seed):
    return RC.fit_state_gmm(X, k, _subsample(X, rows, seed), seed)


def _choose_k(X, k_range, rows, seed):
    return RC.choose_k_bic(X, k_range, _subsample(X, rows, seed), seed)


_align = RC.align_components


def discover_regimes(core: Core, X: np.ndarray, row_dates: np.ndarray,
                     splits, k_range, cfg, label: str, log) -> dict:
    """
    Every row gets a regime from the latest engine fitted strictly before it.

    Rows on or before the first train_end get the first engine's label and
    are flagged in_sample: they describe the data but never enter the IC
    statistics.
    """
    train_ends = [np.datetime64(pd.Timestamp(s["train_end"])) for s in splits]
    tr0 = row_dates <= train_ends[0]
    K, bic = _choose_k(X[tr0], k_range, cfg["gmm_select_rows"], cfg["seed"])
    log(f"    {label}: K={K} by BIC on fold-1 train "
        f"({', '.join(f'{k}:{v:.0f}' for k, v in bic.items())})")
    engines, drift = [], []
    ref = None
    for fi, te in enumerate(train_ends):
        gm = _gmm(X[row_dates <= te], K, cfg["gmm_rows"], cfg["seed"])
        if ref is None:
            ref = gm.means_.copy()
            perm = np.arange(K)
            d = 0.0
        else:
            perm, d = _align(ref, gm.means_)
        engines.append((te, gm, perm))
        drift.append(d)
    lab = np.full(len(X), -1, dtype="int16")
    conf = np.zeros(len(X), dtype="float32")
    in_sample = row_dates <= train_ends[0]
    for i, (te, gm, perm) in enumerate(engines):
        # NO FAR-FUTURE SENTINEL. v4 bounded the last engine with
        # np.datetime64("9999-01-01"); against NANOSECOND timestamps that
        # comparison is silently False (year 9999 does not fit in ns), so on
        # the real panel every row after the last train_end - the whole
        # final test fold - got no regime. The fixtures use microseconds,
        # where it happens to work. The last engine is simply unbounded.
        last = i + 1 == len(engines)
        m = np.ones(len(row_dates), bool) if (i == 0 and last) else (
            (row_dates > te) if last else
            ((row_dates <= engines[i + 1][0]) if i == 0 else
             (row_dates > te) & (row_dates <= engines[i + 1][0])))
        if not m.any():
            continue
        # scored on OBSERVED axes only - v1 imputed 0, i.e. "average", for
        # inputs that were simply unknown
        pr = RC.gmm_proba(gm, X[m])
        lab[m] = perm[pr.argmax(axis=1)]
        conf[m] = pr.max(axis=1)
    centroids = {int(k): list(map(float, ref[k])) for k in range(K)}
    miss = np.isnan(X).any(axis=1)
    return {"label": lab, "conf": conf, "in_sample": in_sample, "K": K,
            "bic": bic, "drift": drift, "centroids": centroids,
            "missing": {"pct_rows_any_missing": float(miss.mean()),
                        "mean_conf_complete": float(conf[~miss].mean()) if (~miss).any() else float("nan"),
                        "mean_conf_partial": float(conf[miss].mean()) if miss.any() else float("nan")}}


# ======================================================================
# ATLAS EVALUATION - per-date rank IC, vectorised
# ======================================================================
def _group_ic(f: np.ndarray, y: np.ndarray, g: np.ndarray, n_groups: int,
              min_n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spearman(f, y) within every group, where y is binary.

    Ranking a binary variable is an affine transform of it, so Spearman
    equals Pearson(rank(f), y). Only f is ranked; the correlation comes from
    bincount sums. Also returns the top-minus-bottom quintile hit rate.
    """
    ok = np.isfinite(f) & np.isfinite(y) & (g >= 0)
    ff, yy, gg = f[ok], y[ok], g[ok]
    rx = pd.Series(ff).groupby(gg).rank(method="average").to_numpy()
    n = np.bincount(gg, minlength=n_groups).astype("float64")
    sx = np.bincount(gg, rx, n_groups); sy = np.bincount(gg, yy, n_groups)
    sxx = np.bincount(gg, rx * rx, n_groups); syy = np.bincount(gg, yy * yy, n_groups)
    sxy = np.bincount(gg, rx * yy, n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        ic = cov / np.sqrt(vx * vy)
        pct = rx / n[gg]
    top = pct >= 0.8; bot = pct <= 0.2
    nt = np.bincount(gg[top], minlength=n_groups).astype("float64")
    nb = np.bincount(gg[bot], minlength=n_groups).astype("float64")
    ht = np.bincount(gg[top], yy[top], n_groups)
    hb = np.bincount(gg[bot], yy[bot], n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = ht / nt - hb / nb
    bad = (n < min_n) | ~np.isfinite(ic)
    ic[bad] = np.nan; spread[bad] = np.nan
    return ic, spread, n


def _summarise(daily_ic, daily_spread, date_fold, horizon, n_folds):
    """Mean IC, overlap-adjusted t, per-fold means and sign stability."""
    ok = np.isfinite(daily_ic)
    x = daily_ic[ok]
    if len(x) < 3:
        return None
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    n_eff = max(len(x) / horizon, 1.0)
    t = mu / (sd / math.sqrt(n_eff)) if sd > 0 else 0.0
    fm = []
    for k in range(n_folds):
        m = ok & (date_fold == k)
        if m.sum() >= 10:
            fm.append(float(daily_ic[m].mean()))
    same = (np.mean([np.sign(v) == np.sign(mu) for v in fm]) if fm else np.nan)
    sp = daily_spread[ok & np.isfinite(daily_spread)]
    return {"n_dates": int(len(x)), "mean_ic": mu, "ic_sd": sd, "t": float(t),
            "fold_means": fm, "fold_sign_frac": float(same),
            "spread": float(sp.mean()) if len(sp) else np.nan}


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


# ======================================================================
# PIPELINE
# ======================================================================
def run(panel_path, *, resume: Optional[str] = None,
        overrides: Optional[dict] = None, verbose: bool = True) -> Path:
    import panel_build as PB
    Run, RunConflict, fingerprint = _run_cls()
    panel_path = Path(panel_path)
    cfg = {**DEFAULTS, **(overrides or {})}
    base = panel_path.parent / "atlas"
    rid = resume or new_run_id(base)
    R = Run(base, rid, cfg, fingerprint(panel_path))
    R.log(f"=== {rid} ({'resume' if resume else 'new'}) ===")
    t_all = time.perf_counter()

    # ---- A: core
    core = Core(panel_path, cfg["target"])
    R.log(f"[A] core: {core.n:,} rows | {len(core.dates):,} dates | "
          f"{len(core.symbols):,} symbols | base rate {np.nanmean(core.y):.4f}")
    research_end, lb_start = RC.lockbox_bounds(core.dates, cfg["lockbox_fraction"],
                                               cfg["horizon"])
    in_research = core.ts <= research_end
    if core.has("close"):
        # research rows only: even a pass/fail check should not read the lockbox
        cl = core.read("close")[in_research]
        fwd = (core.read("label_fwd_ret_5d")[in_research] if core.has("label_fwd_ret_5d")
               else RC.forward_return_for_check(cl, core.sym_code[in_research], cfg["horizon"]))
        timing = RC.assert_timing_contract(cl, core.sym_code[in_research], fwd)
        del cl, fwd
        R.log(f"    timing contract OK: same-day {timing['corr_fwd_vs_same_day']:+.3f}, "
              f"next-day {timing['corr_fwd_vs_next_day']:+.3f}")
    else:
        timing = {"skipped": "no close column"}
    R.log(f"    research <= {str(research_end)[:10]} | lockbox "
          + (f">= {str(lb_start)[:10]} EXCLUDED from every statistic"
             if lb_start is not None else "DISABLED"))
    splits = PB.walk_forward_splits(
        pd.DataFrame({"timestamp": pd.to_datetime(core.ts[in_research])}),
        n_splits=cfg["n_splits"])

    # ---- B: feature library
    store = Store(R, core.n, in_research)
    cache: dict = {}
    plan = build_plan(core, cfg, cache)
    groups: Dict[str, List[Spec]] = {}
    for sp in plan:
        groups.setdefault(sp.group, []).append(sp)
    R.log(f"[B] library plan: {len(plan)} features in {len(groups)} groups "
          f"({', '.join(f'{g}:{len(s)}' for g, s in groups.items())})")
    for g, specs in groups.items():
        if R.done(f"B_{g}"):
            R.log(f"    {g}: checkpoint found")
            continue
        R.log(f"    {g}: generating {len(specs)}")
        store.write_group(g, specs, R.log)
        R.mark(f"B_{g}", n=len(specs))
        # Per group, not once at the end: with --all-representations the
        # cache would otherwise accumulate every panel column (~1.2 GB).
        cache.clear()

    # ---- C: regimes
    if R.done("C"):
        R.log("[C] regimes: checkpoint found")
        rz = np.load(R.path("regimes.npz"))
        rmeta = json.loads(R.path("regimes.json").read_text(encoding="utf-8"))
    else:
        R.log("[C] regime discovery")
        Xs, st_cols, st_missing = RC.stock_state(
            lambda c: core.read(c) if core.has(c) else None, core.date_code)
        if st_missing:
            R.log(f"    WARNING: stock-state inputs missing {st_missing}; "
                  f"{len(st_cols)} of {len(RC.STOCK_STATE_SPEC)} dimensions used")
        np.save(R.path("stock_state.npy"), Xs)
        stock = discover_regimes(core, Xs, core.ts, splits, cfg["k_range_stock"],
                                 cfg, "stock", R.log)
        del Xs
        mk = RC.market_state(core.read("close"), core.date_code, _seg(core),
                             len(core.dates))
        np.savez(R.path("market_state.npz"), **mk)
        mk_cols = list(RC.MARKET_REGIME_INPUTS)
        Xm = np.column_stack([mk[c] for c in mk_cols]).astype("float32")
        # SCALER LEARNED FROM FOLD 1'S TRAINING WINDOW ONLY.
        #
        # v2 took the mean and sd over EVERY date, lockbox included. Tested by
        # rewriting only lockbox data: the market regime of 388 of 760
        # RESEARCH dates changed. The lockbox was shaping the coordinates every
        # research market state was measured in.
        #
        # Fold 1's window precedes every test window and the lockbox, so it is
        # permitted for every engine; and one shared scaler keeps all folds in
        # one coordinate system, which the Hungarian alignment needs.
        tr0 = core.dates <= np.datetime64(pd.Timestamp(splits[0]["train_end"]))
        mu = np.nanmean(Xm[tr0], axis=0); sd = np.nanstd(Xm[tr0], axis=0)
        Xm = (Xm - mu) / np.where(sd > 0, sd, 1)
        scaler = {"fitted_on": f"dates <= {splits[0]['train_end']} (fold-1 train)",
                  "mean": dict(zip(mk_cols, map(float, mu))),
                  "sd": dict(zip(mk_cols, map(float, sd)))}
        market = discover_regimes(core, Xm, core.dates, splits,
                                  cfg["k_range_market"],
                                  {**cfg, "gmm_rows": len(Xm),
                                   "gmm_select_rows": len(Xm)}, "market", R.log)
        R.path("regimes_missing.json").write_text(json.dumps(
            {"stock": stock["missing"], "market": market["missing"]}, indent=1),
            encoding="utf-8")
        np.savez(R.path("regimes.npz"),
                 stock=stock["label"], stock_conf=stock["conf"],
                 stock_in_sample=stock["in_sample"],
                 market_by_date=market["label"])
        rmeta = {"stock": {k: stock[k] for k in ("K", "bic", "drift", "centroids")},
                 "market": {k: market[k] for k in ("K", "bic", "drift", "centroids")},
                 "stock_inputs": st_cols, "market_inputs": mk_cols,
                 "state_engine": RC.STATE_ENGINE_VERSION,
                 "market_scaler": scaler}
        R.path("regimes.json").write_text(json.dumps(rmeta, indent=1), encoding="utf-8")
        R.mark("C", stock_K=stock["K"], market_K=market["K"])
        rz = np.load(R.path("regimes.npz"))
    stock_lab = rz["stock"]; mkt_by_date = rz["market_by_date"]
    Ks, Km = rmeta["stock"]["K"], rmeta["market"]["K"]
    unassigned = int((stock_lab < 0).sum())
    un_dates = int((mkt_by_date < 0).sum())
    R.log(f"    stock K={Ks} | market K={Km} | unassigned rows: {unassigned} "
          f"| unassigned dates: {un_dates}")
    if unassigned or un_dates:
        # Every row gets a regime by construction. Any -1 means rows fell
        # outside every engine's window - v4 lost the entire final fold this
        # way and printed the count as if it were normal. Stop.
        raise RuntimeError(
            f"{unassigned:,} rows and {un_dates:,} dates have no regime. Every row "
            f"must get one; refusing to compute an atlas with missing periods.")

    # test-window mask and fold id per row / per date
    test_fold = np.full(core.n, -1, dtype="int16")
    date_fold = np.full(len(core.dates), -1, dtype="int16")
    for k, sp in enumerate(splits):
        a, b = np.datetime64(pd.Timestamp(sp["test_start"])), \
            np.datetime64(pd.Timestamp(sp["test_end"]))
        test_fold[(core.ts >= a) & (core.ts <= b)] = k
        date_fold[(core.dates >= a) & (core.dates <= b)] = k
    in_test = (test_fold >= 0) & in_research      # the lockbox never enters
    y = np.where(in_test, core.y, np.nan).astype("float64")

    # group keys
    D = len(core.dates)
    g_date = np.where(in_test, core.date_code, -1)
    g_stock = np.where(in_test & (stock_lab >= 0),
                       core.date_code * Ks + stock_lab, -1)
    mkt_row = mkt_by_date[core.date_code]

    # ---- D: atlas, checkpointed per chunk of features
    names = [n for n, r in store.registry.items() if r["coverage"] >= 0.30]
    chunk = 25
    for ci in range(0, len(names), chunk):
        key = f"D_{ci // chunk:04d}"
        if R.done(key):
            continue
        rows = []
        t0 = time.perf_counter()
        for nm in names[ci:ci + chunk]:
            f = store.read(nm).astype("float64")
            # layer ALL + MARKET: one per-date ranking serves both
            ic_d, sp_d, _ = _group_ic(f, y, g_date, D, cfg["min_names"])
            s = _summarise(ic_d, sp_d, date_fold, cfg["horizon"], len(splits))
            if s:
                rows.append({"feature": nm, "layer": "all", "regime": -1, **s})
            for k in range(Km):
                m = mkt_by_date == k
                s = _summarise(np.where(m, ic_d, np.nan), np.where(m, sp_d, np.nan),
                               date_fold, cfg["horizon"], len(splits))
                if s and s["n_dates"] >= cfg["min_dates"]:
                    rows.append({"feature": nm, "layer": "market", "regime": k, **s})
            # layer STOCK: rank within (date, stock regime)
            ic_s, sp_s, _ = _group_ic(f, y, g_stock, D * Ks, cfg["min_names"])
            ic_s = ic_s.reshape(D, Ks); sp_s = sp_s.reshape(D, Ks)
            for k in range(Ks):
                s = _summarise(ic_s[:, k], sp_s[:, k], date_fold,
                               cfg["horizon"], len(splits))
                if s and s["n_dates"] >= cfg["min_dates"]:
                    rows.append({"feature": nm, "layer": "stock", "regime": k, **s})
        out = pd.DataFrame(rows)
        if len(out):
            out["fold_means"] = out["fold_means"].apply(json.dumps)
        out.to_parquet(R.path("atlas_parts", f"{key}.parquet"), index=False)
        R.mark(key, n=len(rows))
        R.log(f"[D] features {ci+1}-{min(ci+chunk, len(names))}/{len(names)} "
              f"({time.perf_counter()-t0:.0f}s)")

    # ---- E: statistics across every cell
    R.log("[E] multiple-testing control + stability")
    parts = sorted(R.path("atlas_parts").glob("D_*.parquet"))
    A = pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)
    from scipy.stats import norm
    A["p"] = 2 * (1 - norm.cdf(np.abs(A["t"])))
    A["q"] = _bh(A["p"].to_numpy())
    A["working"] = ((A["q"] < cfg["fdr_q"])
                    & (A["fold_sign_frac"] >= cfg["min_fold_sign"])
                    & (A["mean_ic"].abs() >= cfg["min_abs_ic"]))
    A["category"] = A["feature"].map(lambda n: store.registry[n]["category"])
    A.to_parquet(R.path("atlas.parquet"), index=False)

    # families among working features
    wf = sorted(A.loc[A["working"], "feature"].unique())
    fam = {}
    if wf:
        rng = np.random.default_rng(cfg["seed"])
        idx = np.where(in_test)[0]
        idx = rng.choice(idx, min(60_000, len(idx)), replace=False)
        M = pd.DataFrame({n: store.read(n)[idx] for n in wf})
        corr = M.rank().corr().abs().fillna(0).to_numpy()
        fid, left = 0, list(range(len(wf)))
        while left:
            i = left.pop(0)
            mem = [i] + [j for j in left if corr[i, j] >= cfg["family_corr"]]
            left = [j for j in left if j not in mem]
            for m_ in mem:
                fam[wf[m_]] = fid
            fid += 1
    A["family"] = A["feature"].map(fam)
    A.to_parquet(R.path("atlas.parquet"), index=False)

    # ---- F: report
    R.log("[F] report")
    write_report(R, A, store, rmeta, core, stock_lab, mkt_by_date, cfg, len(names),
                 timing=timing, lockbox=(research_end, lb_start))
    if not R.done("ledger"):
        RC.ledger_append(panel_path, {"kind": "atlas", "tool": "feature_atlas",
                                      "run": rid, "hash": R.hash,
                                      "cells_tested": int(len(A)),
                                      "working_cells": int(A["working"].sum())})
        R.mark("ledger")
    R.log(f"=== done in {(time.perf_counter()-t_all)/60:.1f} min -> "
          f"{R.dir / 'report.md'}")
    return R.dir


# ======================================================================
# REPORT
# ======================================================================
def write_report(R, A, store, rmeta, core, stock_lab, mkt_by_date, cfg, n_eval,
                 timing=None, lockbox=None):
    L = [f"# Feature Atlas - {R.run_id}", "",
         f"Target `{cfg['target']}` | per-date rank IC | BH-FDR q<{cfg['fdr_q']} "
         f"| sign-stable in >= {cfg['min_fold_sign']:.0%} of folds | "
         f"|IC| >= {cfg['min_abs_ic']}", "",
         "**Exploratory. Nothing here feeds a model.** A 'working' cell is a "
         "feature whose cross-sectional ranking predicted the target inside that "
         "regime, out of sample, after correcting for the number of tests.", "",
         f"**Timing:** {RC.timing_contract(RC.entry_of_target(cfg['target']))}", ""]
    if timing and "corr_fwd_vs_same_day" in timing:
        L += [f"Verified on the data: corr(fwd, same day) "
              f"{timing['corr_fwd_vs_same_day']:+.3f}, corr(fwd, next day) "
              f"{timing['corr_fwd_vs_next_day']:+.3f}.", ""]
    if lockbox and lockbox[1] is not None:
        L += [f"**Lockbox:** sessions from {str(lockbox[1])[:10]} are EXCLUDED "
              f"from every statistic here. It is the same lockbox regime_research "
              f"uses; looking at it here would spend it.", ""]
    mj = R.path("regimes_missing.json")
    if mj.exists():
        mm = json.loads(mj.read_text(encoding="utf-8"))["stock"]
        L += [f"**Missing state:** {mm['pct_rows_any_missing']:.2%} of rows have a "
              f"missing stock-state input and are scored on observed axes only "
              f"(mean confidence {mm['mean_conf_partial']:.3f} vs "
              f"{mm['mean_conf_complete']:.3f} for complete rows).", ""]

    reg = store.registry
    by_cat: Dict[str, int] = {}
    for r in reg.values():
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    n_err = sum(1 for r in reg.values() if r.get("error"))
    L += ["## Library", "",
          f"- features generated: {len(reg)} ({n_err} failed to compute)",
          f"- evaluated (coverage >= 30%): {n_eval}",
          "- by category: " + ", ".join(f"{k} {v}" for k, v in sorted(by_cat.items())),
          f"- cells tested: {len(A):,} (feature x layer x regime)",
          f"- working cells: {int(A['working'].sum()):,} | working features: "
          f"{A.loc[A['working'],'feature'].nunique()} | families: "
          f"{A.loc[A['working'],'family'].nunique()}",
          f"- expected false discoveries among working cells at q<{cfg['fdr_q']}: "
          f"~{int(A['working'].sum() * cfg['fdr_q'])}", ""]

    # regimes
    for layer, key in (("Stock", "stock"), ("Market", "market")):
        m = rmeta[key]
        inputs = rmeta["stock_inputs"] if key == "stock" else rmeta["market_inputs"]
        L += [f"## {layer} regimes (K={m['K']}, chosen by BIC on fold-1 train)", "",
              f"Alignment drift across folds (mean centroid distance to the "
              f"fold-1 reference): {', '.join(f'{d:.2f}' for d in m['drift'])}. "
              f"Large values mean a regime ID no longer describes the same state.", ""]
        L += ["| regime | " + " | ".join(i.replace("D_", "") for i in inputs)
              + " | share | base rate |", "|---|" + "---|" * (len(inputs) + 2)]
        lab = stock_lab if key == "stock" else mkt_by_date[core.date_code]
        for k, cen in m["centroids"].items():
            sel = lab == int(k)
            share = float(sel.mean())
            br = float(np.nanmean(core.y[sel])) if sel.any() else np.nan
            L.append(f"| {layer[0]}{k} | " + " | ".join(f"{v:+.2f}" for v in cen)
                     + f" | {share:.1%} | {br:.3f} |")
        L.append("")
    # stock persistence
    df = pd.DataFrame({"c": core.sym_code, "r": stock_lab})
    brk = (df["r"] != df.groupby("c")["r"].shift()).cumsum()
    runs = df.groupby(brk)["r"].agg(["first", "size"])
    ps = runs.groupby("first")["size"].agg(["mean", "median"])
    L += ["Stock-regime persistence (measured): " + ", ".join(
        f"S{int(k)} mean run {r['mean']:.1f}d" for k, r in ps.iterrows() if k >= 0), ""]

    W = A[A["working"]].copy()
    fmt = lambda r: (f"| `{r.feature}` | {r.category} | {r.mean_ic:+.4f} | "  # noqa
                     f"{r.t:+.1f} | {r.q:.2g} | {r.fold_sign_frac:.0%} | "
                     f"{r.spread:+.3f} | {r.n_dates} |")
    hdr = ["| feature | category | IC | t | q | fold sign | top-bot hit | dates |",
           "|---|---|---|---|---|---|---|---|"]

    L += ["## Works across all stocks (unconditional)", ""] + hdr
    for r in W[W.layer == "all"].sort_values("mean_ic", key=abs, ascending=False).head(25).itertuples():
        L.append(fmt(r))
    L.append("")

    for layer in ("stock", "market"):
        K = rmeta[layer]["K"]
        for k in range(K):
            sub = W[(W.layer == layer) & (W.regime == k)]
            L += [f"### Works in {layer} regime {layer[0].upper()}{k} "
                  f"({len(sub)} cells)", ""] + hdr
            for r in sub.sort_values("mean_ic", key=abs, ascending=False).head(12).itertuples():
                L.append(fmt(r))
            L.append("")

    # flips - the single most interesting finding type
    fl = []
    for (feat, layer), g in W[W.layer != "all"].groupby(["feature", "layer"]):
        if (g.mean_ic > 0).any() and (g.mean_ic < 0).any():
            pos = g[g.mean_ic > 0].sort_values("mean_ic").iloc[-1]
            neg = g[g.mean_ic < 0].sort_values("mean_ic").iloc[0]
            fl.append({"feature": feat, "layer": layer,
                       "pos_regime": int(pos.regime), "pos_ic": pos.mean_ic,
                       "neg_regime": int(neg.regime), "neg_ic": neg.mean_ic})
    FL = pd.DataFrame(fl)
    FL.to_csv(R.path("flips.csv"), index=False)
    L += ["## Sign flips - works one way here, the opposite way there", "",
          "HYPOTHESES, not findings: selected from many tests, so each needs "
          "its own confirmation on data not used to find it.", "",
          "A feature significant with OPPOSITE signs in two regimes. An "
          "unconditional model averages these into nothing; this is exactly the "
          "information regime conditioning exists to recover.", ""]
    if len(FL):
        L += ["| feature | layer | + regime | + IC | - regime | - IC |",
              "|---|---|---|---|---|---|"]
        for r in FL.sort_values("pos_ic", ascending=False).itertuples():
            L.append(f"| `{r.feature}` | {r.layer} | {r.layer[0].upper()}{r.pos_regime} | "
                     f"{r.pos_ic:+.4f} | {r.layer[0].upper()}{r.neg_regime} | {r.neg_ic:+.4f} |")
    else:
        L.append("_None survived the FDR and stability bars._")
    L.append("")

    # matrix
    top = (A.groupby("feature")["mean_ic"].apply(lambda s: s.abs().max())
           .sort_values(ascending=False).head(40).index)
    cols = [("all", -1)] + [("stock", k) for k in range(rmeta["stock"]["K"])] + \
           [("market", k) for k in range(rmeta["market"]["K"])]
    lab = {("all", -1): "ALL", **{("stock", k): f"S{k}" for k in range(rmeta['stock']['K'])},
           **{("market", k): f"M{k}" for k in range(rmeta['market']['K'])}}
    Ai = A.set_index(["feature", "layer", "regime"])
    L += ["## Feature x regime matrix (top 40 by max |IC|)", "",
          "`*` = working (FDR + stable + material). Blank = too few dates.", "",
          "| feature | " + " | ".join(lab[c] for c in cols) + " |",
          "|---|" + "---|" * len(cols)]
    mat = []
    for f in top:
        cells, row = [], {"feature": f}
        for c in cols:
            if (f, *c) in Ai.index:
                r = Ai.loc[(f, *c)]
                cells.append(f"{r.mean_ic:+.3f}{'*' if r.working else ''}")
                row[lab[c]] = r.mean_ic
            else:
                cells.append("")
        L.append(f"| `{f}` | " + " | ".join(cells) + " |")
        mat.append(row)
    pd.DataFrame(mat).to_csv(R.path("feature_regime_matrix.csv"), index=False)
    L.append("")

    W.to_csv(R.path("working_features.csv"), index=False)
    L += ["## Caveats", "",
          "- Exploratory. A working cell is evidence to test, not a validated edge.",
          f"- About {cfg['fdr_q']:.0%} of working cells are expected to be false "
          "discoveries by construction of the FDR bar.",
          "- IC is cross-sectional ranking skill. It is not a return, and it "
          "ignores costs; `top-bot hit` is the gap in the mean target between the "
          "top and bottom quintile within the regime - a hit-rate gap for a 0/1 "
          "target, a return gap for label_exit_ret.",
          "- Regime IDs are aligned across folds by centroid matching; check the "
          "drift figures before trusting a regime's identity over time.",
          "- In-sample regime labels (before the first test window) never enter "
          "the statistics."]
    R.path("report.md").write_text("\n".join(L), encoding="utf-8")


def list_runs(root: Path) -> None:
    for d in sorted((Path(root) / "panel" / "atlas").glob("ATLAS_*")):
        c = json.loads((d / "config.json").read_text(encoding="utf-8"))
        n = len(list((d / "checkpoints").glob("*.done"))) if (d / "checkpoints").exists() else 0
        print(f"  {d.name}  {c['created'][:19]}  {n} checkpoints  "
              f"{'report' if (d / 'report.md').exists() else 'in progress'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run", "report", "list"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--target", default=None,
                    help="label column to measure against, e.g. label_exit_ret "
                         "(realised trade return). Default label_tp_before_sl.")
    ap.add_argument("--all-representations", action="store_true",
                    help="representations of EVERY panel feature, not just the core set")
    a = ap.parse_args()
    root = a.root or os.environ.get("CACHE_DAILY_ROOT")
    if not root:
        raise SystemExit("CACHE_DAILY_ROOT not set and --root not given")
    ppq = Path(root) / "panel" / "panel.parquet"
    ov = {"all_representations": True} if a.all_representations else {}
    if a.target:
        if not a.target.startswith("label_"):
            raise SystemExit("--target must be a label_* column")
        ov["target"] = a.target
    if a.cmd == "run":
        run(ppq, resume=a.resume, overrides=ov)
    elif a.cmd == "list":
        list_runs(Path(root))
    else:
        rid = a.run or a.resume
        print((ppq.parent / "atlas" / rid / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
