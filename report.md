# Feature Atlas - ATLAS_20260928_001

Target `label_exit_ret_o1` | per-date rank IC | BH-FDR q<0.05 | sign-stable in >= 80% of folds | |IC| >= 0.01

**Exploratory. Nothing here feeds a model.** A 'working' cell is a feature whose cross-sectional ranking predicted the target inside that regime, out of sample, after correcting for the number of tests.

**Timing:** Prediction is made at the CLOSE of session T. Every feature uses data through the close of T and nothing later. The target covers sessions T+1..T+H, entered at the close of T. Same-day market state (breadth, median move, dispersion) is therefore legitimate: it is known at T close.

Verified on the data: corr(fwd, same day) +0.008, corr(fwd, next day) +0.425.

**Lockbox:** sessions from 2025-03-05 are EXCLUDED from every statistic here. It is the same lockbox regime_research uses; looking at it here would spend it.

**Missing state:** 0.02% of rows have a missing stock-state input and are scored on observed axes only (mean confidence 0.808 vs 0.798 for complete rows).

## Library

- features generated: 443 (0 failed to compute)
- evaluated (coverage >= 30%): 443
- by category: distribution 5, existing 265, interaction 10, location 8, market-relative 5, momentum 5, representation 138, structure 3, volatility 2, volume 2
- cells tested: 5,886 (feature x layer x regime)
- working cells: 626 | working features: 223 | families: 99
- expected false discoveries among working cells at q<0.05: ~31

## Stock regimes (K=9, chosen by BIC on fold-1 train)

Alignment drift across folds (mean centroid distance to the fold-1 reference): 0.00, 0.33, 0.36, 0.31, 0.39. Large values mean a regime ID no longer describes the same state.

| regime | st_trend | st_trend_strength | st_vol_level | st_vol_change | st_momentum | st_participation | st_location | st_shock | share | base rate |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | -0.27 | -0.05 | -0.21 | -0.11 | -0.32 | +0.14 | -0.21 | +0.02 | 13.0% | 0.000 |
| S1 | -0.07 | -0.04 | +0.28 | -0.04 | -0.04 | -0.06 | -0.03 | +0.13 | 9.0% | -0.000 |
| S2 | +0.03 | +0.10 | +0.34 | +0.40 | +0.01 | -0.04 | -0.06 | +0.06 | 7.7% | -0.001 |
| S3 | +0.29 | -0.06 | +0.09 | +0.05 | +0.24 | +0.32 | +0.01 | +0.27 | 13.3% | -0.002 |
| S4 | -0.30 | -0.12 | -0.04 | -0.09 | -0.29 | -0.30 | -0.15 | -0.04 | 14.0% | -0.000 |
| S5 | +0.06 | -0.06 | -0.25 | -0.08 | +0.10 | -0.03 | +0.00 | -0.02 | 9.2% | -0.000 |
| S6 | +0.25 | +0.34 | +0.09 | +0.05 | +0.28 | -0.07 | +0.31 | -0.01 | 11.3% | -0.000 |
| S7 | +0.01 | -0.09 | -0.08 | -0.03 | +0.01 | +0.01 | -0.02 | -0.36 | 15.8% | 0.000 |
| S8 | +0.42 | +0.17 | +0.13 | +0.10 | +0.42 | +0.27 | +0.45 | +0.12 | 6.7% | -0.001 |

## Market regimes (K=6, chosen by BIC on fold-1 train)

Alignment drift across folds (mean centroid distance to the fold-1 reference): 0.00, 1.21, 3.36, 4.43, 4.36. Large values mean a regime ID no longer describes the same state.

| regime | mk_trend20 | mk_vol20 | mk_breadth20 | mk_disp20 | share | base rate |
|---|---|---|---|---|---|---|
| M0 | -0.03 | +1.16 | +0.26 | +0.61 | 3.3% | 0.005 |
| M1 | -2.53 | +2.74 | -1.46 | +2.00 | 2.6% | 0.003 |
| M2 | +0.80 | -0.25 | +0.84 | -0.63 | 22.1% | -0.002 |
| M3 | -1.01 | +0.49 | -1.19 | +0.36 | 29.0% | 0.002 |
| M4 | +0.17 | -0.46 | +0.00 | +0.79 | 15.0% | -0.004 |
| M5 | +0.27 | -0.79 | +0.08 | -0.65 | 28.0% | -0.001 |

Stock-regime persistence (measured): S0 mean run 2.3d, S1 mean run 2.3d, S2 mean run 2.3d, S3 mean run 2.3d, S4 mean run 2.3d, S5 mean run 2.3d, S6 mean run 2.5d, S7 mean run 2.3d, S8 mean run 2.3d

## Works across all stocks (unconditional)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `X_z_D_range_pct` | existing | -0.0413 | -6.2 | 2.3e-07 | 100% | -0.004 | 1801 |
| `D_range_pct` | existing | -0.0413 | -6.2 | 2.3e-07 | 100% | -0.004 | 1801 |
| `X_rank_D_range_pct` | existing | -0.0413 | -6.2 | 2.3e-07 | 100% | -0.004 | 1801 |
| `D_vol_yz_20` | existing | -0.0394 | -4.0 | 0.002 | 100% | -0.003 | 1801 |
| `X_z_D_atr_pct` | existing | -0.0372 | -3.4 | 0.01 | 100% | -0.003 | 1801 |
| `X_rank_D_atr_pct` | existing | -0.0372 | -3.4 | 0.01 | 100% | -0.003 | 1801 |
| `R_atr_pct__csrank` | representation | -0.0372 | -3.4 | 0.01 | 100% | -0.003 | 1801 |
| `R_atr_pct__csz` | representation | -0.0372 | -3.4 | 0.01 | 100% | -0.003 | 1801 |
| `D_atr_pct` | existing | -0.0372 | -3.4 | 0.01 | 100% | -0.003 | 1801 |
| `D_vol_yz_50` | existing | -0.0369 | -3.8 | 0.0034 | 100% | -0.003 | 1801 |
| `X_z_D_realvol_20` | existing | -0.0340 | -3.6 | 0.0064 | 100% | -0.002 | 1801 |
| `X_relvol_20` | existing | -0.0340 | -3.6 | 0.0064 | 100% | -0.002 | 1801 |
| `R_realvol_20__csz` | representation | -0.0340 | -3.6 | 0.0064 | 100% | -0.002 | 1801 |
| `D_realvol_20` | existing | -0.0340 | -3.6 | 0.0064 | 100% | -0.002 | 1801 |
| `R_realvol_20__csrank` | representation | -0.0340 | -3.6 | 0.0064 | 100% | -0.002 | 1801 |
| `X_rank_D_realvol_20` | existing | -0.0340 | -3.6 | 0.0064 | 100% | -0.002 | 1801 |
| `D_realvol_60` | existing | -0.0311 | -3.2 | 0.019 | 100% | -0.002 | 1801 |
| `N_dist_lo10` | location | -0.0306 | -6.7 | 1.4e-08 | 100% | -0.003 | 1801 |
| `X_z_D_downside_dev_60` | existing | -0.0302 | -3.1 | 0.024 | 100% | -0.002 | 1801 |
| `D_downside_dev_60` | existing | -0.0302 | -3.1 | 0.024 | 100% | -0.002 | 1801 |
| `X_rank_D_downside_dev_60` | existing | -0.0302 | -3.1 | 0.024 | 100% | -0.002 | 1801 |
| `D_WQ_40` | existing | +0.0296 | +8.2 | 1.3e-12 | 100% | +0.003 | 1801 |
| `R_bb_bw_20__csrank` | representation | -0.0292 | -3.5 | 0.008 | 100% | -0.002 | 1801 |
| `D_bb_bw_20` | existing | -0.0292 | -3.5 | 0.008 | 100% | -0.002 | 1801 |
| `R_bb_bw_20__csz` | representation | -0.0292 | -3.5 | 0.008 | 100% | -0.002 | 1801 |

### Works in stock regime S0 (3 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `N_min_ret20` | distribution | +0.0247 | +2.9 | 0.041 | 100% | +0.002 | 1801 |
| `D_WQ_40` | existing | +0.0205 | +3.1 | 0.024 | 100% | +0.002 | 1801 |
| `R_pos_in_52w_range__accel` | representation | -0.0192 | -2.8 | 0.047 | 100% | -0.002 | 1801 |

### Works in stock regime S1 (22 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `X_rank_D_downside_dev_60` | existing | -0.0424 | -4.0 | 0.0022 | 100% | -0.004 | 1668 |
| `D_downside_dev_60` | existing | -0.0424 | -4.0 | 0.0022 | 100% | -0.004 | 1668 |
| `X_z_D_downside_dev_60` | existing | -0.0423 | -4.0 | 0.0023 | 100% | -0.004 | 1668 |
| `D_vol_yz_50` | existing | -0.0413 | -3.7 | 0.0046 | 100% | -0.004 | 1668 |
| `X_rank_D_atr_pct` | existing | -0.0402 | -3.4 | 0.013 | 100% | -0.003 | 1668 |
| `D_atr_pct` | existing | -0.0402 | -3.4 | 0.013 | 100% | -0.003 | 1668 |
| `R_atr_pct__csrank` | representation | -0.0402 | -3.4 | 0.013 | 100% | -0.003 | 1668 |
| `R_atr_pct__csz` | representation | -0.0402 | -3.4 | 0.013 | 100% | -0.003 | 1668 |
| `X_z_D_atr_pct` | existing | -0.0402 | -3.3 | 0.013 | 100% | -0.003 | 1668 |
| `D_vol_yz_20` | existing | -0.0397 | -3.5 | 0.008 | 100% | -0.003 | 1668 |
| `D_realvol_60` | existing | -0.0345 | -3.2 | 0.021 | 100% | -0.002 | 1668 |
| `R_realvol_20__csrank` | representation | -0.0332 | -3.1 | 0.024 | 100% | -0.002 | 1668 |

### Works in stock regime S2 (55 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `D_vol_yz_20` | existing | -0.0501 | -4.8 | 0.00018 | 100% | -0.004 | 1801 |
| `X_z_D_atr_pct` | existing | -0.0489 | -4.4 | 0.00057 | 100% | -0.004 | 1801 |
| `D_atr_pct` | existing | -0.0489 | -4.4 | 0.00057 | 100% | -0.004 | 1801 |
| `R_atr_pct__csz` | representation | -0.0489 | -4.4 | 0.00057 | 100% | -0.004 | 1801 |
| `R_atr_pct__csrank` | representation | -0.0489 | -4.4 | 0.00057 | 100% | -0.004 | 1801 |
| `X_rank_D_atr_pct` | existing | -0.0489 | -4.4 | 0.00057 | 100% | -0.004 | 1801 |
| `D_vol_yz_50` | existing | -0.0470 | -4.5 | 0.00048 | 100% | -0.004 | 1801 |
| `N_dist_lo10` | location | -0.0457 | -5.2 | 2.3e-05 | 100% | -0.005 | 1800 |
| `X_z_D_downside_dev_60` | existing | -0.0449 | -4.6 | 0.00041 | 100% | -0.004 | 1801 |
| `D_downside_dev_60` | existing | -0.0449 | -4.6 | 0.00041 | 100% | -0.004 | 1801 |
| `X_rank_D_downside_dev_60` | existing | -0.0449 | -4.6 | 0.00041 | 100% | -0.004 | 1801 |
| `X_z_D_range_pct` | existing | -0.0425 | -4.7 | 0.00022 | 100% | -0.005 | 1801 |

### Works in stock regime S3 (54 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `X_rank_D_range_pct` | existing | -0.0524 | -5.6 | 3.8e-06 | 100% | -0.005 | 1689 |
| `D_range_pct` | existing | -0.0524 | -5.6 | 3.8e-06 | 100% | -0.005 | 1689 |
| `X_z_D_range_pct` | existing | -0.0524 | -5.6 | 3.8e-06 | 100% | -0.005 | 1689 |
| `D_vol_yz_20` | existing | -0.0462 | -4.0 | 0.0021 | 100% | -0.003 | 1689 |
| `X_rank_D_atr_pct` | existing | -0.0439 | -3.6 | 0.006 | 100% | -0.003 | 1689 |
| `D_atr_pct` | existing | -0.0439 | -3.6 | 0.006 | 100% | -0.003 | 1689 |
| `R_atr_pct__csrank` | representation | -0.0439 | -3.6 | 0.006 | 100% | -0.003 | 1689 |
| `X_z_D_atr_pct` | existing | -0.0439 | -3.6 | 0.006 | 100% | -0.003 | 1689 |
| `R_atr_pct__csz` | representation | -0.0439 | -3.6 | 0.006 | 100% | -0.003 | 1689 |
| `D_vol_yz_50` | existing | -0.0430 | -3.8 | 0.0039 | 100% | -0.003 | 1689 |
| `X_rank_D_realvol_20` | existing | -0.0412 | -3.6 | 0.0057 | 100% | -0.003 | 1689 |
| `X_relvol_20` | existing | -0.0412 | -3.6 | 0.0057 | 100% | -0.003 | 1689 |

### Works in stock regime S4 (2 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `D_WQ_40` | existing | +0.0264 | +3.8 | 0.0033 | 100% | +0.002 | 1801 |
| `D_pdi14` | existing | -0.0196 | -2.8 | 0.043 | 100% | -0.002 | 1801 |

### Works in stock regime S5 (44 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `D_vol_yz_20` | existing | -0.0509 | -5.1 | 5e-05 | 100% | -0.003 | 1801 |
| `D_atr_pct` | existing | -0.0507 | -4.5 | 0.00042 | 100% | -0.003 | 1801 |
| `R_atr_pct__csrank` | representation | -0.0507 | -4.5 | 0.00042 | 100% | -0.003 | 1801 |
| `X_rank_D_atr_pct` | existing | -0.0507 | -4.5 | 0.00042 | 100% | -0.003 | 1801 |
| `R_atr_pct__csz` | representation | -0.0507 | -4.5 | 0.00042 | 100% | -0.003 | 1801 |
| `X_z_D_atr_pct` | existing | -0.0507 | -4.5 | 0.00042 | 100% | -0.003 | 1801 |
| `D_vol_yz_50` | existing | -0.0490 | -5.0 | 5.5e-05 | 100% | -0.003 | 1801 |
| `X_z_D_range_pct` | existing | -0.0451 | -5.3 | 2.2e-05 | 100% | -0.003 | 1801 |
| `X_rank_D_range_pct` | existing | -0.0451 | -5.3 | 2.2e-05 | 100% | -0.003 | 1801 |
| `D_range_pct` | existing | -0.0451 | -5.3 | 2.2e-05 | 100% | -0.003 | 1801 |
| `N_dist_lo10` | location | -0.0361 | -4.4 | 0.00054 | 100% | -0.003 | 1801 |
| `D_body_ratio_rmean50` | existing | +0.0341 | +4.4 | 0.00057 | 100% | +0.003 | 1801 |

### Works in stock regime S6 (85 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `X_z_D_range_pct` | existing | -0.0525 | -6.0 | 7.1e-07 | 100% | -0.005 | 1726 |
| `X_rank_D_range_pct` | existing | -0.0525 | -6.0 | 7.1e-07 | 100% | -0.005 | 1726 |
| `D_range_pct` | existing | -0.0525 | -6.0 | 7.1e-07 | 100% | -0.005 | 1726 |
| `D_vol_yz_20` | existing | -0.0479 | -4.5 | 0.00051 | 100% | -0.004 | 1726 |
| `D_atr_pct` | existing | -0.0463 | -4.0 | 0.0019 | 100% | -0.004 | 1726 |
| `R_atr_pct__csz` | representation | -0.0463 | -4.0 | 0.0019 | 100% | -0.004 | 1726 |
| `X_rank_D_atr_pct` | existing | -0.0463 | -4.0 | 0.0019 | 100% | -0.004 | 1726 |
| `R_atr_pct__csrank` | representation | -0.0463 | -4.0 | 0.0019 | 100% | -0.004 | 1726 |
| `X_z_D_atr_pct` | existing | -0.0463 | -4.0 | 0.0019 | 100% | -0.004 | 1726 |
| `D_vol_yz_50` | existing | -0.0432 | -4.0 | 0.0019 | 100% | -0.003 | 1726 |
| `D_realvol_20` | existing | -0.0419 | -4.1 | 0.0016 | 100% | -0.004 | 1726 |
| `X_relvol_20` | existing | -0.0419 | -4.1 | 0.0016 | 100% | -0.004 | 1726 |

### Works in stock regime S7 (3 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `D_WQ_40` | existing | +0.0213 | +3.2 | 0.019 | 100% | +0.002 | 1801 |
| `D_cmf20_rmean50` | existing | +0.0190 | +2.9 | 0.043 | 80% | +0.001 | 1801 |
| `D_WQ_16` | existing | +0.0171 | +2.9 | 0.039 | 100% | +0.002 | 1801 |

### Works in stock regime S8 (42 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `D_range_pct` | existing | -0.0502 | -4.7 | 0.0003 | 100% | -0.005 | 1542 |
| `X_rank_D_range_pct` | existing | -0.0502 | -4.7 | 0.0003 | 100% | -0.005 | 1542 |
| `X_z_D_range_pct` | existing | -0.0502 | -4.7 | 0.0003 | 100% | -0.005 | 1542 |
| `X_rank_D_atr_pct` | existing | -0.0495 | -3.7 | 0.0054 | 100% | -0.004 | 1542 |
| `X_z_D_atr_pct` | existing | -0.0495 | -3.7 | 0.0054 | 100% | -0.004 | 1542 |
| `R_atr_pct__csrank` | representation | -0.0495 | -3.7 | 0.0054 | 100% | -0.004 | 1542 |
| `R_atr_pct__csz` | representation | -0.0495 | -3.7 | 0.0054 | 100% | -0.004 | 1542 |
| `D_atr_pct` | existing | -0.0495 | -3.7 | 0.0054 | 100% | -0.004 | 1542 |
| `D_vol_yz_20` | existing | -0.0478 | -3.7 | 0.0054 | 100% | -0.004 | 1542 |
| `D_vol_yz_50` | existing | -0.0425 | -3.3 | 0.013 | 100% | -0.003 | 1542 |
| `R_realvol_20__csrank` | representation | -0.0383 | -3.1 | 0.027 | 100% | -0.003 | 1542 |
| `X_relvol_20` | existing | -0.0383 | -3.1 | 0.027 | 100% | -0.003 | 1542 |

### Works in market regime M0 (3 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `D_vol_surge_20` | existing | -0.0439 | -3.0 | 0.028 | 100% | +nan | 91 |
| `D_WQ_16` | existing | +0.0427 | +3.1 | 0.027 | 100% | +0.005 | 91 |
| `D_WQ_13` | existing | +0.0351 | +2.8 | 0.048 | 100% | +0.004 | 91 |

### Works in market regime M1 (0 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|

### Works in market regime M2 (34 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `X_rank_D_range_pct` | existing | -0.0533 | -4.5 | 0.0005 | 100% | -0.005 | 366 |
| `D_range_pct` | existing | -0.0533 | -4.5 | 0.0005 | 100% | -0.005 | 366 |
| `X_z_D_range_pct` | existing | -0.0533 | -4.5 | 0.0005 | 100% | -0.005 | 366 |
| `D_vol_yz_20` | existing | -0.0525 | -2.9 | 0.036 | 100% | -0.005 | 366 |
| `D_vol_yz_50` | existing | -0.0496 | -2.8 | 0.048 | 100% | -0.004 | 366 |
| `D_days_since_5pct_up` | existing | +0.0401 | +3.2 | 0.021 | 100% | +0.004 | 366 |
| `N_dist_lo50` | location | -0.0401 | -3.6 | 0.0063 | 100% | -0.004 | 366 |
| `N_dist_lo10` | location | -0.0396 | -4.2 | 0.0013 | 100% | -0.004 | 366 |
| `D_WQ_40` | existing | +0.0347 | +4.4 | 0.00056 | 100% | +0.004 | 366 |
| `N_dist_lo100` | location | -0.0312 | -2.9 | 0.04 | 100% | -0.003 | 366 |
| `D_pdi14` | existing | -0.0300 | -3.3 | 0.014 | 100% | -0.003 | 366 |
| `D_atr_ratio_14_30` | existing | -0.0299 | -2.9 | 0.042 | 100% | -0.003 | 366 |

### Works in market regime M3 (32 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `R_ema20_angle_deg__d5` | representation | -0.0355 | -2.8 | 0.043 | 100% | -0.005 | 466 |
| `D_mdi14_diff5` | existing | +0.0328 | +3.7 | 0.005 | 100% | +0.004 | 466 |
| `R_rsi14__d5` | representation | -0.0327 | -3.4 | 0.0099 | 100% | -0.004 | 466 |
| `R_dist_from_20h__d5` | representation | -0.0309 | -3.7 | 0.0055 | 100% | -0.004 | 466 |
| `R_rsi7__d5` | representation | -0.0306 | -3.5 | 0.0077 | 100% | -0.004 | 466 |
| `R_pos_in_52w_range__d5` | representation | -0.0300 | -3.0 | 0.032 | 100% | -0.004 | 466 |
| `D_mdi14_rrank10` | existing | +0.0300 | +3.7 | 0.0056 | 100% | +0.003 | 466 |
| `D_WQ_40` | existing | +0.0292 | +4.0 | 0.0023 | 100% | +0.003 | 466 |
| `D_WQ_44` | existing | +0.0288 | +5.1 | 4.1e-05 | 100% | +0.003 | 466 |
| `D_mdi14_diff1` | existing | +0.0276 | +3.9 | 0.0026 | 100% | +0.004 | 466 |
| `R_pos_in_52w_range__accel` | representation | -0.0270 | -3.5 | 0.0096 | 100% | -0.003 | 466 |
| `R_dist_from_20h__accel` | representation | -0.0264 | -3.8 | 0.004 | 100% | -0.003 | 466 |

### Works in market regime M4 (20 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `X_z_D_range_pct` | existing | -0.0608 | -3.9 | 0.0026 | 100% | -0.006 | 323 |
| `D_range_pct` | existing | -0.0608 | -3.9 | 0.0026 | 100% | -0.006 | 323 |
| `X_rank_D_range_pct` | existing | -0.0608 | -3.9 | 0.0026 | 100% | -0.006 | 323 |
| `D_days_since_5pct_up` | existing | +0.0427 | +2.8 | 0.044 | 100% | +0.005 | 323 |
| `D_pdi14` | existing | -0.0333 | -3.4 | 0.011 | 100% | -0.004 | 323 |
| `D_WQ_40` | existing | +0.0302 | +3.7 | 0.0046 | 100% | +0.003 | 323 |
| `R_realvol_ratio_20_60__d5` | representation | -0.0269 | -3.8 | 0.0038 | 100% | -0.003 | 323 |
| `R_atr_ratio_14_30__d5` | representation | -0.0267 | -3.0 | 0.033 | 100% | -0.003 | 323 |
| `R_bb_bw_20__tsz60` | representation | -0.0258 | -2.9 | 0.04 | 100% | -0.003 | 323 |
| `D_WQ_44` | existing | +0.0255 | +3.5 | 0.0082 | 100% | +0.003 | 323 |
| `R_compress_state__csz` | representation | -0.0254 | -3.1 | 0.024 | 100% | -0.003 | 323 |
| `D_compress_state` | existing | -0.0254 | -3.1 | 0.024 | 100% | -0.003 | 323 |

### Works in market regime M5 (31 cells)

| feature | category | IC | t | q | fold sign | top-bot hit | dates |
|---|---|---|---|---|---|---|---|
| `X_z_D_range_pct` | existing | -0.0438 | -3.7 | 0.0051 | 100% | -0.004 | 471 |
| `X_rank_D_range_pct` | existing | -0.0438 | -3.7 | 0.0051 | 100% | -0.004 | 471 |
| `D_range_pct` | existing | -0.0438 | -3.7 | 0.0051 | 100% | -0.004 | 471 |
| `N_dist_lo10` | location | -0.0299 | -3.7 | 0.0044 | 100% | -0.003 | 471 |
| `D_WQ_40` | existing | +0.0289 | +4.1 | 0.0017 | 100% | +0.003 | 471 |
| `D_breakout_high_20` | existing | -0.0251 | -4.0 | 0.002 | 80% | +nan | 471 |
| `D_mdi14_diff1` | existing | +0.0250 | +3.7 | 0.0044 | 100% | +0.003 | 471 |
| `D_pdi14` | existing | -0.0244 | -3.4 | 0.012 | 100% | -0.002 | 471 |
| `D_breakout_high_50` | existing | -0.0240 | -4.1 | 0.0018 | 100% | +nan | 471 |
| `R_atr_ratio_14_30__tsz60` | representation | -0.0234 | -3.1 | 0.025 | 100% | -0.002 | 471 |
| `D_weekly_trend` | existing | -0.0227 | -3.4 | 0.011 | 80% | +nan | 471 |
| `R_atr_ratio_14_30__d5` | representation | -0.0224 | -3.5 | 0.0075 | 100% | -0.003 | 471 |

## Sign flips - works one way here, the opposite way there

HYPOTHESES, not findings: selected from many tests, so each needs its own confirmation on data not used to find it.

A feature significant with OPPOSITE signs in two regimes. An unconditional model averages these into nothing; this is exactly the information regime conditioning exists to recover.

| feature | layer | + regime | + IC | - regime | - IC |
|---|---|---|---|---|---|
| `X_z_D_ret_skew_60` | stock | S5 | +0.0206 | S3 | -0.0251 |
| `D_ret_skew_60` | stock | S5 | +0.0206 | S3 | -0.0251 |
| `X_rank_D_ret_skew_60` | stock | S5 | +0.0206 | S3 | -0.0251 |

## Feature x regime matrix (top 40 by max |IC|)

`*` = working (FDR + stable + material). Blank = too few dates.

| feature | ALL | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | M0 | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `X_z_D_atr_pct` | -0.037* | -0.025 | -0.040* | -0.049* | -0.044* | -0.025 | -0.051* | -0.046* | -0.024 | -0.049* | +0.047 | -0.019 | -0.053 | -0.019 | -0.068 | -0.042 |
| `X_rank_D_atr_pct` | -0.037* | -0.025 | -0.040* | -0.049* | -0.044* | -0.025 | -0.051* | -0.046* | -0.024 | -0.049* | +0.047 | -0.019 | -0.053 | -0.019 | -0.068 | -0.042 |
| `D_atr_pct` | -0.037* | -0.025 | -0.040* | -0.049* | -0.044* | -0.025 | -0.051* | -0.046* | -0.024 | -0.049* | +0.047 | -0.019 | -0.053 | -0.019 | -0.068 | -0.042 |
| `R_atr_pct__csrank` | -0.037* | -0.025 | -0.040* | -0.049* | -0.044* | -0.025 | -0.051* | -0.046* | -0.024 | -0.049* | +0.047 | -0.019 | -0.053 | -0.019 | -0.068 | -0.042 |
| `R_atr_pct__csz` | -0.037* | -0.025 | -0.040* | -0.049* | -0.044* | -0.025 | -0.051* | -0.046* | -0.024 | -0.049* | +0.047 | -0.019 | -0.053 | -0.019 | -0.068 | -0.042 |
| `D_vol_yz_20` | -0.039* | -0.027 | -0.040* | -0.050* | -0.046* | -0.028 | -0.051* | -0.048* | -0.025 | -0.048* | +0.039 | -0.040 | -0.053* | -0.024 | -0.065 | -0.042 |
| `D_vol_yz_50` | -0.037* | -0.027 | -0.041* | -0.047* | -0.043* | -0.023 | -0.049* | -0.043* | -0.024 | -0.042* | +0.043 | -0.033 | -0.050* | -0.021 | -0.063 | -0.041 |
| `X_z_D_range_pct` | -0.041* | -0.021 | -0.024 | -0.042* | -0.052* | -0.019 | -0.045* | -0.052* | -0.021 | -0.050* | -0.005 | -0.039 | -0.053* | -0.023 | -0.061* | -0.044* |
| `X_rank_D_range_pct` | -0.041* | -0.021 | -0.024 | -0.042* | -0.052* | -0.019 | -0.045* | -0.052* | -0.021 | -0.050* | -0.005 | -0.039 | -0.053* | -0.023 | -0.061* | -0.044* |
| `D_range_pct` | -0.041* | -0.021 | -0.024 | -0.042* | -0.052* | -0.019 | -0.045* | -0.052* | -0.021 | -0.050* | -0.005 | -0.039 | -0.053* | -0.023 | -0.061* | -0.044* |
| `R_drawdown_252__d5` | -0.022* | -0.014 | -0.016 | -0.024 | -0.021 | -0.004 | -0.010 | -0.020 | -0.008 | -0.008 | -0.046 | -0.061 | -0.018 | -0.031 | -0.003 | -0.016 |
| `D_WQ_29` | +0.026* | +0.017 | +0.016 | +0.030* | +0.018 | +0.004 | +0.013 | +0.031* | +0.006 | +0.021 | +0.042 | +0.059 | +0.023 | +0.030 | +0.018 | +0.022* |
| `R_atr_ratio_14_30__d20` | -0.017* | -0.005 | -0.002 | -0.018 | -0.018 | -0.011 | -0.020 | -0.012 | -0.006 | -0.015 | -0.011 | -0.058 | -0.016 | -0.009 | -0.023 | -0.015 |
| `R_pos_in_52w_range__d5` | -0.023* | -0.017 | -0.016 | -0.026* | -0.016 | -0.007 | -0.012 | -0.020 | -0.010 | -0.013 | -0.045 | -0.057 | -0.018 | -0.030* | -0.009 | -0.019 |
| `X_z_D_realvol_20` | -0.034* | -0.021 | -0.033* | -0.041* | -0.041* | -0.026 | -0.006 | -0.042* | -0.017 | -0.038* | +0.026 | -0.016 | -0.047 | -0.021 | -0.056 | -0.037 |
| `X_relvol_20` | -0.034* | -0.021 | -0.033* | -0.041* | -0.041* | -0.026 | -0.006 | -0.042* | -0.017 | -0.038* | +0.026 | -0.016 | -0.047 | -0.021 | -0.056 | -0.037 |
| `R_realvol_20__csz` | -0.034* | -0.021 | -0.033* | -0.041* | -0.041* | -0.026 | -0.006 | -0.042* | -0.017 | -0.038* | +0.026 | -0.016 | -0.047 | -0.021 | -0.056 | -0.037 |
| `X_rank_D_realvol_20` | -0.034* | -0.021 | -0.033* | -0.041* | -0.041* | -0.026 | -0.006 | -0.042* | -0.017 | -0.038* | +0.026 | -0.016 | -0.047 | -0.021 | -0.056 | -0.037 |
| `D_realvol_20` | -0.034* | -0.021 | -0.033* | -0.041* | -0.041* | -0.026 | -0.006 | -0.042* | -0.017 | -0.038* | +0.026 | -0.016 | -0.047 | -0.021 | -0.056 | -0.037 |
| `R_realvol_20__csrank` | -0.034* | -0.021 | -0.033* | -0.041* | -0.041* | -0.026 | -0.006 | -0.042* | -0.017 | -0.038* | +0.026 | -0.016 | -0.047 | -0.021 | -0.056 | -0.037 |
| `R_rsi14__tsz60` | -0.017* | -0.003 | -0.008 | -0.019 | -0.014 | -0.002 | -0.007 | -0.024* | +0.003 | -0.013 | -0.022 | -0.055 | -0.020 | -0.019 | -0.005 | -0.015 |
| `R_donch_pos_20__tsz60` | -0.015* | -0.004 | -0.003 | -0.019 | -0.013 | -0.004 | -0.003 | -0.020 | +0.004 | -0.017 | -0.022 | -0.054 | -0.015 | -0.021 | -0.002 | -0.010 |
| `D_dist_from_20l` | -0.019* | -0.009 | -0.009 | -0.021 | -0.014 | -0.007 | +0.002 | -0.022* | +0.003 | -0.015 | -0.031 | -0.053 | -0.023 | -0.019 | -0.013 | -0.011 |
| `D_realvol_60` | -0.031* | -0.016 | -0.034* | -0.041* | -0.040* | -0.019 | -0.003 | -0.038* | -0.015 | -0.036* | +0.034 | -0.009 | -0.042 | -0.019 | -0.053 | -0.036 |
| `R_ema20_angle_deg__tsz60` | -0.020* | -0.005 | -0.011 | -0.023 | -0.016 | -0.005 | -0.008 | -0.029* | +0.001 | -0.019 | -0.027 | -0.052 | -0.022 | -0.023 | -0.006 | -0.016 |
| `R_bb_bw_20__csz` | -0.029* | -0.013 | -0.020 | -0.035* | -0.038* | -0.019 | -0.017 | -0.038* | -0.020 | -0.036* | +0.018 | -0.011 | -0.041 | -0.011 | -0.051 | -0.035 |
| `R_bb_bw_20__csrank` | -0.029* | -0.013 | -0.020 | -0.035* | -0.038* | -0.019 | -0.017 | -0.038* | -0.020 | -0.036* | +0.018 | -0.011 | -0.041 | -0.011 | -0.051 | -0.035 |
| `D_bb_bw_20` | -0.029* | -0.013 | -0.020 | -0.035* | -0.038* | -0.019 | -0.017 | -0.038* | -0.020 | -0.036* | +0.018 | -0.011 | -0.041 | -0.011 | -0.051 | -0.035 |
| `D_ret_5d_roll_std` | -0.029* | -0.013 | -0.030 | -0.037* | -0.036* | -0.016 | -0.004 | -0.034* | -0.016 | -0.034 | +0.031 | -0.018 | -0.042 | -0.013 | -0.051 | -0.032 |
| `R_drawdown_252__accel` | -0.020* | -0.018 | -0.015 | -0.021 | -0.017 | -0.008 | -0.008 | -0.016 | -0.012 | -0.006 | -0.038 | -0.051 | -0.012 | -0.028 | -0.013 | -0.015 |
| `R_dist_from_20h__tsz60` | -0.014* | -0.003 | -0.012 | -0.019 | -0.009 | -0.003 | -0.004 | -0.016 | +0.004 | -0.017 | -0.026 | -0.051 | -0.014 | -0.019 | +0.003 | -0.010 |
| `X_rank_D_ema20_angle_deg` | -0.019* | +0.009 | -0.000 | -0.022 | -0.029* | +0.013 | -0.013 | -0.040* | +0.011 | -0.034* | -0.039 | -0.051 | -0.030 | -0.013 | -0.012 | -0.010 |
| `D_ema20_angle_deg` | -0.019* | +0.009 | -0.000 | -0.022 | -0.029* | +0.013 | -0.013 | -0.040* | +0.011 | -0.034* | -0.039 | -0.051 | -0.030 | -0.013 | -0.012 | -0.010 |
| `R_ema20_angle_deg__csrank` | -0.019* | +0.009 | -0.000 | -0.022 | -0.029* | +0.013 | -0.013 | -0.040* | +0.011 | -0.034* | -0.039 | -0.051 | -0.030 | -0.013 | -0.012 | -0.010 |
| `R_ema20_angle_deg__csz` | -0.019* | +0.009 | -0.000 | -0.022 | -0.029* | +0.013 | -0.013 | -0.040* | +0.011 | -0.034* | -0.039 | -0.051 | -0.030 | -0.013 | -0.012 | -0.010 |
| `X_z_D_ema20_angle_deg` | -0.019* | +0.009 | -0.000 | -0.022 | -0.029* | +0.013 | -0.013 | -0.040* | +0.011 | -0.034* | -0.039 | -0.051 | -0.030 | -0.013 | -0.012 | -0.010 |
| `D_downside_dev_60` | -0.030* | -0.020 | -0.042* | -0.045* | -0.032* | -0.026 | -0.008 | -0.036* | -0.020 | -0.037* | +0.027 | -0.007 | -0.036 | -0.023 | -0.051 | -0.034 |
| `X_rank_D_downside_dev_60` | -0.030* | -0.020 | -0.042* | -0.045* | -0.032* | -0.026 | -0.008 | -0.036* | -0.020 | -0.037* | +0.027 | -0.007 | -0.036 | -0.023 | -0.051 | -0.034 |
| `X_z_D_downside_dev_60` | -0.030* | -0.020 | -0.042* | -0.045* | -0.032* | -0.026 | -0.008 | -0.036* | -0.020 | -0.037* | +0.027 | -0.007 | -0.036 | -0.023 | -0.051 | -0.034 |
| `R_ema20_angle_deg__d20` | -0.015* | +0.002 | +0.006 | -0.024 | -0.016 | +0.002 | -0.001 | -0.018 | +0.003 | -0.026 | -0.023 | -0.051 | -0.024 | -0.019 | -0.003 | -0.005 |

## Caveats

- Exploratory. A working cell is evidence to test, not a validated edge.
- About 5% of working cells are expected to be false discoveries by construction of the FDR bar.
- IC is cross-sectional ranking skill. It is not a return, and it ignores costs; `top-bot hit` is the gap in the mean target between the top and bottom quintile within the regime - a hit-rate gap for a 0/1 target, a return gap for label_exit_ret.
- Regime IDs are aligned across folds by centroid matching; check the drift figures before trusting a regime's identity over time.
- In-sample regime labels (before the first test window) never enter the statistics.