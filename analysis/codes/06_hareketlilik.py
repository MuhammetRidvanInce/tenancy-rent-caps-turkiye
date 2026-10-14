"""Residential mobility among tenants: share of new tenants (<=1 year in dwelling), SILC 2014-2025.

Equation (3): new_i = a_r + g*(t-2019) + sum_{k=2021..2025} d_k 1[t=k] + controls, estimated on all
tenant households 2014-2025; d_k are deviations from the 2014-2020 linear trend.
"""
import numpy as np
import pandas as pd
from common import PROCESSED, TABLES, wls_cluster, register, save_registry, wmean

D = pd.read_csv(PROCESSED / "gyka_hane_2014_2025.csv", low_memory=False)
T = D[(D.HH020 == 2) & (D.HH040 > 0) & (D.HG110 > 0) & D.tenure_years.between(0, 80)].copy()
T["new"] = (T.tenure_years <= 1).astype(float)
T["trend"] = T.year - 2019
for y in (2021, 2022, 2023, 2024, 2025):
    T[f"d{y}"] = (T.year == y).astype(float)
CTRL = "C(dwelling) + C(rooms) + log_area + C(hhsize_g) + C(child_g) + C(head_age_g) + head_female"
rows = []
for spec, f in [("trend_only", "new ~ trend + d2021 + d2022 + d2023 + d2024 + d2025 + C(HB031)"),
                ("trend_controls", f"new ~ trend + d2021 + d2022 + d2023 + d2024 + d2025 + C(HB031) + {CTRL}")]:
    m, X, y, idx = wls_cluster(f, T, "HB040", "HB031")
    for t in ["trend", "d2021", "d2022", "d2023", "d2024", "d2025"]:
        r = register(f"MOB_{spec}_{t}", "new", "GYKA kiracı 2014-2025", spec, t, m, len(idx), 26)
        rows.append({"spec": spec, "term": t, "coef": r["coef"], "se": r["se"], "p": r["p"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": len(idx)})
share = T.groupby("year").apply(lambda g: pd.Series({"share_new_pct": 100 * wmean(g.new, g.HB040), "N": len(g)})).reset_index()
pd.DataFrame(rows).to_csv(TABLES / "T5_hareketlilik.csv", index=False, encoding="utf-8")
share.to_csv(TABLES / "T5b_yeni_kiraci_payi.csv", index=False, encoding="utf-8")
save_registry("06_hareketlilik")
print(pd.DataFrame(rows).round(4).to_string()); print(share.round(2).to_string())
