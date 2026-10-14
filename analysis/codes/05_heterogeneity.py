"""Dose (tenure length) and heterogeneity (equivalised income quintile, large metropolitan regions).

Tenure bins: new (0-1 years, reference), 2-4, 5-9, 10+ years. Under the Turkish Code of Obligations
(art. 344), landlords may sue for a court-determined market rent after five years of tenancy, so
tenants with 5+ years were less insulated from market rents than tenants with 2-4 years.
"""
import numpy as np
import pandas as pd
from common import (PROCESSED, TABLES, wls_cluster, register, register_lincom, save_registry,
                    lincom, wquantile, METRO_REGIONS, WildClusterBootstrap)

S = pd.read_csv(PROCESSED / "gyka_kiraci_analiz_2016_2025.csv", low_memory=False)
CTRL = "C(dwelling) + C(rooms) + log_area + C(heating) + C(hhsize_g) + C(child_g) + C(head_age_g) + head_female"
PER = {"P21": S.year == 2021, "P22": S.year == 2022, "P2324": S.year.isin([2023, 2024]), "P25": S.year == 2025}
OUT = ["lrent", "heavy", "arr_rent", "rent_burden40"]
rows = []

# (a) tenure bins
S["b2_4"] = S.tenure_years.between(2, 4).astype(float)
S["b5_9"] = S.tenure_years.between(5, 9).astype(float)
S["b10p"] = (S.tenure_years >= 10).astype(float)
terms = []
for b in ["b2_4", "b5_9", "b10p"]:
    for p, msk in PER.items():
        S[f"{b}_{p}"] = S[b] * msk
        terms.append(f"{b}_{p}")
for dv in OUT:
    f = f"{dv} ~ b2_4 + b5_9 + b10p + " + " + ".join(terms) + f" + C(ry) + {CTRL}"
    m, X, y, idx = wls_cluster(f, S, "HB040", "HB031")
    for t in terms:
        r = register(f"DOSE_{dv}_{t}", dv, "GYKA kiracı 2016-2025", "kiracılık süresi dilimleri", t, m, len(idx), 26)
        rows.append({"analysis": "tenure_bins", "outcome": dv, "term": t, "coef": r["coef"], "se": r["se"], "p": r["p"],
                     "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": len(idx)})
    for p in PER:
        est, se, pv = lincom(m, {f"b2_4_{p}": 1, f"b5_9_{p}": -1})
        rows.append({"analysis": "tenure_bins", "outcome": dv, "term": f"diff_2_4_minus_5_9_{p}", "coef": est, "se": se, "p": pv, "n": len(idx)})

# (b) equivalised income quintile among tenants, re-ranked within year (income refers to the previous year)
S["incq"] = np.nan
for y, g in S.groupby("year"):
    cuts = [wquantile(g.eq_income, g.HB040, q) for q in (.2, .4, .6, .8)]
    S.loc[g.index, "incq"] = np.searchsorted(cuts, g.eq_income.to_numpy(), side="right") + 1
for q in range(1, 6):
    sub = S[S.incq == q].copy()
    for dv in ["lrent", "heavy", "arr_rent"]:
        f = f"{dv} ~ sit + sit_P21 + sit_P22 + sit_P2324 + sit_P25 + C(ry) + {CTRL}"
        sub["sit_P21"] = sub.sit * (sub.year == 2021); sub["sit_P22"] = sub.sit * (sub.year == 2022); sub["sit_P2324"] = sub.sit * sub.year.isin([2023, 2024]); sub["sit_P25"] = sub.sit * (sub.year == 2025)
        m, X, y, idx = wls_cluster(f, sub, "HB040", "HB031")
        for t in ["sit_P21", "sit_P22", "sit_P2324", "sit_P25"]:
            r = register(f"INC_Q{q}_{dv}_{t}", dv, f"GYKA kiracı, eşdeğer gelir Q{q}", "dönemlik DiD", t, m, len(idx), 26)
            rows.append({"analysis": f"income_Q{q}", "outcome": dv, "term": t, "coef": r["coef"], "se": r["se"], "p": r["p"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": len(idx)})

# Joint Q5-Q1 tests: all slopes AND region-year effects differ by quintile.
# This reproduces the separate point estimates and retains cross-quintile
# covariance from their shared regional clusters. Added after the original analysis.
for p, mask in PER.items():
    S[f"sit_{p}"] = S.sit * mask
S["ryq"] = S.ry + "_" + S.incq.astype(str)
fq = f"lrent ~ C(incq)*(sit + sit_P21 + sit_P22 + sit_P2324 + sit_P25 + {CTRL})"
mq, Xq, yq, iq = wls_cluster(fq, S, "HB040", "HB031", fe="ryq", rank_revealing=True)
bootq = WildClusterBootstrap(mq, Xq, yq, S, "HB040", "HB031")
joint_rows = []
for period in ["P2324", "P25"]:
    term = f"C(incq)[T.5.0]:sit_{period}"
    r = register_lincom(f"INC_JOINT_Q5_Q1_{period}", "lrent", "GYKA kiracı 2016-2025",
                        "gelir dilimine özgü kontroller ve bölge×yıl FE", f"Q5_minus_Q1_{period}",
                        mq, {term: 1}, len(iq), 26, bootq)
    separate = [next(x["coef"] for x in rows if x["analysis"] == f"income_Q{q}" and
                     x["outcome"] == "lrent" and x["term"] == f"sit_{period}") for q in [1, 5]]
    assert np.isclose(r["coef"], separate[1] - separate[0], atol=1e-10)
    joint_rows.append(r)
pd.DataFrame(joint_rows).to_csv(TABLES / "T4b_gelir_dilimleri_fark_testi.csv", index=False, encoding="utf-8")

# (c) three largest metropolitan regions (Istanbul, Ankara, Izmir) vs other regions: interaction model
S["metro"] = S.HB031.isin(METRO_REGIONS).astype(float)
for p in PER:
    S[f"sit_{p}"] = S.sit * PER[p]
    S[f"sitm_{p}"] = S[f"sit_{p}"] * S.metro
for dv in ["lrent", "heavy", "arr_rent"]:
    f = f"{dv} ~ sit + sit:metro + sit_P21 + sit_P22 + sit_P2324 + sit_P25 + sitm_P21 + sitm_P22 + sitm_P2324 + sitm_P25 + C(ry) + {CTRL}"
    m, X, y, idx = wls_cluster(f, S, "HB040", "HB031")
    for t in ["sit_P2324", "sitm_P2324", "sit_P25", "sitm_P25"]:
        r = register(f"METRO_{dv}_{t}", dv, "GYKA kiracı 2016-2025", "büyükşehir etkileşimi (3 bölge)", t, m, len(idx), 26)
        rows.append({"analysis": "metro_interaction", "outcome": dv, "term": t, "coef": r["coef"], "se": r["se"], "p": r["p"],
                     "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": len(idx)})

R = pd.DataFrame(rows)
R.to_csv(TABLES / "T4_doz_heterojenlik.csv", index=False, encoding="utf-8")
save_registry("05_doz_heterojenlik")
print(R.round(4).to_string())
