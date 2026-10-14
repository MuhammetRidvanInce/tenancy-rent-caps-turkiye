"""Robustness of the pooled SILC estimates (equation 2) for log rent, heavy burden and rent arrears."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import patsy
from common import PROCESSED, TABLES, wls_cluster, register, save_registry, QUAKE_REGIONS, SEED

S0 = pd.read_csv(PROCESSED / "gyka_kiraci_analiz_2016_2025.csv", low_memory=False)
CTRL = "C(dwelling) + C(rooms) + log_area + C(heating) + C(hhsize_g) + C(child_g) + C(head_age_g) + head_female"
TERMS = ["sit_P21", "sit_P22", "sit_P2324", "sit_P25"]


def prep(S, newmax=1, sitmin=2):
    S = S[(S.tenure_years <= newmax) | (S.tenure_years >= sitmin)].copy()
    S["sit"] = (S.tenure_years >= sitmin).astype(float)
    S["sit_P21"] = S.sit * (S.year == 2021)
    S["sit_P22"] = S.sit * (S.year == 2022)
    S["sit_P2324"] = S.sit * S.year.isin([2023, 2024])
    S["sit_P25"] = S.sit * (S.year == 2025)
    S["sit_trend"] = S.sit * (S.year - 2019)
    S["w1"] = 1.0
    return S


def ipw(S):
    """Within-year propensity of being a sitting tenant; ATT weights p/(1-p) for new tenants."""
    S = S.copy(); S["w_ipw"] = np.nan
    for y, g in S.groupby("year"):
        X = patsy.dmatrix(f"{CTRL} + C(HB031)", g, return_type="dataframe")
        g = g.loc[X.index]
        lr = LogisticRegression(max_iter=3000, C=1e6, random_state=SEED).fit(X, g.sit, sample_weight=g.HB040)
        p = np.clip(lr.predict_proba(X)[:, 1], 0.02, 0.98)
        S.loc[X.index, "w_ipw"] = g.loc[X.index, "HB040"] * np.where(g.loc[X.index, "sit"] == 1, 1.0, p / (1 - p))
    return S


base = prep(S0)
SPECS = [
    ("R0 Ana model", base, f"C(ry) + {CTRL}", "HB040", TERMS),
    ("R1 Bölge FE + yıl FE", base, f"C(HB031) + C(year) + {CTRL}", "HB040", TERMS),
    ("R2 Kontrolsüz (yalnız bölge×yıl FE)", base, "C(ry)", "HB040", TERMS),
    ("R3 Ağırlıksız", base, f"C(ry) + {CTRL}", "w1", TERMS),
    ("R4 2020 hariç", base[base.year != 2020], f"C(ry) + {CTRL}", "HB040", TERMS),
    ("R5 Deprem bölgeleri hariç", base[~base.HB031.isin(QUAKE_REGIONS)], f"C(ry) + {CTRL}", "HB040", TERMS),
    ("R6 Yeni kiracı = 0 yıl", prep(S0, 0, 2), f"C(ry) + {CTRL}", "HB040", TERMS),
    ("R7 Yeni ≤2 yıl, mevcut ≥3 yıl", prep(S0, 2, 3), f"C(ry) + {CTRL}", "HB040", TERMS),
    ("R8 IPW (yıl içi eğilim skoru)", ipw(base), f"C(ry) + {CTRL}", "w_ipw", TERMS),
    ("R9 Mevcut kiracıya özgü doğrusal trend", base, f"sit_trend + C(ry) + {CTRL}", "HB040", TERMS),
    ("R11 2021 referansa dahil (eski tanım)", base.assign(sit_P21=0.0), f"C(ry) + {CTRL}", "HB040", ["sit_P22", "sit_P2324", "sit_P25"]),
]
rows = []
for name, S, rhs, w, terms in SPECS:
    for dv in ["lrent", "heavy", "arr_rent"]:
        f = f"{dv} ~ sit + " + " + ".join(terms) + f" + {rhs}"
        m, X, y, idx = wls_cluster(f, S, w, "HB031")
        G = S.loc[idx, "HB031"].nunique()
        for t in terms:
            r = register(f"ROB_{name[:3].strip()}_{dv}_{t}", dv, name, "sağlamlık", t, m, len(idx), G)
            rows.append({"spec": name, "outcome": dv, "term": t, "coef": r["coef"], "se": r["se"], "p": r["p"], "n": len(idx), "clusters": G})
# placebo: reference period only, fake post 2019-2020 vs 2016-2018
pl = base[base.year <= 2020].copy(); pl["sit_fake"] = pl.sit * (pl.year >= 2019)
for dv in ["lrent", "heavy", "arr_rent"]:
    m, X, y, idx = wls_cluster(f"{dv} ~ sit + sit_fake + C(ry) + {CTRL}", pl, "HB040", "HB031")
    r = register(f"ROB_PL_{dv}", dv, "Plasebo 2016-2020 (sahte dönem 2019-2020)", "plasebo", "sit_fake", m, len(idx), 26)
    rows.append({"spec": "R10 Plasebo: sahte dönem 2019-2020", "outcome": dv, "term": "sit_fake", "coef": r["coef"], "se": r["se"], "p": r["p"], "n": len(idx), "clusters": 26})
R = pd.DataFrame(rows)
R.to_csv(TABLES / "T7_saglamlik.csv", index=False, encoding="utf-8")
save_registry("08_saglamlik")
print(R.round(4).to_string())
