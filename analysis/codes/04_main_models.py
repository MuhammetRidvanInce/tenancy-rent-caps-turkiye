"""Main design (SILC 2016-2025): sitting tenants (>=2 years in the dwelling) vs new tenants (<=1 year).

Equation (1): event study with region-by-year fixed effects, base year 2019 (reference period 2016-2020).
Equation (2): calendar periods 2021, 2022, 2023-2024, 2025; reference 2016-2020.
These estimate conditional gap changes, not a separately identified causal policy effect.
A base-2021 event study (original specification) is kept as a robustness table (T2b).
Weights: SILC household weights (HB040). Inference: CR1 clustered by 26 NUTS-2 regions with t(25);
restricted wild cluster bootstrap-t (Webb weights, B=9999) for coefficients and contrasts.
"""
import numpy as np
import pandas as pd
from common import (PROCESSED, TABLES, wls_cluster, WildClusterBootstrap, register,
                    register_lincom, save_registry, lincom)

D = pd.read_csv(PROCESSED / "gyka_hane_2014_2025.csv", low_memory=False)
S = D[(D.HH020 == 2) & (D.HH040 > 0) & (D.HG110 > 0) & D.tenure_years.between(0, 80) & D.year.between(2016, 2025)].copy()
S["sit"] = (S.tenure_years >= 2).astype(float)
S["lrent"] = np.log(S.HH040)
S["rent_burden40"] = (S.HH040 * 12 / S.HG110 > 0.40).astype(float)
S["heavy"] = (S.HE060 == 1).astype(float)
S["arr_rent"] = S.HE010.isin([1, 2]).astype(float)
S["arr_util"] = S.HE020.isin([1, 2]).astype(float)
S["hard"] = S.HE040.isin([1, 2]).astype(float)
S["ry"] = S.HB031 + "_" + S.year.astype(str)
S.to_csv(PROCESSED / "gyka_kiraci_analiz_2016_2025.csv", index=False, encoding="utf-8")

CTRL = "C(dwelling) + C(rooms) + log_area + C(heating) + C(hhsize_g) + C(child_g) + C(head_age_g) + head_female"
OUTCOMES = {"lrent": "Log aylık kira", "rent_burden40": "Kira/gelir > %40", "heavy": "Konut masrafı çok yük",
            "arr_rent": "Kira/borç ödemesinde gecikme", "arr_util": "Fatura ödemesinde gecikme", "hard": "Geçinmekte zorlanma"}
YEARS = [y for y in range(2016, 2026) if y != 2019]
YEARS21 = [y for y in range(2016, 2026) if y != 2021]
for y in range(2016, 2026):
    S[f"sit_{y}"] = S.sit * (S.year == y)
S["sit_P21"] = S.sit * (S.year == 2021)
S["sit_P22"] = S.sit * (S.year == 2022)
S["sit_P2324"] = S.sit * S.year.isin([2023, 2024])
S["sit_P25"] = S.sit * (S.year == 2025)

es_rows, pooled_rows, es21, level_rows = [], [], [], []
for dv, lab in OUTCOMES.items():
    f_es = f"{dv} ~ sit + " + " + ".join(f"sit_{y}" for y in YEARS) + f" + C(ry) + {CTRL}"
    m, X, yv, idx = wls_cluster(f_es, S, "HB040", "HB031")
    for y in YEARS:
        r = register(f"ES_{dv}_{y}", dv, "GYKA kiracı 2016-2025", "olay çalışması, bölge×yıl SE", f"sit_{y}", m, len(idx), 26)
        es_rows.append({"outcome": dv, "label": lab, "year": y, "coef": r["coef"], "se": r["se"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"]})
    es_rows.append({"outcome": dv, "label": lab, "year": 2019, "coef": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": np.nan})
    # joint pre-trend test: 2016, 2017, 2018, 2020 relative to 2019
    pre = ["sit_2016", "sit_2017", "sit_2018", "sit_2020"]
    ft = m.f_test(" , ".join(f"{t} = 0" for t in pre))
    pooled_rows.append({"outcome": dv, "label": lab, "term": "pretrend_F_2016_2020_base2019", "coef": float(np.squeeze(ft.fvalue)),
                        "se": np.nan, "p": float(ft.pvalue), "p_wcb": np.nan, "n": len(idx)})
    f_p = f"{dv} ~ sit + sit_P21 + sit_P22 + sit_P2324 + sit_P25 + C(ry) + {CTRL}"
    mp, Xp, yp, idxp = wls_cluster(f_p, S, "HB040", "HB031")
    bootstrap = WildClusterBootstrap(mp, Xp, yp, S, "HB040", "HB031")
    ybar = float(np.average(S.loc[idxp][S.loc[idxp].sit == 1][dv], weights=S.loc[idxp][S.loc[idxp].sit == 1].HB040))
    for term in ["sit", "sit_P21", "sit_P22", "sit_P2324", "sit_P25"]:
        boot = bootstrap.test(term) if term != "sit" else {"p_wcb": np.nan}
        pw = boot["p_wcb"]
        r = register(f"POOL_{dv}_{term}", dv, "GYKA kiracı 2016-2025", "dönemlik DiD, bölge×yıl FE", term, mp, len(idxp), 26, pw,
                     extra={"mean_dep_sitting": ybar, **boot})
        pooled_rows.append({**r, "label": lab})
    for name, wts in [("cap_minus_2022", {"sit_P2324": 1, "sit_P22": -1}), ("2025_minus_cap", {"sit_P25": 1, "sit_P2324": -1})]:
        r = register_lincom(f"POOL_{dv}_{name}", dv, "GYKA kiracı 2016-2025",
                            "dönemler arası koşullu fark değişimi", name, mp, wts, len(idxp), 26, bootstrap)
        pooled_rows.append({**r, "label": lab})
    if dv == "lrent":
        for period, term in [("2016-2020", None), ("2021", "sit_P21"), ("2022", "sit_P22"),
                             ("2023-2024", "sit_P2324"), ("2025", "sit_P25")]:
            wts = {"sit": 1} if term is None else {"sit": 1, term: 1}
            r = register_lincom(f"LEVEL_lrent_{period}", dv, "GYKA kiracı 2016-2025",
                                "toplam koşullu log kira farkı", period, mp, wts, len(idxp), 26)
            change = 0.0 if term is None else float(mp.params[term])
            level_rows.append({"period": period, "change_log": change,
                               "ratio_change_pct": 100 * np.expm1(change),
                               "level_log": r["coef"], "level_se": r["se"],
                               "level_gap_pct": 100 * np.expm1(r["coef"]),
                               "level_ci_low_pct": 100 * np.expm1(r["ci_low"]),
                               "level_ci_high_pct": 100 * np.expm1(r["ci_high"]), "n": len(idxp)})
    # robustness: original base-2021 event study
    f21 = f"{dv} ~ sit + " + " + ".join(f"sit_{y}" for y in YEARS21) + f" + C(ry) + {CTRL}"
    m21, *_ = wls_cluster(f21, S, "HB040", "HB031")
    for y in YEARS21:
        es21.append({"outcome": dv, "year": y, "coef": float(m21.params[f"sit_{y}"]), "se": float(m21.bse[f"sit_{y}"]), "p": float(m21.pvalues[f"sit_{y}"])})
    ft21 = m21.f_test(" , ".join(f"sit_{y} = 0" for y in range(2016, 2021)))
    es21.append({"outcome": dv, "year": "F_2016_2020", "coef": float(np.squeeze(ft21.fvalue)), "se": np.nan, "p": float(ft21.pvalue)})
    print(dv, "done", flush=True)

pd.DataFrame(es_rows).sort_values(["outcome", "year"]).to_csv(TABLES / "T2_olay_calismasi.csv", index=False, encoding="utf-8")
pd.DataFrame(es21).to_csv(TABLES / "T2b_olay_calismasi_baz2021.csv", index=False, encoding="utf-8")
P = pd.DataFrame(pooled_rows)
P.to_csv(TABLES / "T3_donemlik_did.csv", index=False, encoding="utf-8")
pd.DataFrame(level_rows).to_csv(TABLES / "T3b_kira_farki_duzey_degisim.csv", index=False, encoding="utf-8")
save_registry("04_ana_model")
print(P.round(4).to_string())
