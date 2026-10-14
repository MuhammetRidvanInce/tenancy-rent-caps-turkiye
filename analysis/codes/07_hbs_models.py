"""Household Budget Survey replication and consumption outcomes: sitting vs new tenants.

Years 2016-2019 (reference period), 2022 (cap introduced in June; interviews spread over the year),
2024 (cap in force until July; interviews spread over the year). Base year in the event study: 2019.
Weights: FAKTOR. The public HBS files carry no region or interview month; standard errors are HC1.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy
from common import PROCESSED, TABLES, register, save_registry

H = pd.read_csv(PROCESSED / "hhba_kiraci_2016_2024.csv", low_memory=False)
H["sit"] = (H.tenure_years >= 2).astype(float)
H["lrent"] = np.log(H.KIRA_TUTAR)
c = H.cons_total
H["sh_rent"] = 100 * H.KIRA_TUTAR / c
H["sh_food"] = 100 * H.c_food / c
H["sh_health"] = 100 * H.c_health / c
H["sh_education"] = 100 * H.c_education / c
H["sh_transport"] = 100 * H.c_transport / c
H["sh_restaurants"] = 100 * H.c_restaurants / c
H["lcons"] = np.log(c)
H["lfood"] = np.log(H.c_food.clip(lower=1))
H["lcons_nonrent"] = np.log((c - H.KIRA_TUTAR).clip(lower=1))
pc = H.purch_total
H["psh_rent"] = 100 * H.KIRA_TUTAR / pc
H["psh_food"] = 100 * H.p_food / pc
OUT = {"lrent": "Log aylık kira", "sh_rent": "Kira payı (%)", "sh_food": "Gıda payı (%)", "sh_health": "Sağlık payı (%)",
       "sh_education": "Eğitim payı (%)", "sh_transport": "Ulaştırma payı (%)", "sh_restaurants": "Lokanta-otel payı (%)",
       "lcons": "Log toplam tüketim", "lcons_nonrent": "Log kira dışı tüketim", "lfood": "Log gıda harcaması",
       "psh_rent": "Kira payı, yalnız satın alış (%)", "psh_food": "Gıda payı, yalnız satın alış (%)"}
CTRL = "C(hhsize_g) + C(child_g) + C(rooms) + log_area + C(heating) + C(KONUT_TIP)"
YEARS = [2016, 2017, 2018, 2022, 2024]
for y in YEARS:
    H[f"sit_{y}"] = H.sit * (H.year == y)
H["sit_P22"] = H.sit * (H.year == 2022)
H["sit_P24"] = H.sit * (H.year == 2024)


def fit(formula):
    yv, X = patsy.dmatrices(formula, H, return_type="dataframe", NA_action="drop")
    m = sm.WLS(yv, X, weights=H.loc[yv.index, "FAKTOR"]).fit(cov_type="HC1")
    return m, yv.index


es, pooled = [], []
for dv, lab in OUT.items():
    m, idx = fit(f"{dv} ~ sit + " + " + ".join(f"sit_{y}" for y in YEARS) + f" + C(year) + {CTRL}")
    for y in YEARS:
        r = register(f"HBS_ES_{dv}_{y}", dv, "HHBA kiracı 2016-2024", "olay çalışması, HC1", f"sit_{y}", m, len(idx))
        es.append({"outcome": dv, "label": lab, "year": y, "coef": r["coef"], "se": r["se"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"]})
    es.append({"outcome": dv, "label": lab, "year": 2019, "coef": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": np.nan})
    ft = m.f_test("sit_2016 = 0, sit_2017 = 0, sit_2018 = 0")
    pooled.append({"outcome": dv, "label": lab, "term": "pretrend_F_2016_2018_base2019", "coef": float(np.squeeze(ft.fvalue)),
                   "se": np.nan, "p": float(ft.pvalue), "n": len(idx)})
    mp, idxp = fit(f"{dv} ~ sit + sit_P22 + sit_P24 + C(year) + {CTRL}")
    sub = H.loc[idxp]
    ybar = float(np.average(sub[(sub.sit == 1) & (sub.year <= 2019)][dv], weights=sub[(sub.sit == 1) & (sub.year <= 2019)].FAKTOR))
    for t in ["sit", "sit_P22", "sit_P24"]:
        r = register(f"HBS_POOL_{dv}_{t}", dv, "HHBA kiracı 2016-2024", "dönemlik DiD, HC1", t, mp, len(idxp), extra={"mean_dep_sitting_pre": ybar})
        pooled.append({"outcome": dv, "label": lab, "term": t, "coef": r["coef"], "se": r["se"], "p": r["p"],
                       "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": len(idxp), "mean_dep_sitting_pre": ybar})
pd.DataFrame(es).sort_values(["outcome", "year"]).to_csv(TABLES / "T6_hhba_olay_calismasi.csv", index=False, encoding="utf-8")
pd.DataFrame(pooled).to_csv(TABLES / "T6b_hhba_donemlik_did.csv", index=False, encoding="utf-8")
desc = H.groupby(["year", "sit"]).apply(lambda g: pd.Series({"N": len(g), **{k: np.average(g[k], weights=g.FAKTOR) for k in ["sh_rent", "sh_food", "sh_education", "sh_health"]},
                                                              "median_rent": g.KIRA_TUTAR.median()})).reset_index()
desc.to_csv(TABLES / "T6c_hhba_tanimlayici.csv", index=False, encoding="utf-8")
save_registry("07_hhba_model")
print(pd.DataFrame(pooled).round(4).to_string()); print(desc.round(2).to_string())
