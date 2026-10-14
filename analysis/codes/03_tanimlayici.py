"""Descriptive statistics (SILC tenants) and the housing-cost (HH050) measurement diagnostic."""
import numpy as np
import pandas as pd
from common import PROCESSED, TABLES, wmean, wquantile, METRO_REGIONS

D = pd.read_csv(PROCESSED / "gyka_hane_2014_2025.csv", low_memory=False)
T = D[(D.HH020 == 2) & (D.HH040 > 0) & (D.HG110 > 0) & D.tenure_years.between(0, 80)].copy()
T["new"] = (T.tenure_years <= 1).astype(int)
T["rent_inc"] = T.HH040 * 12 / T.HG110
T["metro"] = T.HB031.isin(METRO_REGIONS).astype(float)

def period(y):
    return {2016: "2016-2020", 2017: "2016-2020", 2018: "2016-2020", 2019: "2016-2020", 2020: "2016-2020",
            2021: "2021", 2022: "2022", 2023: "2023-2024", 2024: "2023-2024", 2025: "2025"}.get(y)

T["period"] = T.year.map(period)
S = T[T.period.notna()]
rows = []
for (p, g), q in S.groupby(["period", "new"]):
    w = q.HB040
    rows.append({"period": p, "group": "new" if g == 1 else "sitting", "N": len(q),
                 "median_rent_TL": wquantile(q.HH040, w, .5), "median_rent_to_income_pct": 100 * wquantile(q.rent_inc, w, .5),
                 "rent_burden_gt40_pct": 100 * wmean(q.rent_inc > .40, w), "heavy_burden_pct": 100 * wmean(q.HE060 == 1, w),
                 "rent_arrears_pct": 100 * wmean(q.HE010.isin([1, 2]), w), "utility_arrears_pct": 100 * wmean(q.HE020.isin([1, 2]), w),
                 "hard_to_make_ends_meet_pct": 100 * wmean(q.HE040.isin([1, 2]), w),
                 "median_eq_income_TL": wquantile(q.eq_income, w, .5), "rooms": wmean(q.HH060, w), "area_m2": wmean(q.HH070, w),
                 "hh_size": wmean(q.hh_size, w), "head_age": wmean(q.head_age, w), "metro_pct": 100 * wmean(q.metro, w),
                 "tenure_years_mean": wmean(q.tenure_years, w)})
T1 = pd.DataFrame(rows)
T1["order"] = T1.period.map({"2016-2020": 0, "2021": 1, "2022": 2, "2023-2024": 3, "2025": 4})
T1 = T1.sort_values(["order", "group"]).drop(columns="order")
T1.to_csv(TABLES / "T1_tanimlayici_kiracilar.csv", index=False, encoding="utf-8")

# yearly raw series (Figure 1)
ser = []
for y, q in T[T.year >= 2014].groupby("year"):
    a, b = q[q.new == 1], q[q.new == 0]
    ser.append({"year": y, "median_rent_new": wquantile(a.HH040, a.HB040, .5), "median_rent_sitting": wquantile(b.HH040, b.HB040, .5),
                "mean_log_rent_gap": wmean(np.log(b.HH040), b.HB040) - wmean(np.log(a.HH040), a.HB040),
                "share_new_pct": 100 * wmean(q.new, q.HB040), "N": len(q)})
pd.DataFrame(ser).to_csv(TABLES / "T1b_yillik_kira_serisi.csv", index=False, encoding="utf-8")

# HH050 measurement diagnostic, all tenure types (Appendix)
diag = []
for y, q in D[D.HH020.isin([1, 2]) & (D.HG110 > 0)].groupby("year"):
    o, t = q[q.HH020 == 1], q[q.HH020 == 2]
    to = t[t.HH050.notna()]
    oo = o[o.HH050.notna()]
    diag.append({"year": y, "owners_HH050_missing_pct": 100 * o.HH050.isna().mean(),
                 "tenants_total_cost_equals_rent_pct": 100 * wmean(t.HH050 == t.HH040, t.HB040),
                 "tenants_median_nonrent_cost_TL": wquantile((to.HH050 - to.HH040).clip(lower=0), to.HB040, .5),
                 "tenants_overburden_total_pct": 100 * wmean(to.HH050 * 12 / to.HG110 > .40, to.HB040),
                 "tenants_overburden_rent_only_pct": 100 * wmean(t.HH040 * 12 / t.HG110 > .40, t.HB040),
                 "owners_overburden_total_pct": 100 * wmean(oo.HH050 * 12 / oo.HG110 > .40, oo.HB040)})
pd.DataFrame(diag).to_csv(TABLES / "A3_hh050_olcum_tanisi.csv", index=False, encoding="utf-8")
print(T1.round(2).to_string()); print(pd.DataFrame(ser).round(3).to_string()); print(pd.DataFrame(diag).round(2).to_string())
