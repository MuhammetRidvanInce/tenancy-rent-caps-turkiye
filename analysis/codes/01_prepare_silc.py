"""SILC (GYKA) cross-sections 2014-2025: household-level analysis file.

Output: datas/processed/gyka_hane_2014_2025.csv (all tenure types) and a sample-flow table.
Income (HG110) refers to the previous calendar year; housing variables to the interview (Feb-Jun).
"""
import numpy as np
import pandas as pd
from common import resolve, read_tuik, num, write_manifest, PROCESSED, TABLES, REGION_RECODE

YEARS = range(2014, 2026)
rows, flow = [], []
for y in YEARS:
    h = read_tuik(resolve(y, "GYKA", "h"))
    fk = read_tuik(resolve(y, "GYKA", "fk"))
    if "BULTEN" not in fk:
        fk["BULTEN"] = num(fk["FERT_ID"]) // 100
    fk["age"] = num(fk["FK070"])
    fk["female"] = (num(fk["FK090"]) == 2).astype(float)
    fk["head"] = num(fk["FK095"]) == 1
    comp = fk.groupby("BULTEN").agg(hh_size=("FERT_ID", "size"),
                                    n_child=("age", lambda a: int((a < 15).sum())),
                                    n_14p=("age", lambda a: int((a >= 14).sum())),
                                    n_65p=("age", lambda a: int((a >= 65).sum())))
    comp["eq_scale"] = 1 + 0.5 * (comp.n_14p - 1).clip(lower=0) + 0.3 * (comp.hh_size - comp.n_14p)
    head = fk[fk["head"]].drop_duplicates("BULTEN").set_index("BULTEN")[["age", "female"]]
    head.columns = ["head_age", "head_female"]
    keep = ["BULTEN", "HB031", "HB040", "HH010", "HH020", "HH030", "HH040", "HH050", "HH060", "HH070",
            "HH080", "HG110", "HG010", "HG040", "HE010", "HE020", "HE040", "HE060", "HE100", "HE110"]
    d = h[[c for c in keep if c in h]].copy()
    for c in d.columns:
        if c != "HB031":
            d[c] = num(d[c])
    d["HB031"] = d["HB031"].astype(str).str.strip().replace(REGION_RECODE)
    d = d.merge(comp, left_on="BULTEN", right_index=True, how="left").merge(head, left_on="BULTEN", right_index=True, how="left")
    d.insert(0, "year", y)
    rows.append(d)
    t = d[d.HH020 == 2]
    flow.append({"year": y, "households": len(d), "tenants": len(t),
                 "tenants_rent_pos": int((t.HH040 > 0).sum()),
                 "tenants_valid_tenure": int(((t.HH040 > 0) & t.HH030.between(1900, y)).sum()),
                 "tenants_income_pos": int(((t.HH040 > 0) & t.HH030.between(1900, y) & (t.HG110 > 0)).sum()),
                 "owners": int((d.HH020 == 1).sum()),
                 "owners_HH050_missing": int(((d.HH020 == 1) & d.HH050.isna()).sum())})

D = pd.concat(rows, ignore_index=True)
# harmonised dwelling type (code list changed in 2018; both lists from the TurkStat SILC dictionaries)
# <=2017: 1 detached, 2 semi-detached/terraced, 3 apartment <10 flats, 4 apartment 10+ flats, 5 other
# >=2018: 11 single, 121/122 two-dwelling, 131/132 3-9 dwellings, 141/142 10+ dwellings, 2 other
ht = D.HH010
D["dwelling"] = np.select([ht.isin([1, 2, 11, 121, 122]), ht.isin([3, 131, 132]), ht.isin([4, 141, 142])],
                          ["detached", "apt_small", "apt_large"], "other")
pre = (D.year < 2018) & (ht == 2)
D.loc[pre, "dwelling"] = "detached"
D.loc[(D.year >= 2018) & (ht == 2), "dwelling"] = "other"
D.loc[(D.year < 2018) & (ht == 5), "dwelling"] = "other"
D["rooms"] = D.HH060.clip(upper=6)
D["log_area"] = np.log(D.HH070.clip(20, 400))
D["heating"] = D.HH080.fillna(9).astype(int).clip(upper=9)
D["tenure_years"] = D.year - D.HH030
D["eq_income"] = D.HG110 / D.eq_scale
D["head_age_g"] = pd.cut(D.head_age, [0, 29, 39, 49, 64, 120], labels=["<30", "30-39", "40-49", "50-64", "65+"]).astype(str)
D["child_g"] = D.n_child.clip(upper=3)
D["hhsize_g"] = D.hh_size.clip(upper=6)
D.to_csv(PROCESSED / "gyka_hane_2014_2025.csv", index=False, encoding="utf-8")
pd.DataFrame(flow).to_csv(TABLES / "A1_gyka_orneklem_akisi.csv", index=False, encoding="utf-8")
write_manifest("gyka")
print(pd.DataFrame(flow).to_string())
print("dwelling x year\n", pd.crosstab(D.year, D.dwelling))
