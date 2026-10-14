"""Household Budget Survey (HHBA) 2016-2019, 2022, 2024: tenant households with consumption shares.

Consumption: monthly TL by COICOP division from the item file. Main total excludes imputed rent
(COICOP 04.2: items 4210, 4220). Robustness total: purchases only (ELDEEDIS_SEKIL == 1).
"""
import numpy as np
import pandas as pd
from common import resolve, read_tuik, num, write_manifest, PROCESSED, TABLES

YEARS = [2016, 2017, 2018, 2019, 2022, 2024]
IMPUTED = [4210, 4220]  # COICOP 04.2 imputed rent (owner-occupiers, rent-free) in the 2016+ item lists
DIV = {1: "food", 2: "alc_tob", 3: "clothing", 4: "housing", 5: "furnish", 6: "health", 7: "transport",
       8: "communic", 9: "recreation", 10: "education", 11: "restaurants", 12: "misc"}
rows, flow = [], []
for y in YEARS:
    h = read_tuik(resolve(y, "HHBA", "hane"))
    f = read_tuik(resolve(y, "HHBA", "fert"))
    t = read_tuik(resolve(y, "HHBA", "tuketim"))
    kcol = [c for c in t.columns if c.startswith("HBS_KOD")][0]
    vcol = [c for c in t.columns if c.startswith("DEGER")][0]
    t["k"] = num(t[kcol]).astype("Int64")
    t["v"] = num(t[vcol]).fillna(0.0)
    t["acq"] = num(t["ELDEEDIS_SEKIL"])
    t["div"] = (t["k"] // 1000).astype(int)          # 1111 -> 1 ; 10500 -> 10
    t["imp"] = t["k"].isin(IMPUTED)
    base = t[~t.imp]
    tot = base.groupby("BIRIMNO").v.sum().rename("cons_total")
    divs = base.groupby(["BIRIMNO", "div"]).v.sum().unstack(fill_value=0.0)
    divs.columns = ["c_" + DIV.get(int(c), f"d{c}") for c in divs.columns]
    purch = t[(t.acq == 1) & ~t.imp]
    ptot = purch.groupby("BIRIMNO").v.sum().rename("purch_total")
    pdiv = purch.groupby(["BIRIMNO", "div"]).v.sum().unstack(fill_value=0.0)
    pdiv.columns = ["p_" + DIV.get(int(c), f"d{c}") for c in pdiv.columns]
    rent_item = t[t.k == 4110].groupby("BIRIMNO").v.sum().rename("rent_item")
    age = num(f["YAS"])
    comp = pd.DataFrame({"n_child": (age < 15).groupby(f.BIRIMNO).sum(),
                         "n_65p": (age >= 65).groupby(f.BIRIMNO).sum()})
    keep = ["BIRIMNO", "MULKIYET", "KIRA_TUTAR", "KONUT_SURE_YIL", "ODA_SAYI", "KONUT_ALAN", "ISITMA_SISTEM",
            "KONUT_TIP", "HH_TIP_HBA", "HHB", "EFB_OECD", "HARCAMA", "FAKTOR"]
    d = h[[c for c in keep if c in h]].copy()
    for c in d.columns:
        d[c] = num(d[c])
    d = (d.set_index("BIRIMNO").join(tot).join(divs).join(ptot).join(pdiv).join(rent_item).join(comp).reset_index())
    d.insert(0, "year", y)
    ten = d[d.MULKIYET == 2]
    flow.append({"year": y, "households": len(d), "tenants": len(ten),
                 "tenants_rent_pos": int((ten.KIRA_TUTAR > 0).sum()),
                 "tenants_valid": int(((ten.KIRA_TUTAR > 0) & ten.KONUT_SURE_YIL.notna() & (ten.cons_total > 0)).sum()),
                 "rent_item_matches_rent_share": float(np.mean(np.isclose(ten.rent_item.fillna(0), ten.KIRA_TUTAR, rtol=0.02)))})
    rows.append(d)

D = pd.concat(rows, ignore_index=True)
D = D[(D.MULKIYET == 2) & (D.KIRA_TUTAR > 0) & D.KONUT_SURE_YIL.notna() & (D.cons_total > 0)].copy()
for c in [c for c in D.columns if c.startswith("c_")]:
    D[c] = D[c].fillna(0.0)
for c in [c for c in D.columns if c.startswith("p_")]:
    D[c] = D[c].fillna(0.0)
D["tenure_years"] = D.KONUT_SURE_YIL
D["rooms"] = D.ODA_SAYI.clip(upper=6)
D["log_area"] = np.log(D.KONUT_ALAN.clip(20, 400))
D["heating"] = D.ISITMA_SISTEM.fillna(9).astype(int)
D["hhsize_g"] = D.HHB.clip(upper=6)
D["child_g"] = D.n_child.clip(upper=3)
D.to_csv(PROCESSED / "hhba_kiraci_2016_2024.csv", index=False, encoding="utf-8")
pd.DataFrame(flow).to_csv(TABLES / "A2_hhba_orneklem_akisi.csv", index=False, encoding="utf-8")
write_manifest("hhba")
print(pd.DataFrame(flow).to_string())
print(D.groupby("year")[["KIRA_TUTAR", "cons_total", "c_housing", "c_food"]].median())
