"""Package validation: single model registry, required outputs, source manifest, output hashes."""
import hashlib
import json
import pandas as pd
from common import TABLES, FIGURES, VALID, METADATA, PROCESSED

regs = sorted(TABLES.glob("registry_*.csv"))
reg = pd.concat([pd.read_csv(p) for p in regs], ignore_index=True)
reg.to_csv(TABLES / "model_registry.csv", index=False, encoding="utf-8")
required_tables = ["A1_gyka_orneklem_akisi", "A2_hhba_orneklem_akisi", "A3_hh050_olcum_tanisi", "T1_tanimlayici_kiracilar",
                   "T1b_yillik_kira_serisi", "T2_olay_calismasi", "T2b_olay_calismasi_baz2021", "T3_donemlik_did",
                   "T3b_kira_farki_duzey_degisim", "T4_doz_heterojenlik", "T4b_gelir_dilimleri_fark_testi",
                   "T5_hareketlilik", "T5b_yeni_kiraci_payi", "T6_hhba_olay_calismasi",
                   "T6b_hhba_donemlik_did", "T6c_hhba_tanimlayici", "T7_saglamlik"]
required_figs = ["sekil1_medyan_kira", "sekil2_olay_calismasi_kira", "sekil3_olay_calismasi_yuk_gecikme",
                 "sekil4_kiracilik_suresi_doz", "sekil5_yeni_kiraci_payi", "sekil6_hhba_olay_calismasi"]
checks = {"tables_present": all((TABLES / f"{t}.csv").exists() for t in required_tables),
          "figures_present": all((FIGURES / f"{f}.png").exists() and (FIGURES / f"{f}.pdf").exists() for f in required_figs),
          "registry_rows": int(len(reg)),
          "registry_no_missing_coef": bool(reg.coef.notna().all()),
          "pooled_wcb_present": bool(reg[reg.model_id.str.startswith("POOL_") & (reg.term != "sit")].p_wcb.notna().all())}
tested = reg[reg.model_id.str.startswith(("POOL_", "INC_JOINT_")) & (reg.term != "sit")]
checks["wcb_9999_and_positive"] = bool((tested.wcb_B == 9999).all() and tested.p_wcb.between(.0001, 1).all())
checks["pooled_contrasts_complete"] = int(reg[reg.model_id.str.startswith("POOL_") & reg.term.isin(["cap_minus_2022", "2025_minus_cap"])].shape[0]) == 12
checks["income_joint_tests_complete"] = int(reg.model_id.str.startswith("INC_JOINT_").sum()) == 2
src = {}
for p in METADATA.glob("sources_*.json"):
    src[p.stem] = json.loads(p.read_text(encoding="utf-8"))
(METADATA / "sources.json").write_text(json.dumps(src, ensure_ascii=False, indent=2), encoding="utf-8")
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
baseline = {"tables": {p.name: h(p) for p in sorted(TABLES.glob("*.csv"))},
            "figures": {p.name: h(p) for p in sorted(FIGURES.glob("*.pdf"))},
            "processed": {p.name: h(p) for p in sorted(PROCESSED.glob("*.csv"))}}
(METADATA / "package_baseline.json").write_text(json.dumps(baseline, indent=1), encoding="utf-8")
(VALID / "package_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
print(json.dumps(checks, indent=2))
assert checks["tables_present"] and checks["figures_present"] and checks["registry_no_missing_coef"] and checks["pooled_wcb_present"]
assert checks["wcb_9999_and_positive"] and checks["pooled_contrasts_complete"] and checks["income_joint_tests_complete"]
