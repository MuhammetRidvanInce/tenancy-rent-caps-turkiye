"""Formats manuscript tables directly from the analysis outputs (prevents transcription errors)."""
import numpy as np
import pandas as pd
from common import TABLES

def f(x, d=3):
    if pd.isna(x):
        return "–"
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")

def stars(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))

def pstr(p):
    return "<0,001" if p < 0.001 else f(p, 3)

def pwstr(p):
    return f(p, 4)

out = []
# ---- Table 1 descriptives
T1 = pd.read_csv(TABLES / "T1_tanimlayici_kiracilar.csv")
T1["col"] = T1.period.str.replace("-", "–") + " " + T1.group.map({"new": "Yeni", "sitting": "Mevcut"})
rows = [("Gözlem sayısı (hane)", "N", 0), ("Medyan aylık kira (TL)", "median_rent_TL", 0),
        ("Medyan kira/gelir oranı (%)", "median_rent_to_income_pct", 1), ("Kira/gelir > %40 (%)", "rent_burden_gt40_pct", 1),
        ("Konut masrafı çok yük (%)", "heavy_burden_pct", 1), ("Kira/borç ödemesinde gecikme (%)", "rent_arrears_pct", 1),
        ("Fatura ödemesinde gecikme (%)", "utility_arrears_pct", 1), ("Geçinmekte zorlanma (%)", "hard_to_make_ends_meet_pct", 1),
        ("Medyan eşdeğer gelir (TL, önceki yıl)", "median_eq_income_TL", 0), ("Oda sayısı", "rooms", 2), ("Konut alanı (m²)", "area_m2", 1),
        ("Hane büyüklüğü", "hh_size", 2), ("Hanehalkı sorumlusunun yaşı", "head_age", 1), ("Üç büyükşehir bölgesinde (%)", "metro_pct", 1),
        ("Konutta oturma süresi (yıl)", "tenure_years_mean", 2)]
cols = list(T1.col)
out.append("TABLO 1\n| Gösterge | " + " | ".join(cols) + " |\n|---|" + "---:|" * len(cols))
for lab, k, d in rows:
    out.append(f"| {lab} | " + " | ".join(f(v, d) for v in T1[k]) + " |")

# ---- Table 2 pooled DiD
P = pd.read_csv(TABLES / "T3_donemlik_did.csv")
labs = {"lrent": "Log aylık kira", "rent_burden40": "Kira/gelir > %40", "heavy": "Konut masrafı çok yük",
        "arr_rent": "Kira/borç gecikmesi", "arr_util": "Fatura gecikmesi", "hard": "Geçinmekte zorlanma"}
terms = [("sit_P21", "Mevcut × 2021"), ("sit_P22", "Mevcut × 2022"), ("sit_P2324", "Mevcut × 2023–2024"), ("sit_P25", "Mevcut × 2025")]
out.append("\nTABLO 2\n| | " + " | ".join(labs.values()) + " |\n|---|" + "---:|" * len(labs))
for t, tl in terms:
    c, s, w = [], [], []
    for o in labs:
        r = P[(P.outcome == o) & (P.term == t)].iloc[0]
        c.append(f(r.coef) + stars(r.p_wcb)); s.append(f"({f(r.se)})"); w.append(f"[{pwstr(r.p_wcb)}]")
    out.append(f"| {tl} | " + " | ".join(c) + " |"); out.append("| | " + " | ".join(s) + " |"); out.append("| | " + " | ".join(w) + " |")
for t, tl in [("cap_minus_2022", "2023–2024 − 2022"), ("2025_minus_cap", "2025 − 2023–2024")]:
    c, w = [], []
    for o in labs:
        r = P[(P.outcome == o) & (P.term == t)].iloc[0]
        c.append(f"{f(r.coef)}{stars(r.p_wcb)} ({f(r.se)})")
        w.append(f"[{pwstr(r.p_wcb)}]")
    out.append(f"| {tl} | " + " | ".join(c) + " |")
    out.append("| | " + " | ".join(w) + " |")
c = []
for o in labs:
    r = P[(P.outcome == o) & (P.term == "pretrend_F_2016_2020_base2019")].iloc[0]
    c.append(f"{f(r.coef, 2)} (p = {f(r.p, 3)})")
out.append("| Ön-trend F(4, 25) | " + " | ".join(c) + " |")
from common import PROCESSED
S = pd.read_csv(PROCESSED / "gyka_kiraci_analiz_2016_2025.csv", low_memory=False)
ref = S[(S.sit == 1) & S.year.between(2016, 2020)]
c = [f(np.average(ref[o], weights=ref.HB040), 3) for o in labs]
out.append("| Mevcut kiracı ortalaması, 2016–2020 | " + " | ".join(c) + " |")
out.append("| Gözlem sayısı | " + " | ".join(f(P[(P.outcome == o) & (P.term == 'sit')].n.iloc[0], 0) for o in labs) + " |")

# ---- Table 3 dose and income
D = pd.read_csv(TABLES / "T4_doz_heterojenlik.csv")
out.append("\nTABLO 3 (A: kiracılık süresi; B: gelir dilimi; C: büyükşehir)")
out.append("| | Log kira 2023–24 | Log kira 2025 | Kira/gelir > %40 2023–24 | Çok yük 2023–24 | Kira/borç gecikmesi 2025 |\n|---|---:|---:|---:|---:|---:|")
for b, bl in [("b2_4", "2–4 yıl"), ("b5_9", "5–9 yıl"), ("b10p", "10+ yıl")]:
    cells = []
    for o, p in [("lrent", "P2324"), ("lrent", "P25"), ("rent_burden40", "P2324"), ("heavy", "P2324"), ("arr_rent", "P25")]:
        r = D[(D.analysis == "tenure_bins") & (D.outcome == o) & (D.term == f"{b}_{p}")].iloc[0]
        cells.append(f"{f(r.coef)}{stars(r.p)} ({f(r.se)})")
    out.append(f"| {bl} | " + " | ".join(cells) + " |")
cells = []
for o, p in [("lrent", "P2324"), ("lrent", "P25"), ("rent_burden40", "P2324"), ("heavy", "P2324"), ("arr_rent", "P25")]:
    r = D[(D.analysis == "tenure_bins") & (D.outcome == o) & (D.term == f"diff_2_4_minus_5_9_{p}")].iloc[0]
    cells.append(f"{f(r.coef)} (p = {f(r.p, 3)})")
out.append("| Fark: 2–4 − 5–9 yıl | " + " | ".join(cells) + " |")
out.append("\n| Eşdeğer gelir dilimi | Log kira 2023–24 | Log kira 2025 | Çok yük 2023–24 | Kira/borç gecikmesi 2025 | N |\n|---|---:|---:|---:|---:|---:|")
for q in range(1, 6):
    cells = []
    for o, p in [("lrent", "sit_P2324"), ("lrent", "sit_P25"), ("heavy", "sit_P2324"), ("arr_rent", "sit_P25")]:
        r = D[(D.analysis == f"income_Q{q}") & (D.outcome == o) & (D.term == p)].iloc[0]
        cells.append(f"{f(r.coef)}{stars(r.p)} ({f(r.se)})")
    n = int(D[(D.analysis == f"income_Q{q}")].n.iloc[0])
    out.append(f"| Q{q} | " + " | ".join(cells) + f" | {f(n, 0)} |")
J = pd.read_csv(TABLES / "T4b_gelir_dilimleri_fark_testi.csv")
a = J[J.term == "Q5_minus_Q1_P2324"].iloc[0]
b = J[J.term == "Q5_minus_Q1_P25"].iloc[0]
out.append(f"| Q5 − Q1 (ortak model) | {f(a.coef)} ({f(a.se)}) | {f(b.coef)} ({f(b.se)}) | – | – | {f(a.n, 0)} |")
out.append(f"| Q5 − Q1: t(25) p | {pwstr(a.p)} | {pwstr(b.p)} | – | – | – |")
out.append(f"| Q5 − Q1: bootstrap p | {pwstr(a.p_wcb)} | {pwstr(b.p_wcb)} | – | – | – |")
out.append("\n| Büyükşehir etkileşimi | Log kira | Çok yük | Kira/borç gecikmesi |\n|---|---:|---:|---:|")
for t, tl in [("sit_P2324", "Mevcut × tavan (diğer bölgeler)"), ("sitm_P2324", "Ek fark: büyükşehir × tavan"), ("sit_P25", "Mevcut × 2025 (diğer bölgeler)"), ("sitm_P25", "Ek fark: büyükşehir × 2025")]:
    cells = []
    for o in ["lrent", "heavy", "arr_rent"]:
        r = D[(D.analysis == "metro_interaction") & (D.outcome == o) & (D.term == t)].iloc[0]
        cells.append(f"{f(r.coef)}{stars(r.p)} ({f(r.se)})")
    out.append(f"| {tl} | " + " | ".join(cells) + " |")

# ---- Table 4 mobility
M = pd.read_csv(TABLES / "T5_hareketlilik.csv"); S5 = pd.read_csv(TABLES / "T5b_yeni_kiraci_payi.csv")
out.append("\nTABLO 4\n| Terim | Kontrolsüz | Kontrollü |\n|---|---:|---:|")
for t, tl in [("trend", "Doğrusal eğilim (yıllık)"), ("d2021", "Sapma 2021"), ("d2022", "Sapma 2022"), ("d2023", "Sapma 2023"), ("d2024", "Sapma 2024"), ("d2025", "Sapma 2025")]:
    a = M[(M.spec == "trend_only") & (M.term == t)].iloc[0]; b = M[(M.spec == "trend_controls") & (M.term == t)].iloc[0]
    out.append(f"| {tl} | {f(a.coef)}{stars(a.p)} ({f(a.se)}) | {f(b.coef)}{stars(b.p)} ({f(b.se)}) |")
out.append("| Gözlem sayısı | 72.527 | 72.526 |")
out.append("Yeni kiracı payı (%): " + ", ".join(f"{int(r.year)}: {f(r.share_new_pct, 1)}" for r in S5.itertuples()))

# ---- Table 5 HBS
H = pd.read_csv(TABLES / "T6b_hhba_donemlik_did.csv")
hl = {"lrent": "Log kira", "sh_rent": "Kira payı", "sh_food": "Gıda payı", "sh_health": "Sağlık payı", "sh_education": "Eğitim payı",
      "sh_transport": "Ulaştırma payı", "sh_restaurants": "Lokanta-otel payı", "lcons": "Log toplam tüketim", "lcons_nonrent": "Log kira dışı tüketim",
      "lfood": "Log gıda harcaması", "psh_rent": "Kira payı (satın alış)", "psh_food": "Gıda payı (satın alış)"}
out.append("\nTABLO 5\n| Bağımlı değişken | Mevcut × 2022 | Mevcut × 2024 | Ön-trend F(3) p | Mevcut kiracı ort. (2016–2019) |\n|---|---:|---:|---:|---:|")
for o, l in hl.items():
    a = H[(H.outcome == o) & (H.term == "sit_P22")].iloc[0]; b = H[(H.outcome == o) & (H.term == "sit_P24")].iloc[0]
    pt = H[(H.outcome == o) & (H.term == "pretrend_F_2016_2018_base2019")].iloc[0]
    out.append(f"| {l} | {f(a.coef)}{stars(a.p)} ({f(a.se)}) | {f(b.coef)}{stars(b.p)} ({f(b.se)}) | {f(pt.p, 3)} | {f(a.mean_dep_sitting_pre, 2)} |")

# ---- Table 6 robustness
R = pd.read_csv(TABLES / "T7_saglamlik.csv")
out.append("\nTABLO 6\n| Tanım | Log kira 2023–24 | Log kira 2025 | Çok yük 2023–24 | Çok yük 2025 | Kira/borç gecikmesi 2025 | N |\n|---|---:|---:|---:|---:|---:|---:|")
order = [s for s in R.spec.unique() if not s.startswith("R10")] + [s for s in R.spec.unique() if s.startswith("R10")]
for s in order:
    q = R[R.spec == s]
    if s.startswith("R10"):
        cells = [f"{f(q[q.outcome == o].iloc[0].coef)}{stars(q[q.outcome == o].iloc[0].p)} ({f(q[q.outcome == o].iloc[0].se)})" for o in ["lrent", "heavy", "arr_rent"]]
        out.append(f"| {s} | {cells[0]} | – | {cells[1]} | – | {cells[2]} | {f(q.n.iloc[0], 0)} |")
        continue
    cells = []
    for o, t in [("lrent", "sit_P2324"), ("lrent", "sit_P25"), ("heavy", "sit_P2324"), ("heavy", "sit_P25"), ("arr_rent", "sit_P25")]:
        r = q[(q.outcome == o) & (q.term == t)].iloc[0]
        cells.append(f"{f(r.coef)}{stars(r.p)} ({f(r.se)})")
    out.append(f"| {s} | " + " | ".join(cells) + f" | {f(q.n.iloc[0], 0)} |")

# ---- Appendix tables
A1 = pd.read_csv(TABLES / "A1_gyka_orneklem_akisi.csv")
out.append("\nTABLO A1\n| Yıl | Hane | Kiracı | Analiz örnekleminde kiracı | Ev sahibi | Ev sahibi, HH050 boş |\n|---|---:|---:|---:|---:|---:|")
for r in A1.itertuples():
    out.append(f"| {r.year} | {f(r.households, 0)} | {f(r.tenants, 0)} | {f(r.tenants_income_pos, 0)} | {f(r.owners, 0)} | {f(r.owners_HH050_missing, 0)} |")
A3 = pd.read_csv(TABLES / "A3_hh050_olcum_tanisi.csv")
out.append("\nTABLO A2\n| Yıl | Ev sahibi, HH050 boş (%) | Kiracı, HH050 = kira (%) | Kiracı, medyan kira dışı masraf (TL) | Kiracı aşırı yük, toplam masraf (%) | Kiracı aşırı yük, yalnız kira (%) | Ev sahibi aşırı yük (%) |\n|---|---:|---:|---:|---:|---:|---:|")
for r in A3.itertuples():
    out.append(f"| {r.year} | {f(r.owners_HH050_missing_pct, 1)} | {f(r.tenants_total_cost_equals_rent_pct, 1)} | {f(r.tenants_median_nonrent_cost_TL, 0)} | {f(r.tenants_overburden_total_pct, 1)} | {f(r.tenants_overburden_rent_only_pct, 1)} | {f(r.owners_overburden_total_pct, 1)} |")
L = pd.read_csv(TABLES / "T3b_kira_farki_duzey_degisim.csv")
out.append("\nTABLO A3\n| Dönem | Referansa göre değişim (log) | Kira oranındaki değişim (%) | Toplam koşullu fark (log) | Toplam koşullu fark (%) | Toplam fark için %95 GA (%) |\n|---|---:|---:|---:|---:|---:|")
for r in L.itertuples():
    out.append(f"| {r.period.replace('-', '–')} | {f(r.change_log)} | {f(r.ratio_change_pct, 1)} | {f(r.level_log)} | {f(r.level_gap_pct, 1)} | [{f(r.level_ci_low_pct, 1)}; {f(r.level_ci_high_pct, 1)}] |")
(TABLES / "makale_tablolari.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
