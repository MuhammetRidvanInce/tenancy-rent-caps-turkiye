"""Figures (PNG 300 dpi + PDF). Palette: validated categorical slots 1-3 (blue, orange, aqua) with
secondary encoding (marker shape, line style) for print and colour-vision deficiency."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import TABLES, FIGURES

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#dcdad4", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False, "lines.linewidth": 2})


def cap_band(ax, x0=2022.5, x1=2024.5, label=True):
    ax.axvspan(x0, x1, color="#efeee9", zorder=0)
    if label:
        ax.text((x0 + x1) / 2, ax.get_ylim()[1], "2023–2024", ha="center", va="top", fontsize=7.5, color=INK2)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)


# Figure 1: raw median rents
s = pd.read_csv(TABLES / "T1b_yillik_kira_serisi.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(s.year, s.median_rent_new, color=ORANGE, marker="o", ms=5, label="Yeni kiracı (yıl farkı 0–1)")
ax.plot(s.year, s.median_rent_sitting, color=BLUE, marker="s", ms=5, ls="--", label="Mevcut kiracı (yıl farkı ≥2)")
ax.set_yscale("log"); ax.set_ylabel("Medyan aylık kira (TL, log ölçek)"); ax.set_xticks(s.year)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", ".")))
cap_band(ax)
ax.annotate("Yeni", (s.year.iloc[-1], s.median_rent_new.iloc[-1]), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
ax.annotate("Mevcut", (s.year.iloc[-1], s.median_rent_sitting.iloc[-1]), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
ax.legend(loc="upper left")
save(fig, "sekil1_medyan_kira")

# Figure 2: event study, log rent
es = pd.read_csv(TABLES / "T2_olay_calismasi.csv")
e = es[es.outcome == "lrent"].sort_values("year")
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.axhline(0, color=INK2, lw=0.8)
ax.errorbar(e.year, e.coef, yerr=[e.coef - e.ci_low, e.ci_high - e.coef], fmt="o", color=BLUE, ms=6, capsize=3, lw=1.5)
ax.plot(e.year, e.coef, color=BLUE, lw=1, alpha=.6)
ax.set_xticks(e.year); ax.set_ylabel("Koşullu log kira farkındaki değişim, %95 GA")
cap_band(ax)
ax.text(2019, ax.get_ylim()[0], "baz yıl", ha="center", va="bottom", fontsize=7.5, color=INK2)
save(fig, "sekil2_olay_calismasi_kira")

# Figure 3: event study, burden and arrears (small multiples, independent axes)
labs = {"heavy": "Konut masrafı çok yük", "arr_rent": "Kira/borç ödemesinde gecikme", "arr_util": "Fatura ödemesinde gecikme"}
fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.2), sharey=True)
for ax, (k, lab), col, mk in zip(axs, labs.items(), [BLUE, ORANGE, AQUA], ["o", "s", "^"]):
    e = es[es.outcome == k].sort_values("year")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.errorbar(e.year, e.coef, yerr=[e.coef - e.ci_low, e.ci_high - e.coef], fmt=mk, color=col, ms=5, capsize=2, lw=1.3)
    ax.set_title(lab, fontsize=9, color=INK); ax.set_xticks([2016, 2018, 2020, 2022, 2024]); cap_band(ax, label=False)
axs[0].set_ylabel("Katsayı (olasılık birimi), %95 GA")
save(fig, "sekil3_olay_calismasi_yuk_gecikme")

# Figure 4: tenure-length dose, log rent
d = pd.read_csv(TABLES / "T4_doz_heterojenlik.csv")
d = d[(d.analysis == "tenure_bins") & (d.outcome == "lrent") & d.term.str.match(r"b\d")].copy()
d["bin"] = d.term.str.extract(r"(b2_4|b5_9|b10p)")[0]; d["per"] = d.term.str.extract(r"_(P\d+)$")[0]
pmap = {"P21": "2021", "P22": "2022", "P2324": "2023–2024", "P25": "2025"}
bmap = {"b2_4": ("2–4 yıl", BLUE, "o"), "b5_9": ("5–9 yıl", ORANGE, "s"), "b10p": ("10+ yıl", AQUA, "^")}
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.axhline(0, color=INK2, lw=0.8)
for i, (b, (lab, col, mk)) in enumerate(bmap.items()):
    q = d[d.bin == b].set_index("per").loc[list(pmap)]
    x = np.arange(4) + (i - 1) * 0.22
    ax.errorbar(x, q.coef, yerr=[q.coef - q.ci_low, q.ci_high - q.coef], fmt=mk, color=col, ms=6, capsize=3, lw=1.5, label=lab)
ax.set_xticks(range(4)); ax.set_xticklabels(pmap.values()); ax.set_ylabel("Koşullu log kira farkındaki değişim, %95 GA")
ax.legend(title="Konutta oturma süresi", loc="lower left", fontsize=8, title_fontsize=8)
save(fig, "sekil4_kiracilik_suresi_doz")

# Figure 5: share of new tenants and 2014-2021 linear trend
sh = pd.read_csv(TABLES / "T5b_yeni_kiraci_payi.csv")
pre = sh[sh.year <= 2020]; b = np.polyfit(pre.year, pre.share_new_pct, 1)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(sh.year, sh.share_new_pct, color=BLUE, marker="o", ms=5, label="Gözlenen pay")
xx = np.arange(2014, 2026); ax.plot(xx, np.polyval(b, xx), color=INK2, ls="--", lw=1.2, label="2014–2020 doğrusal trendi")
ax.set_xticks(sh.year); ax.set_ylabel("Yeni kiracı payı (%)"); cap_band(ax); ax.legend(loc="lower left")
save(fig, "sekil5_yeni_kiraci_payi")

# Figure 6: HBS event study
h = pd.read_csv(TABLES / "T6_hhba_olay_calismasi.csv")
labs = {"lrent": "Log aylık kira", "sh_rent": "Kira payı (yüzde puan)", "sh_food": "Gıda payı (yüzde puan)"}
fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.2))
for ax, (k, lab), col, mk in zip(axs, labs.items(), [BLUE, ORANGE, AQUA], ["o", "s", "^"]):
    e = h[h.outcome == k].sort_values("year")
    x = np.arange(len(e))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.errorbar(x, e.coef, yerr=[e.coef - e.ci_low, e.ci_high - e.coef], fmt=mk, color=col, ms=5, capsize=2, lw=1.3)
    ax.set_xticks(x); ax.set_xticklabels(e.year.astype(int), fontsize=8); ax.set_title(lab, fontsize=9, color=INK)
axs[0].set_ylabel("Katsayı, %95 GA (baz yıl 2019)")
save(fig, "sekil6_hhba_olay_calismasi")
print("figures written")
