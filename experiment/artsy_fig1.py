"""Artsy Figure 1: colour variety vs mate choice.
- coloured by mating BEHAVIOUR type (broadcast spawn ... livebearer)
- as many species names labelled as possible, de-collided with leader lines
- editorial / naturalist-plate styling (serif, warm paper)
- 6 hero fish cut-outs in the margins with lines back to their dots
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy import stats
from adjustText import adjust_text
from fish_data import SPECIES, MODE_ORDER

ROOT = os.path.dirname(os.path.abspath(__file__))
CUT  = os.path.join(ROOT, "data", "cutouts")

# ---- palette: ordered random -> choice, cool -> warm (naturalist watercolours)
PAL = {
    "Broadcast / group spawn": "#6b8ea3",
    "Pelagic pair spawn":      "#4f9d8e",
    "Nest builder / guarder":  "#d7a13b",
    "Harem / sex-changer":     "#cf8050",
    "Mouthbrooder":            "#94a04f",
    "Male brooder":            "#9481b0",
    "Livebearer (internal)":   "#cf6b6b",
}
PAPER="#f4ecdb"; INK="#3a342b"; SOFT="#6f665600"; GRID="#ddd0b7"; ACCENT="#9b3d2e"

plt.rcParams.update({
    "font.family":"serif", "font.serif":["Palatino","Palatino Linotype","Georgia","DejaVu Serif"],
    "text.color":INK, "axes.labelcolor":INK, "xtick.color":INK, "ytick.color":INK,
    "figure.facecolor":PAPER, "axes.facecolor":PAPER, "axes.edgecolor":"#b7a988", "axes.linewidth":1.1,
})

df = pd.read_csv(os.path.join(ROOT,"fish_metrics.csv"))
modes = {s["scientific"]: s["mating_mode"] for s in SPECIES}
df["mating_mode"] = df["scientific"].map(modes)
q = df[df.n_images >= 15].copy().reset_index(drop=True)
x = q["mate_choice_index"].values; y = q["hue_entropy"].values

HEROES = ["Katsuwonus pelamis","Paracanthurus hepatus","Pomacanthus imperator",
          "Synchiropus splendidus","Betta splendens","Aulonocara stuartgranti"]

fig, ax = plt.subplots(figsize=(19, 13))
ax.set_xlim(-3.4, 13.4); ax.set_ylim(0.9, 4.8)
rng = np.random.default_rng(2)
jx = x + rng.uniform(-0.13, 0.13, len(x))
q["_jx"] = jx

# regression + soft CI
b1,b0 = np.polyfit(x, y, 1); xl = np.linspace(0,10,50)
boot = np.array([np.polyval(np.polyfit(x[i], y[i],1), xl) for i in
                 (rng.integers(0,len(x),len(x)) for _ in range(500))])
lo,hi = np.percentile(boot,[2.5,97.5],axis=0)
ax.fill_between(xl, lo, hi, color=ACCENT, alpha=0.09, zorder=1)
ax.plot(xl, b0+b1*xl, color=ACCENT, lw=2.2, alpha=0.85, zorder=2)

# points
for mode in MODE_ORDER:
    m = (q["mating_mode"]==mode).values
    ax.scatter(jx[m], y[m], s=115, color=PAL[mode], edgecolor="#fbf6ea",
               linewidth=1.1, alpha=0.92, zorder=4, label=mode)

# ---- labels for as many species as possible (skip heroes; image labels them)
texts = []
for _,r in q.iterrows():
    if r["scientific"] in HEROES: continue
    texts.append(ax.text(r["_jx"], r["hue_entropy"], r["common"], fontsize=6.6,
                         style="italic", color="#57503f", ha="center", va="center", zorder=6))
adjust_text(texts, x=list(jx), y=list(y), ax=ax,
            expand_points=(1.8,2.0), expand_text=(1.3,1.5),
            force_text=(0.6,0.9), force_points=(0.4,0.7),
            arrowprops=dict(arrowstyle="-", color="#b6a988", lw=0.4, alpha=0.8), lim=1000)

# ---- 6 hero cut-outs in the margins with leader lines
# (margin x, margin y in data coords) — low-x on the left, high-x on the right,
# central-x in the TOP margin above its own dot, so leader lines stay short.
HERO_POS = {
 "Katsuwonus pelamis":     (-2.35, 3.35),
 "Paracanthurus hepatus":  (-2.35, 1.65),
 "Pomacanthus imperator":  ( 5.85, 4.55),
 "Synchiropus splendidus": (12.25, 3.6),
 "Betta splendens":        (12.35, 2.5),
 "Aulonocara stuartgranti":(12.35, 1.45),
}
LABEL_DY = {"Pomacanthus imperator": 0.30}     # label above (top-margin fish)
for _,r in q.iterrows():
    sci = r["scientific"]
    if sci not in HERO_POS: continue
    png = os.path.join(CUT, sci.replace(" ","_")+".png")
    if not os.path.exists(png): continue
    img = plt.imread(png)
    zoom = 150.0 / max(img.shape[:2])
    mxy = HERO_POS[sci]
    ax.annotate("", xy=(r["_jx"], r["hue_entropy"]), xytext=mxy,
                arrowprops=dict(arrowstyle="-", color="#8a7d61", lw=1.0, alpha=0.9,
                                shrinkA=26, shrinkB=6,
                                connectionstyle="arc3,rad=0.10"), zorder=5)
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom), mxy, frameon=False, zorder=7,
                        box_alignment=(0.5,0.5))
    ax.add_artist(ab)
    dy = LABEL_DY.get(sci, -0.30)
    ax.text(mxy[0], mxy[1]+dy, r["common"], fontsize=9, style="italic",
            ha="center", color=INK, zorder=8)

# axis dressing
ax.set_xticks(range(0,11,2))
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines["left"].set_bounds(1.2,4.0); ax.spines["bottom"].set_bounds(0,10)
ax.grid(axis="both", color=GRID, lw=0.6, alpha=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("mate-choice index      random / broadcast spawning   to   strict female choice",
              fontsize=13, labelpad=10)
ax.set_ylabel("colour variety   —   distinct hues worn by the fish", fontsize=13, labelpad=10)

rho,p = stats.spearmanr(x,y)
fig.text(0.09, 0.95, "Fish that choose their mates paint with a wider palette",
         fontsize=23, style="italic", color=INK)
fig.text(0.09, 0.915, f"colour variety rises with mate choice across {len(q)} tropical fish   ·   "
         f"Spearman ρ = {rho:+.2f},  p < 0.001,  phylogeny-corrected p = 0.011",
         fontsize=11.5, color="#6a6252")

leg = ax.legend(title="mating behaviour", frameon=False, loc="lower center",
                ncol=4, fontsize=10.5, title_fontsize=11.5, bbox_to_anchor=(0.5,-0.175))
leg.get_title().set_style("italic")
fig.subplots_adjust(left=0.06, right=0.97, top=0.87, bottom=0.15)
fig.savefig(os.path.join(ROOT,"fig1_artsy_colour_variety.png"), dpi=150, facecolor=PAPER)
print("wrote fig1_artsy_colour_variety.png")
