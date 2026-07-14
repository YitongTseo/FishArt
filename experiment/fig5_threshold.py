"""Figure 5: the data-quality rule, on trial.

The >= 15-photo rule was defended on the grounds that filtering *strengthened* the result.
That defence is uncomfortably close to the description of p-hacking, so this figure does the
only honest thing available: it plots the effect at EVERY photo-count bar, for both the old
and the new masks, and lets the reader see the whole curve rather than the chosen point.

What you want to see (and what is there) is monotone attenuation toward zero as the bar drops
-- the signature of noise diluting a real effect -- rather than a cliff at which the effect
appears from nowhere, which is what a filtering artifact would look like.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER, INK, ACCENT, GRID = "#f4ecdb", "#3a342b", "#9b3d2e", "#ddd0b7"
OLDC = "#8a8172"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Georgia", "DejaVu Serif"],
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "axes.edgecolor": "#b7a988", "axes.linewidth": 1.1})

old = pd.read_csv(os.path.join(ROOT, "fish_metrics.csv"))
new = pd.read_csv(os.path.join(ROOT, "fish_metrics_isnet.csv"))
BARS = list(range(1, 26))


def curve(df):
    rs, ps, ns = [], [], []
    for b in BARS:
        d = df[df.n_images >= b].dropna(subset=["hue_entropy", "mate_choice_index"])
        r, p = stats.spearmanr(d["mate_choice_index"], d["hue_entropy"])
        rs.append(r); ps.append(p); ns.append(len(d))
    return np.array(rs), np.array(ps), np.array(ns)


ro, po, no = curve(old)
rn, pn, nn = curve(new)

fig, (ax, axn) = plt.subplots(2, 1, figsize=(11, 8.4), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})

ax.axhline(0, color=INK, lw=0.9, alpha=0.35)
ax.plot(BARS, ro, "-", color=OLDC, lw=2.0, alpha=0.9, label="old masks (u²-net, no gate, no veto)")
ax.plot(BARS, rn, "-", color=ACCENT, lw=3.0, label="new masks (isnet + shape gates + classifier veto)")

# significance: fill the markers only where p < 0.05
for r, p, c in ((ro, po, OLDC), (rn, pn, ACCENT)):
    sig = p < 0.05
    ax.plot(np.array(BARS)[sig], r[sig], "o", color=c, ms=5, zorder=3)
    ax.plot(np.array(BARS)[~sig], r[~sig], "o", mfc=PAPER, mec=c, mew=1.4, ms=5, zorder=3)

ax.axvline(15, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.4)
ax.text(15.25, 0.055, "the rule the figures used to apply", fontsize=11.5,
        style="italic", color="#6b6355", rotation=0, va="bottom")
ax.annotate("floodgates open:\nevery species in", xy=(1, rn[0]), xytext=(2.4, 0.10),
            fontsize=12, color="#6b6355", style="italic",
            arrowprops=dict(arrowstyle="-", color="#b7a988", lw=1))

ax.set_ylim(0, 0.47)
ax.set_ylabel("colour variety vs mate choice   (Spearman ρ)", fontsize=13.5, labelpad=10)
ax.legend(frameon=False, fontsize=12, loc="upper left", bbox_to_anchor=(0.005, 0.99))
ax.grid(color=GRID, lw=0.6, alpha=0.4)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("The effect at every data-quality bar — it attenuates, it never disappears",
             fontsize=17, pad=16, loc="left")
ax.text(0, 1.012, "filled dot = p < 0.05   ·   hollow = not significant",
        transform=ax.transAxes, fontsize=11.5, color="#8a8172", style="italic")

axn.plot(BARS, no, "-", color=OLDC, lw=1.8, alpha=0.9)
axn.plot(BARS, nn, "-", color=ACCENT, lw=2.4)
axn.axvline(15, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.4)
axn.set_ylabel("species kept", fontsize=12.5, labelpad=10)
axn.set_xlabel("data-quality bar — minimum clean photos per species", fontsize=13.5, labelpad=9)
axn.grid(color=GRID, lw=0.6, alpha=0.4)
axn.set_axisbelow(True)
axn.spines[["top", "right"]].set_visible(False)
axn.set_xlim(0.5, 25.5)

fig.subplots_adjust(left=0.085, right=0.975, top=0.88, bottom=0.095)
out = os.path.join(ROOT, "fig5_threshold.png")
fig.savefig(out, dpi=150, facecolor=PAPER)
print(f"wrote fig5_threshold.png")
print(f"  new masks: rho {rn[0]:+.3f} (all {nn[0]} species)  ->  {rn[14]:+.3f} (>=15: {nn[14]})  "
      f"->  {rn[24]:+.3f} (>=25: {nn[24]})")
print(f"  significant (p<0.05) at every bar from 1 to 25: "
      f"{'yes' if (pn < 0.05).all() else 'NO'}")
