"""Honest 'forest plot' of the hypothesis test: the correlation between mate
choice and striking-ness, computed many ways. Shows how support appears in broad
samples and disappears under phylogenetic / sex / subset controls."""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
full = pd.read_csv(os.path.join(ROOT,"fish_metrics.csv"))
sx   = pd.read_csv(os.path.join(ROOT,"sexed_metrics.csv"))

def rho_ci(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x)|np.isnan(y)); x,y = x[m],y[m]
    r,p = stats.spearmanr(x,y); n=len(x)
    z = np.arctanh(r); se = 1/np.sqrt(n-3)
    lo,hi = np.tanh(z-1.96*se), np.tanh(z+1.96*se)
    return r,lo,hi,p,n

x_full = full["mate_choice_index"]
famm = full.groupby("family").agg(x=("mate_choice_index","mean"),
        hue=("hue_entropy","mean"), col=("colorfulness","mean")).reset_index()
dm = sx.dropna(subset=["male_hue"]); dm = dm[dm["n_male"]>=6]
dd = sx.dropna(subset=["dichromatism_dE"])

# (label, measure-series, x-series)  grouped by conceptual measure
rows = [
 ("COLOUR VARIETY (hue entropy)", None, None),
 ("  full sample",                full["hue_entropy"], x_full),
 ("  family means (phylo)",       famm["hue"], famm["x"]),
 ("  sex-annot. subset, mixed",   None, None),  # filled below
 ("  sex-annot. subset, MALE-only", dm["male_hue"], dm["mate_choice_index"]),
 ("COLOUR INTENSITY (colorfulness)", None, None),
 ("  full sample",                full["colorfulness"], x_full),
 ("  sex-annot. subset, mixed",   None, None),  # filled below
 ("  sex-annot. subset, MALE-only", dm["male_colorfulness"], dm["mate_choice_index"]),
 ("SEXUAL DICHROMATISM", None, None),
 ("  male-vs-female colour dist.", dd["dichromatism_dE"], dd["mate_choice_index"]),
]
# fill mixed colorfulness on the sex subset via merge
mix_sub = dm.merge(full[["scientific","colorfulness","hue_entropy"]], on="scientific")
rows[3] = ("  sex-annot. subset, mixed", mix_sub["hue_entropy"], mix_sub["mate_choice_index"])
rows[7] = ("  sex-annot. subset, mixed", mix_sub["colorfulness"], mix_sub["mate_choice_index"])

fig, ax = plt.subplots(figsize=(11, 8))
y = 0; yticks=[]; ylabels=[]
for label, ys, xs in rows[::-1]:
    yticks.append(y); ylabels.append(label)
    if ys is not None and xs is not None and len(ys)>4:
        r,lo,hi,p,n = rho_ci(xs, ys)
        col = "#2a9d8f" if r>0 else "#e76f51"
        sig = "*" if p<0.05 else ("." if p<0.10 else "")
        ax.plot([lo,hi],[y,y],color=col,lw=2.5,solid_capstyle="round")
        ax.plot(r,y,"o",color=col,ms=9,mec="black",mew=0.6)
        ax.text(hi+0.02, y, f"$\\rho$={r:+.2f}{sig}  (n={n})", va="center", fontsize=8.5)
    y += 1
ax.axvline(0, color="black", lw=1)
ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=9.5)
for t,(label,*_ ) in zip(ax.get_yticklabels(), rows[::-1]):
    if not label.startswith("  "): t.set_fontweight("bold")
ax.set_xlim(-0.75, 0.75); ax.set_xlabel("Spearman correlation with mate-choice index  (95% CI; * p<0.05, . p<0.10)")
ax.set_title("Does mate choice predict striking-ness? — the same test, computed many ways",
             weight="bold", fontsize=12)
ax.text(0.5,-0.13,"Green = supports essay (positive)   Orange = against.  "
        "Support appears in broad samples on colour VARIETY, and vanishes / reverses under controls.",
        transform=ax.transAxes, ha="center", fontsize=8.5, style="italic", color="#444")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(ROOT,"fig_summary_forest.png"), dpi=150, bbox_inches="tight")
print("wrote fig_summary_forest.png")
