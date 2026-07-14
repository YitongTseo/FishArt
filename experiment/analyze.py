"""Stage 3: build every candidate Y-axis, correlate each with the behavioural
mate-choice index, and plot. Includes species-level 'distinctiveness'
(how far a species stands out from all others in feature space)."""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist

ROOT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(ROOT, "fish_metrics.csv"))

COLOUR  = ["colorfulness","hue_entropy","saturation","sat_p90","high_chroma_frac","n_color_clusters"]
PATTERN = ["contrast","edge_density"]
SHAPE   = ["aspect_ratio","body_depth","solidity","silhouette_cplx","extent"]

def z(s): return (s - s.mean()) / s.std(ddof=0)
Z = df.copy()
for c in COLOUR+PATTERN+SHAPE:
    Z[c] = z(df[c].fillna(df[c].median()))

def knn_distinctiveness(cols, k=5):
    X = Z[cols].values
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, :k].mean(axis=1)

# ---- composite Y-axes ----
df["strike_colour"] = Z[COLOUR].mean(axis=1)
df["shape_elaboration"] = Z[["silhouette_cplx","body_depth"]].mean(axis=1) - Z["solidity"]
df["distinct_colour"] = knn_distinctiveness(COLOUR)
df["distinct_shape"]  = knn_distinctiveness(SHAPE)
df["distinct_overall"] = knn_distinctiveness(COLOUR+SHAPE+PATTERN)

x = df["mate_choice_index"].values
CANDIDATES = (COLOUR + PATTERN + SHAPE +
              ["strike_colour","shape_elaboration","distinct_colour","distinct_shape","distinct_overall"])

print(f"n = {len(df)} species\n{'metric':20s} {'rho':>7s} {'p':>8s}")
res = []
for c in CANDIDATES:
    rho, p = stats.spearmanr(x, df[c])
    res.append((c, rho, p))
    print(f"{c:20s} {rho:+7.3f} {p:8.3f}")
res.sort(key=lambda t: -t[1])

# ---- figure 1: correlation summary (all Y-axes at once) ----
fig, ax = plt.subplots(figsize=(9, 8))
names = [r[0] for r in res]; rhos = [r[1] for r in res]; ps = [r[2] for r in res]
colors = ["#2a9d8f" if rr>0 else "#e76f51" for rr in rhos]
y = np.arange(len(names))
ax.barh(y, rhos, color=colors, edgecolor="black", lw=0.5)
for yi, (rr, pp) in enumerate(zip(rhos, ps)):
    star = "*" if pp<0.05 else ("." if pp<0.10 else "")
    ax.text(rr + (0.01 if rr>=0 else -0.01), yi, f"{rr:+.2f}{star}",
            va="center", ha="left" if rr>=0 else "right", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Spearman correlation with mate-choice index  (* p<0.05, . p<0.10)")
ax.set_title(f"Which measure of striking-ness tracks mate choice?  (n={len(df)} species)", weight="bold")
ax.invert_yaxis(); ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(ROOT,"fig_correlation_summary.png"), dpi=150)

# ---- phylogenetic robustness: correlate FAMILY MEANS (controls pseudoreplication) ----
fam_mean = df.groupby("family").agg(x=("mate_choice_index","mean"),
            hue_entropy=("hue_entropy","mean"), colorfulness=("colorfulness","mean"),
            aspect_ratio=("aspect_ratio","mean"), body_depth=("body_depth","mean")).reset_index()
print(f"\n--- family-mean control (n={len(fam_mean)} families) ---")
for c in ["hue_entropy","colorfulness","aspect_ratio","body_depth"]:
    rho,p = stats.spearmanr(fam_mean["x"], fam_mean[c]); print(f"  {c:14s} rho={rho:+.3f} p={p:.3f}")

# ---- figure 2: scatter grid of the 6 most informative Y-axes ----
picks = ["hue_entropy","colorfulness","aspect_ratio","body_depth","contrast","distinct_overall"]
fams = sorted(df["family"].unique())
cmap = plt.get_cmap("tab20b"); fam_color = {f: cmap(i%20) for i,f in enumerate(fams)}
rng = np.random.default_rng(3)
fig2, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, c in zip(axes.flat, picks):
    yv = df[c].values; jx = x + rng.uniform(-0.15,0.15,len(x))
    ax.scatter(jx, yv, s=45, c=[fam_color[f] for f in df["family"]], edgecolor="black", lw=0.4, alpha=0.9)
    b,a = np.polyfit(x, yv, 1); xs = np.linspace(x.min(),x.max(),40)
    ax.plot(xs, a+b*xs, "--", color="crimson", lw=2)
    rho,p = stats.spearmanr(x, yv)
    ax.text(0.03,0.96,f"$\\rho$={rho:+.2f}  p={p:.3g}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round",fc="white",ec="gray",alpha=0.9), fontsize=9.5)
    ax.set_title(c, weight="bold", fontsize=11); ax.grid(alpha=0.25)
    ax.set_xlabel("mate-choice index")
fig2.suptitle("Candidate striking-ness axes vs mate choice", weight="bold", fontsize=14)
fig2.tight_layout(rect=(0,0,1,0.97)); fig2.savefig(os.path.join(ROOT,"fig_scatter_grid.png"), dpi=150)

df.sort_values("mate_choice_index").to_csv(os.path.join(ROOT,"fish_analysed.csv"), index=False)
print("\nwrote fig_correlation_summary.png, fig_scatter_grid.png, fish_analysed.csv")
