"""Figure 4 (the wild one): every datapoint IS the fish. Each species' cut-out
sits at (mate-choice, colour-variety) on a washed background whose colour tells
you which mating behaviour owns that region of the plane. A gallery that
doubles as a scatter plot.

The mating mode is carried by the *background* rather than by a halo behind
each fish: a soft, watercolour-like field built from a per-mode kernel density,
where each mode's opacity is its share of the local density. Where one
behaviour dominates the region reads as its colour; where behaviours mix the
colours mix; where there are no fish it fades back to bare paper.
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from scipy import stats
from scipy.ndimage import gaussian_filter
from fish_data import SPECIES, MODE_ORDER

ROOT = os.path.dirname(os.path.abspath(__file__))
CUT  = os.path.join(ROOT, "data", os.environ.get("CUTDIR", "cutouts"))
OUTPNG = os.environ.get("OUTPNG", "fig4_fish_gallery.png")

PAL = {"Broadcast / group spawn":"#6b8ea3","Pelagic pair spawn":"#4f9d8e",
       "Nest builder / guarder":"#d7a13b","Harem / sex-changer":"#cf8050",
       "Mouthbrooder":"#94a04f","Male brooder":"#9481b0","Livebearer (internal)":"#cf6b6b"}
PAPER="#f4ecdb"; INK="#3a342b"; ACCENT="#9b3d2e"; GRID="#ddd0b7"

# --- how loud the background wash is -----------------------------------------
MAXA   = 1.00   # opacity of a mode where it completely owns the region
GAMMA  = 2.4    # >1 sharpens the hand-over between modes (lower = muddier)
HX, HY = 0.90, 0.28   # kernel bandwidth, in data units (x is 11 wide, y is 3)
LO, HI = 0.06, 0.34   # density window over which paper takes on full pigment
SAT    = 1.75   # push the palette off the paper a little
NORM   = 0.5    # discount a mode's density by count^NORM, so the 4 male-brooders
                # still tint their corner instead of being buried by 36 nest-builders

plt.rcParams.update({
    "font.family":"serif","font.serif":["Palatino","Palatino Linotype","Georgia","DejaVu Serif"],
    "text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
    "figure.facecolor":PAPER,"axes.facecolor":PAPER,"axes.edgecolor":"#b7a988","axes.linewidth":1.1})

df = pd.read_csv(os.path.join(ROOT, os.environ.get("METRICS", "fish_metrics.csv")))
modes = {s["scientific"]:s["mating_mode"] for s in SPECIES}
df["mating_mode"] = df["scientific"].map(modes)
MINIMG = int(os.environ.get("MINIMG", 15))    # MINIMG=1 opens the floodgates: every species that has a cut-out
q = df[df.n_images >= MINIMG].copy().reset_index(drop=True)
q["png"] = q["scientific"].apply(lambda s: os.path.join(CUT, s.replace(" ","_")+".png"))
q = q[q["png"].apply(os.path.exists)].reset_index(drop=True)
x = q["mate_choice_index"].values.astype(float)
y = q["hue_entropy"].values.astype(float)

fig, ax = plt.subplots(figsize=(22, 14))
XLIM = (-0.8, 10.8); YLIM = (1.15, 4.15)
ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
rng = np.random.default_rng(5)
jx = x + rng.uniform(-0.16, 0.16, len(x))

# ---------------------------------------------------------------- the wash
def hex2rgb(h):
    h = h.lstrip("#"); return np.array([int(h[i:i+2],16)/255 for i in (0,2,4)])

NX, NY = 560, 380
gx = np.linspace(*XLIM, NX); gy = np.linspace(*YLIM, NY)
GX, GY = np.meshgrid(gx, gy)

dens, eff = {}, {}
for mode in MODE_ORDER:
    m = (q["mating_mode"] == mode).values
    d = np.zeros_like(GX)
    for xi, yi in zip(jx[m], y[m]):
        d += np.exp(-0.5*(((GX-xi)/HX)**2 + ((GY-yi)/HY)**2))
    dens[mode] = d
    eff[mode]  = d/max(m.sum(), 1)**NORM          # size-discounted, for the colour contest

# how much "ink" the paper takes here: full where fish are dense, none where empty.
# Referenced to a high quantile rather than the peak, or the one very tight cluster
# (broadcast spawners, all piled at x=0) would set the scale and bleach everything else.
tot = sum(dens.values()) + 1e-9
te  = sum(eff.values()) + 1e-9
t = tot/np.quantile(tot, 0.90)
mass = np.clip((t - LO)/(HI - LO), 0, 1)
mass = mass*mass*(3 - 2*mass)          # smoothstep -> soft, rounded splotch edges

canvas = np.ones((NY, NX, 3)) * hex2rgb(PAPER)
alphas = {}
for mode in MODE_ORDER:
    share = eff[mode]/te                         # this mode's slice of the local density
    a = MAXA * (share**GAMMA) * mass
    alphas[mode] = a
    col = np.clip(0.5 + (hex2rgb(PAL[mode]) - 0.5)*SAT, 0, 1)
    canvas = canvas*(1 - a[...,None]) + col[None,None,:]*a[...,None]

# pigment pooling: watercolour darkens slightly where a splotch meets its edge
edge = np.zeros((NY, NX))
for mode in MODE_ORDER:
    gyx, gxx = np.gradient(alphas[mode])
    edge = np.maximum(edge, np.hypot(gxx, gyx))
edge = gaussian_filter(edge, 2.0)
if edge.max() > 0: edge /= edge.max()
canvas *= (1 - 0.09*edge[...,None])

# a breath of paper grain so it reads as pigment, not a gradient
grain = rng.normal(0, 1, (NY, NX, 1))
canvas = np.clip(canvas + gaussian_filter(grain, 1.2)*0.009, 0, 1)

ax.imshow(canvas, origin="lower", extent=[*XLIM, *YLIM], aspect="auto",
          interpolation="bilinear", zorder=0)

# ------------------------------------------------------- trend + its spread
b1, b0 = np.polyfit(x, y, 1)
xl = np.linspace(XLIM[0], XLIM[1], 200)
fit = b0 + b1*xl

resid = y - (b0 + b1*x)
sd = resid.std(ddof=2)

boot = np.empty((1000, len(xl)))
for i in range(1000):
    k = rng.integers(0, len(x), len(x))
    bb1, bb0 = np.polyfit(x[k], y[k], 1)
    boot[i] = bb0 + bb1*xl
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

# The spread of the species about the line (±1 SD). It covers a big diagonal
# swathe, so it has to be whispered — at any real opacity it stops being a veil
# over the wash and becomes a pink stripe competing with it.
ax.fill_between(xl, fit-sd, fit+sd, color=ACCENT, alpha=0.035, lw=0, zorder=1)
ax.plot(xl, fit-sd, color=ACCENT, lw=0.7, alpha=0.20, zorder=1)
ax.plot(xl, fit+sd, color=ACCENT, lw=0.7, alpha=0.20, zorder=1)
# the uncertainty of the line itself (95% bootstrap CI) — the inner ribbon
ax.fill_between(xl, lo, hi, color=ACCENT, alpha=0.22, lw=0, zorder=1)
# the line: a pale casing underneath so it stays legible over any splotch
ax.plot(xl, fit, color=PAPER, lw=6.5, alpha=0.55, solid_capstyle="round", zorder=2)
ax.plot(xl, fit, color=ACCENT, lw=3.2, alpha=0.92, solid_capstyle="round", zorder=2)

# ------------------------------------------------------------------- the fish
def with_shadow(img):
    """A soft drop-shadow so each cut-out lifts off the wash."""
    a = gaussian_filter(img[...,3], 3.2)
    sh = np.zeros_like(img)
    sh[...,:3] = 0.18
    sh[...,3] = a*0.32
    return sh

order = np.argsort([-plt.imread(p).shape[0]*plt.imread(p).shape[1] for p in q["png"]])
for i in order:
    img = plt.imread(q["png"].iloc[i])
    if img.shape[2] == 3:
        img = np.dstack([img, np.ones(img.shape[:2], img.dtype)])
    zoom = 62.0 / max(img.shape[:2])
    ax.add_artist(AnnotationBbox(OffsetImage(with_shadow(img), zoom=zoom), (jx[i], y[i]),
                  xybox=(3.5, -3.5), boxcoords="offset points",
                  frameon=False, zorder=3, box_alignment=(0.5,0.5)))
    ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=zoom), (jx[i], y[i]),
                  frameon=False, zorder=4, box_alignment=(0.5,0.5)))

# --------------------------------------------- name each splotch in the plot
# Label a mode where it actually owns the paper, rather than in a legend the eye
# has to travel to. Two constraints: the label must sit where that mode wins the
# colour contest, and as far from any cut-out as possible so it lands on open wash.
def darken(h, f=0.55):
    return tuple(np.clip(hex2rgb(h)*f, 0, 1))

stack = np.stack([alphas[m] for m in MODE_ORDER])          # (mode, ny, nx)
dom   = np.argmax(stack, axis=0)
fishd = np.full(GX.shape, np.inf)                          # distance to nearest fish
for xi, yi in zip(jx, y):
    fishd = np.minimum(fishd, np.hypot((GX-xi)/HX, (GY-yi)/HY))

# keep the text off the frame: a centred label needs room for its own width
inside = ((GX > XLIM[0]+1.9) & (GX < XLIM[1]-1.9) &
          (GY > YLIM[0]+0.15) & (GY < YLIM[1]-0.12))

placed, labeled = [], []
for mi, mode in enumerate(MODE_ORDER):
    a = alphas[mode]
    core = a/max(a.max(), 1e-9)
    own = (dom == mi) & (core > 0.60) & (mass > 0.60) & inside
    if not own.any():                       # nowhere it clearly owns -> relax once
        own = (dom == mi) & (core > 0.35) & inside
    if not own.any():
        continue
    # sit in open wash (far from any fish) but stay in the heart of the colour
    score = np.where(own, fishd + 1.6*core, -np.inf)
    for px, py in placed:                                  # keep labels off each other
        score -= 6.0*np.exp(-0.5*(((GX-px)/1.1)**2 + ((GY-py)/0.34)**2))
    j, i = np.unravel_index(np.argmax(score), score.shape)
    lx, ly = gx[i], gy[j]
    placed.append((lx, ly)); labeled.append(mode)
    ax.text(lx, ly, mode, fontsize=17, style="italic", color=darken(PAL[mode]),
            ha="center", va="center", zorder=5,
            path_effects=[pe.withStroke(linewidth=4.5, foreground=PAPER, alpha=0.85)])

missing = [m for m in MODE_ORDER if m not in labeled]

# ------------------------------------------------------------------ furniture
ax.set_xticks(range(0,11,2))
ax.tick_params(labelsize=16, length=5, pad=6)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(color=GRID, lw=0.6, alpha=0.35, zorder=0.5); ax.set_axisbelow(False)
ax.set_xlabel("mate-choice index      random / broadcast spawning   to   strict female choice",
              fontsize=21, labelpad=14)
ax.set_ylabel("colour variety   —   distinct hues worn by the fish", fontsize=21, labelpad=14)

rho, p = stats.spearmanr(x, y)
pstr = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"     # never hard-code the p-value
fig.text(0.055, 0.947, f"each fish sits at its own mate-choice and colour-variety   ·   {len(q)} species   ·   "
         f"the palette widens rightward  (ρ = {rho:+.2f}, {pstr})", fontsize=19, color="#5a5346")
fig.text(0.055, 0.917, "background colour = the mating behaviour that dominates that region of the plane;   "
         "the band is the ±1 SD spread of species about the trend",
         fontsize=15, color="#8a8172", style="italic")

if missing:   # any mode with no room to be named in-plot still gets a swatch below
    handles = [Patch(facecolor=PAL[m], edgecolor="none", alpha=0.8, label=m) for m in missing]
    ax.legend(handles=handles, frameon=False, loc="lower center", ncol=len(missing),
              fontsize=15, bbox_to_anchor=(0.5, -0.115), handlelength=1.6, handleheight=1.0)
fig.subplots_adjust(left=0.062, right=0.975, top=0.895,
                    bottom=0.115 if missing else 0.085)
fig.savefig(os.path.join(ROOT, OUTPNG), dpi=145, facecolor=PAPER)
print(f"wrote {OUTPNG}  ({len(q)} fish, cutouts from {os.path.basename(CUT)})")
