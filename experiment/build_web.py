"""Build the interactive version of the gallery (fig4) as a static web page.

Writes ../docs/ — the folder GitHub Pages serves — containing:

    index.html            the page (self-contained; no CDN, no build step)
    assets/wash.webp      the watercolour background field, rendered by the same
                          kernel-density code as fig4_fish_gallery.py
    assets/fish/*.webp    one cut-out per species, downscaled from data/cutouts/
    data.json             every species' coordinates, scores and metrics, plus the
                          trend line, its bootstrap CI, and the in-plot mode labels

The wash, the jitter, the trend and the label placement are computed here exactly
as in fig4 (same seed, same call order), so the live page is the figure — only
now you can walk around inside it.
"""
import warnings; warnings.filterwarnings("ignore")
import os, json, shutil
import numpy as np, pandas as pd
from PIL import Image
from scipy import stats
from scipy.ndimage import gaussian_filter
from fish_data import SPECIES, MODE_ORDER

ROOT = os.path.dirname(os.path.abspath(__file__))
CUT  = os.path.join(ROOT, "data", "cutouts")
DOCS = os.path.abspath(os.path.join(ROOT, "..", "docs"))
FISHDIR = os.path.join(DOCS, "assets", "fish")

# --- identical to fig4 -------------------------------------------------------
PAL = {"Broadcast / group spawn":"#6b8ea3","Pelagic pair spawn":"#4f9d8e",
       "Nest builder / guarder":"#d7a13b","Harem / sex-changer":"#cf8050",
       "Mouthbrooder":"#94a04f","Male brooder":"#9481b0","Livebearer (internal)":"#cf6b6b"}
PAPER="#f4ecdb"; INK="#3a342b"; ACCENT="#9b3d2e"; GRID="#ddd0b7"
MAXA, GAMMA = 1.00, 2.4
HX, HY = 0.90, 0.28
LO, HI = 0.06, 0.34
SAT, NORM = 1.75, 0.5
XLIM = (-0.8, 10.8); YLIM = (1.15, 4.15)

MAX_FISH_PX = 512          # longest side of a cut-out on the web (most are smaller and stay native)
WASH_SCALE  = 2            # render the wash at 2x fig4's grid, for zooming into

os.makedirs(FISHDIR, exist_ok=True)

def hex2rgb(h):
    h = h.lstrip("#"); return np.array([int(h[i:i+2],16)/255 for i in (0,2,4)])

# ---------------------------------------------------------------- the species
df = pd.read_csv(os.path.join(ROOT, os.environ.get("METRICS", "fish_metrics.csv")))
modes = {s["scientific"]: s["mating_mode"] for s in SPECIES}
df["mating_mode"] = df["scientific"].map(modes)
MINIMG = int(os.environ.get("MINIMG", 15))    # MINIMG=1 opens the floodgates: every species that has a cut-out
q = df[df.n_images >= MINIMG].copy().reset_index(drop=True)
q["png"] = q["scientific"].apply(lambda s: os.path.join(CUT, s.replace(" ", "_") + ".png"))
q = q[q["png"].apply(os.path.exists)].reset_index(drop=True)

x = q["mate_choice_index"].values.astype(float)
y = q["hue_entropy"].values.astype(float)

rng = np.random.default_rng(5)                    # same seed, same first draw as fig4
jx = x + rng.uniform(-0.16, 0.16, len(x))         # -> the fish land where they do in the PNG

# ------------------------------------------------------------------- the wash
NX, NY = 560*WASH_SCALE, 380*WASH_SCALE
gx = np.linspace(*XLIM, NX); gy = np.linspace(*YLIM, NY)
GX, GY = np.meshgrid(gx, gy)

dens, eff = {}, {}
for mode in MODE_ORDER:
    m = (q["mating_mode"] == mode).values
    d = np.zeros_like(GX)
    for xi, yi in zip(jx[m], y[m]):
        d += np.exp(-0.5*(((GX-xi)/HX)**2 + ((GY-yi)/HY)**2))
    dens[mode] = d
    eff[mode]  = d/max(m.sum(), 1)**NORM

tot = sum(dens.values()) + 1e-9
te  = sum(eff.values()) + 1e-9
t = tot/np.quantile(tot, 0.90)
mass = np.clip((t - LO)/(HI - LO), 0, 1)
mass = mass*mass*(3 - 2*mass)

canvas = np.ones((NY, NX, 3)) * hex2rgb(PAPER)
alphas = {}
for mode in MODE_ORDER:
    share = eff[mode]/te
    a = MAXA * (share**GAMMA) * mass
    alphas[mode] = a
    col = np.clip(0.5 + (hex2rgb(PAL[mode]) - 0.5)*SAT, 0, 1)
    canvas = canvas*(1 - a[...,None]) + col[None,None,:]*a[...,None]

edge = np.zeros((NY, NX))
for mode in MODE_ORDER:
    gyx, gxx = np.gradient(alphas[mode])
    edge = np.maximum(edge, np.hypot(gxx, gyx))
edge = gaussian_filter(edge, 2.0*WASH_SCALE)
if edge.max() > 0: edge /= edge.max()
canvas *= (1 - 0.09*edge[...,None])

grain = rng.normal(0, 1, (NY, NX, 1))
canvas = np.clip(canvas + gaussian_filter(grain, 1.2*WASH_SCALE)*0.009, 0, 1)

# origin="lower" in matplotlib -> flip for an image whose first row is the top
wash = Image.fromarray((canvas[::-1]*255).astype(np.uint8))
wash.save(os.path.join(DOCS, "assets", "wash.webp"), quality=92, method=6)

# ------------------------------------------------------- trend + its spread
b1, b0 = np.polyfit(x, y, 1)
xl = np.linspace(XLIM[0], XLIM[1], 200)
fit = b0 + b1*xl
sd = (y - (b0 + b1*x)).std(ddof=2)
boot = np.empty((1000, len(xl)))
for i in range(1000):
    k = rng.integers(0, len(x), len(x))
    bb1, bb0 = np.polyfit(x[k], y[k], 1)
    boot[i] = bb0 + bb1*xl
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

# --------------------------------------------- where each mode owns the paper
stack = np.stack([alphas[m] for m in MODE_ORDER])
dom   = np.argmax(stack, axis=0)
fishd = np.full(GX.shape, np.inf)
for xi, yi in zip(jx, y):
    fishd = np.minimum(fishd, np.hypot((GX-xi)/HX, (GY-yi)/HY))
inside = ((GX > XLIM[0]+1.9) & (GX < XLIM[1]-1.9) &
          (GY > YLIM[0]+0.15) & (GY < YLIM[1]-0.12))

labels, placed = [], []
for mi, mode in enumerate(MODE_ORDER):
    a = alphas[mode]
    core = a/max(a.max(), 1e-9)
    own = (dom == mi) & (core > 0.60) & (mass > 0.60) & inside
    if not own.any():
        own = (dom == mi) & (core > 0.35) & inside
    if not own.any():
        continue
    score = np.where(own, fishd + 1.6*core, -np.inf)
    for px, py in placed:
        score -= 6.0*np.exp(-0.5*(((GX-px)/1.1)**2 + ((GY-py)/0.34)**2))
    j, i = np.unravel_index(np.argmax(score), score.shape)
    placed.append((gx[i], gy[j]))
    labels.append({"mode": mode, "x": float(gx[i]), "y": float(gy[j])})

# ------------------------------------------------------------------- the fish
fish = []
for i in range(len(q)):
    r = q.iloc[i]
    src = r["png"]
    im = Image.open(src).convert("RGBA")
    s = MAX_FISH_PX/max(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width*s)), max(1, round(im.height*s))), Image.LANCZOS)
    name = r["scientific"].replace(" ", "_") + ".webp"
    im.save(os.path.join(FISHDIR, name), quality=86, method=6)
    fish.append({
        "img": name,
        "w": im.width, "h": im.height,
        "scientific": r["scientific"], "common": r["common"],
        "family": r["family"], "habitat": r["habitat"], "note": r["note"],
        "mode": r["mating_mode"],
        "x": float(jx[i]), "y": float(y[i]),
        "mci": float(r["mate_choice_index"]),
        "c": [int(r["c_fert"]), int(r["c_court"]), int(r["c_system"]), int(r["c_choice"])],
        "hue_entropy": float(r["hue_entropy"]),
        "colorfulness": float(r["colorfulness"]),
        "saturation": float(r["saturation"]),
        "n_images": int(r["n_images"]),
    })

rho, p = stats.spearmanr(x, y)
payload = {
    "xlim": list(XLIM), "ylim": list(YLIM),
    "palette": PAL, "modeOrder": MODE_ORDER,
    "paper": PAPER, "ink": INK, "accent": ACCENT, "grid": GRID,
    "rho": float(rho), "p": float(p), "n": len(q),
    "trend": {"x": xl.round(4).tolist(), "fit": fit.round(4).tolist(),
              "lo": lo.round(4).tolist(), "hi": hi.round(4).tolist(),
              "sdLo": (fit-sd).round(4).tolist(), "sdHi": (fit+sd).round(4).tolist()},
    "labels": labels,
    "fish": fish,
}
with open(os.path.join(DOCS, "data.json"), "w") as f:
    json.dump(payload, f, separators=(",", ":"))

open(os.path.join(DOCS, ".nojekyll"), "w").close()

size = sum(os.path.getsize(os.path.join(dp, f))
           for dp, _, fs in os.walk(DOCS) for f in fs)
print(f"wrote docs/  ({len(fish)} fish, {len(labels)} in-plot labels, "
      f"rho={rho:+.2f}, {size/1e6:.1f} MB total)")
