"""How much does the >= 15-photo rule matter — and does the answer depend on it?

The README defends the >= 15 rule by noting that filtering *strengthened* the signal
(rho +0.19 -> +0.31), which is the right direction if the discarded species were noise.
But "the result got better when I dropped data" is exactly what p-hacking also looks
like, so the honest move is not to defend the threshold: it is to show the result at
EVERY threshold, including no threshold at all, and let the reader see the whole curve.

A per-species value is a median over n photos. At n = 1 that median is one photo, and a
photo of a fish in a bucket is worth as much as a portrait. So we expect attenuation
toward zero as the bar drops, not a sign flip — attenuation is measurement noise diluting
a real effect. If instead the effect only exists above some bar, that is a red flag, and
it should be visible here.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
old = pd.read_csv(os.path.join(ROOT, "fish_metrics.csv"))
new = pd.read_csv(os.path.join(ROOT, "fish_metrics_isnet.csv"))

COL = "hue_entropy"


def at(df, lo):
    d = df[df.n_images >= lo].dropna(subset=[COL, "mate_choice_index"])
    if len(d) < 8:
        return np.nan, np.nan, len(d)
    r, p = stats.spearmanr(d["mate_choice_index"], d[COL])
    return r, p, len(d)


print("colour variety (hue_entropy) vs mate choice, at every photo-count bar\n")
print(f"{'min photos':>10s}   {'OLD (u2netp)':>22s}   {'NEW (isnet+gates+veto)':>24s}")
print(f"{'':>10s}   {'rho':>7s} {'p':>8s} {'n':>4s}   {'rho':>7s} {'p':>8s} {'n':>4s}")
print("-" * 62)
for lo in [1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 25]:
    ro, po, no = at(old, lo)
    rn, pn, nn = at(new, lo)
    tag = "   <- the rule the figures use" if lo == 15 else ""
    tag = "   <- floodgates open" if lo == 1 else tag
    print(f"{lo:>10d}   {ro:+7.3f} {po:8.4f} {no:4d}   {rn:+7.3f} {pn:8.4f} {nn:4d}{tag}")

# Weighting is the principled alternative to a cliff: keep every species, but let a
# species measured on 25 photos count for more than one measured on a single photo.
print("\nrather than a cliff — keep every species, weight it by sqrt(photos):")
for name, df in [("OLD (u2netp)", old), ("NEW (isnet)", new)]:
    d = df.dropna(subset=[COL, "mate_choice_index"])
    x = stats.rankdata(d["mate_choice_index"]); y = stats.rankdata(d[COL])
    w = np.sqrt(d["n_images"].values)
    xm = np.average(x, weights=w); ym = np.average(y, weights=w)
    cov = np.average((x-xm)*(y-ym), weights=w)
    r = cov/np.sqrt(np.average((x-xm)**2, weights=w)*np.average((y-ym)**2, weights=w))
    # p-value from the weighted r via an effective sample size
    neff = w.sum()**2/np.sum(w**2)
    t = r*np.sqrt(max(neff-2, 1)/max(1-r*r, 1e-12))
    p = 2*stats.t.sf(abs(t), max(neff-2, 1))
    print(f"   {name:14s} weighted rho = {r:+.3f}   p = {p:.4f}   "
          f"n = {len(d)} species (n_eff = {neff:.0f})")

print("\nhow many species each bar throws away:")
for name, df in [("OLD", old), ("NEW", new)]:
    tot = len(df)
    print(f"   {name}: {tot} species measured; "
          f"{(df.n_images < 15).sum()} sit below 15 photos, "
          f"{(df.n_images < 5).sum()} below 5, {(df.n_images == 1).sum()} have just one")
