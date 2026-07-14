"""Does the improved segmentation change the result?

The gallery was drawn with the good segmenter (isnet @768 + shape gates + the fish/coral
classifier veto) but *scored* with the old one (u2netp @400, no gate, no veto). This
script re-runs the whole comparison on the two metric tables and reports whether the
headline — colour variety tracks mate choice, rho = +0.31 — survives being measured on
masks that are actually fish.

    python features_isnet.py          # writes fish_metrics_isnet.csv
    python compare_segmenters.py
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
MIN_IMG = 15          # the data-quality rule used by the figures

METRICS = ["hue_entropy", "colorfulness", "saturation", "sat_p90", "high_chroma_frac",
           "n_color_clusters", "contrast", "edge_density", "aspect_ratio", "body_depth",
           "solidity", "silhouette_cplx", "extent"]

old = pd.read_csv(os.path.join(ROOT, "fish_metrics.csv"))
new = pd.read_csv(os.path.join(ROOT, "fish_metrics_isnet.csv"))


def rho(d, col):
    d = d[d.n_images >= MIN_IMG].dropna(subset=[col, "mate_choice_index"])
    if len(d) < 10:
        return np.nan, np.nan, len(d)
    r, p = stats.spearmanr(d["mate_choice_index"], d[col])
    return r, p, len(d)


print(f"photos per species kept (median):  u2netp {old.n_images.median():.0f}   "
      f"isnet+gates+veto {new.n_images.median():.0f}")
print(f"species passing the >= {MIN_IMG}-photo rule:   u2netp {(old.n_images >= MIN_IMG).sum()}   "
      f"isnet+gates+veto {(new.n_images >= MIN_IMG).sum()}\n")

print("Spearman rho of each measure against the mate-choice index")
print(f"{'measure':18s} {'OLD rho':>8s} {'p':>8s} {'n':>4s}   {'NEW rho':>8s} {'p':>8s} {'n':>4s}   {'shift':>7s}")
print("-"*76)
for c in METRICS:
    ro, po, no = rho(old, c)
    rn, pn, nn = rho(new, c)
    if np.isnan(ro) or np.isnan(rn):
        continue
    flag = ""
    if (po < 0.05) != (pn < 0.05):
        flag = "  <- significance flips"
    print(f"{c:18s} {ro:+8.3f} {po:8.4f} {no:4d}   {rn:+8.3f} {pn:8.4f} {nn:4d}   "
          f"{rn-ro:+7.3f}{flag}")

# ---- how much did individual species move, and which ones? -------------------
m = old.merge(new, on="scientific", suffixes=("_old", "_new"))
m = m[(m.n_images_old >= MIN_IMG) & (m.n_images_new >= MIN_IMG)]
d = (m.hue_entropy_new - m.hue_entropy_old)
r_agree = stats.spearmanr(m.hue_entropy_old, m.hue_entropy_new)[0]
print(f"\ncolour variety, old vs new, on the {len(m)} species both pipelines keep:")
print(f"   rank agreement rho = {r_agree:+.3f}   mean shift = {d.mean():+.3f}   "
      f"|shift| median = {d.abs().median():.3f}")

big = m.assign(shift=d).reindex(d.abs().sort_values(ascending=False).index).head(8)
print("\n   the species the new segmenter moved most:")
for _, r in big.iterrows():
    print(f"     {r['common_old'][:28]:30s} {r['hue_entropy_old']:.2f} -> {r['hue_entropy_new']:.2f} "
          f"({r['shift']:+.2f})   photos {int(r['n_images_old'])} -> {int(r['n_images_new'])}")

dropped = set(old[old.n_images >= MIN_IMG].scientific) - set(new[new.n_images >= MIN_IMG].scientific)
if dropped:
    names = old.set_index("scientific").loc[sorted(dropped), "common"].tolist()
    print(f"\n   {len(dropped)} species fall below the {MIN_IMG}-photo bar under the stricter pipeline:")
    print("     " + ", ".join(names[:12]) + (" ..." if len(names) > 12 else ""))
