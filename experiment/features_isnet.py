"""Re-measure the striking-ness metrics with the GOOD segmenter.

The point of this script is a fairness check on the headline result. The gallery
(make_cutouts.py) is drawn with isnet-general-use @768 + shape gates + a fish/coral
classifier veto — but the numbers those fish are *plotted at* came from features.py,
which trusts whatever u2netp @400px returned, with no shape gate and no veto. So the
figure was drawn with the good segmenter and scored with the bad one. If a mask grabs
coral or an aquarium wall, we measured the colour variety of coral.

Here every photo that contributes to a species' median must survive the same bar the
gallery cut-outs do:

  1. prefilter  — cheap cached u2netp masks rank the ~80 photos; keep the top CANDS
                  that aren't already hard-rejected on shape.
  2. re-segment — isnet-general-use at 768px (cached in data/masks_isnet/).
  3. gate       — fishiness() hard-rejects rectangular crops, frame-hugging blobs,
                  fragmented masks, sprawling coral.
  4. veto       — fishnet's ImageNet ResNet-50 must think the blob is more fish than
                  reef furniture / diver. Note this is the *comparative* ratio
                  p_fish/(p_fish+p_reef), not the raw p_fish: most reef fish are not
                  ImageNet classes at all, so a raw-probability threshold would throw
                  away perfectly good fish. Photos are still *ranked* by the same
                  shape x (0.15 + 1.2*p_fish) score make_cutouts uses.
  5. measure    — the 13 metrics of features.py, on the largest connected component
                  only, so stray blobs cannot contribute pixels.

Median per species, as before. Writes fish_metrics_isnet.csv with the same columns as
fish_metrics.csv, so every downstream script can read it via METRICS=fish_metrics_isnet.csv.
"""
import warnings; warnings.filterwarnings("ignore")
import os, glob, sys
import numpy as np, cv2, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG  = os.path.join(ROOT, "data", "images")

CANDS = int(os.environ.get("CANDS", 40))   # photos per species promoted to isnet
WANT  = int(os.environ.get("WANT", 25))    # best survivors kept for the median (features.py uses 25)
RVETO = float(os.environ.get("RVETO", 0.5))  # reject if ImageNet calls it more reef than fish

from features import colour_feats, pattern_feats, shape_feats   # the SAME metrics
from segment import mask_for, load_resized
from make_cutouts import isnet_mask, fishiness, cutout          # the SAME quality bar


def clean_fg(m, shape):
    """Largest connected component of the mask, opened/closed — the fish, and only it."""
    if m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]))
    b = (m > 128).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    ncc, lbl, st, _ = cv2.connectedComponentsWithStats(b, 8)
    if ncc <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    fg = (lbl == k)
    # erode by one, matching cutout()'s halo trim, so mask fringe pixels (which carry
    # background colour) cannot inflate hue variety
    fg = cv2.erode(fg.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)
    return fg if fg.sum() > 800 else None


def prefilter(sci, n=CANDS):
    """Rank a species' photos with the cheap cached u2netp masks (same idea as make_cutouts)."""
    d = os.path.join(IMG, sci.replace(" ", "_"))
    out = []
    for fp in sorted(glob.glob(os.path.join(d, "*.jpg"))):
        m = mask_for(fp); img = load_resized(fp)
        if m is None or img is None:
            continue
        s, _ = fishiness(img, m)
        if s > 0:
            out.append((s, fp))
    out.sort(key=lambda t: -t[0])
    return [fp for _, fp in out[:n]]


def process_species(sci):
    from fishnet import fishness as netfish
    rows, seen, killed = [], 0, {"shape": 0, "veto": 0, "mask": 0}
    for fp in prefilter(sci):
        img, m = isnet_mask(fp)                       # cached where make_cutouts already ran
        if m is None:
            continue
        seen += 1
        s, _ = fishiness(img, m)                      # the shape gates
        if s <= 0:
            killed["shape"] += 1; continue
        rgba = cutout(img, m)
        if rgba is None:
            killed["mask"] += 1; continue
        ratio, pfish, _ = netfish(rgba)               # the classifier veto
        if ratio < RVETO:                             # more reef furniture than fish
            killed["veto"] += 1; continue
        fg = clean_fg(m, img.shape[:2])
        if fg is None:
            killed["mask"] += 1; continue
        try:
            rows.append({"_score": s*(0.15 + 1.2*pfish), "_p_fish": pfish, "_ratio": ratio,
                         **colour_feats(img, fg), **pattern_feats(img, fg), **shape_feats(fg)})
        except Exception:
            continue
    rows.sort(key=lambda r: -r["_score"])             # best-scoring photos first, as in make_cutouts
    return rows[:WANT], {"seen": seen, **killed}


def _worker(sp):
    try:
        return sp["scientific"], *process_species(sp["scientific"])
    except Exception:
        return sp["scientific"], [], {"seen": 0, "shape": 0, "veto": 0, "mask": 0}


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from fish_data import SPECIES
    from multiprocessing import Pool

    only = set(sys.argv[1:]) or None
    todo = [sp for sp in SPECIES if not only or sp["scientific"] in only]

    per_image, agg, results = [], [], {}
    tally = {"seen": 0, "shape": 0, "veto": 0, "mask": 0}
    with Pool(5) as pool:                             # 5 x 2 threads, leaves the box usable
        for i, (sci, rows, kill) in enumerate(pool.imap_unordered(_worker, todo)):
            results[sci] = rows
            for k in tally: tally[k] += kill.get(k, 0)
            print(f"[{i+1:3d}/{len(todo)}] {sci:34s} {len(rows):2d} kept  "
                  f"(of {kill['seen']:2d} isnet: shape killed {kill['shape']}, "
                  f"classifier vetoed {kill['veto']})", flush=True)

    for sp in todo:
        rows = results.get(sp["scientific"], [])
        for r in rows:
            per_image.append({"scientific": sp["scientific"], **r})
        if not rows:
            continue
        med = pd.DataFrame(rows).drop(columns=["_score"]).median(numeric_only=True).to_dict()
        agg.append({**sp, "n_images": len(rows), **med})

    print(f"\nphotos re-segmented with isnet: {tally['seen']}  |  "
          f"killed by shape gates: {tally['shape']}  |  vetoed by the classifier: {tally['veto']}  |  "
          f"unusable mask: {tally['mask']}")

    pd.DataFrame(per_image).to_csv(os.path.join(ROOT, "metrics_per_image_isnet.csv"), index=False)
    out = pd.DataFrame(agg)
    out.to_csv(os.path.join(ROOT, "fish_metrics_isnet.csv"), index=False)
    print(f"\nwrote fish_metrics_isnet.csv  ({len(out)} species, "
          f"{(out.n_images >= 15).sum()} with >= 15 clean photos)")


if __name__ == "__main__":
    main()
