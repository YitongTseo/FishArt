"""Re-pick a cut-out for species the automatic cascade got wrong.

make_cutouts.py shortlists only 12 photos (ranked by the cheap u2netp prefilter) and then
takes whatever wins its shape x fishness score. For a handful of species that is not enough:
the foxface lost its face to a truncated mask, the Moorish idol scored a *rock* with a fish
next to it, and the rasbora and swordtail came with their aquariums attached.

This script does the expensive thing for a named species — segment EVERY photo with isnet,
score them all, and lay the survivors out as a contact sheet so a human can look. Nothing is
auto-committed: `--sheet` proposes, `--pick` disposes.

    python repick.py --sheet "Zanclus cornutus"        # -> repick/Zanclus_cornutus.png
    python repick.py --pick  "Zanclus cornutus" 041    # -> data/cutouts/Zanclus_cornutus.png
"""
import warnings; warnings.filterwarnings("ignore")
import os, glob, sys, argparse
import numpy as np, cv2
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG  = os.path.join(ROOT, "data", "images")
OUT  = os.path.join(ROOT, "data", "cutouts")
SHEETS = os.path.join(ROOT, "repick")
os.makedirs(SHEETS, exist_ok=True)

from make_cutouts import isnet_mask, fishiness, cutout
from fishnet import fishness as netfish

TOP = 15          # how many candidates to show on the sheet
PAPER = (244, 236, 219)


def candidates(sci):
    d = os.path.join(IMG, sci.replace(" ", "_"))
    rows = []
    for fp in sorted(glob.glob(os.path.join(d, "*.jpg"))):
        img, m = isnet_mask(fp)                       # every photo, not a shortlist of 12
        if m is None:
            continue
        s, parts = fishiness(img, m)
        if s <= 0:
            continue
        rgba = cutout(img, m)
        if rgba is None:
            continue
        ratio, pf, _ = netfish(rgba)
        rows.append(dict(fp=fp, tag=os.path.basename(fp)[:-4], rgba=rgba,
                         shape=s, ratio=ratio, p_fish=pf,
                         total=s*(0.15 + 1.2*pf)*(0.3 + 0.7*ratio)))
    rows.sort(key=lambda r: -r["total"])
    return rows


def sheet(sci):
    rows = candidates(sci)[:TOP]
    if not rows:
        print(f"{sci}: no candidate survived the shape gates"); return
    cols, CW, CH = 5, 300, 250
    n = len(rows)
    r_ = (n + cols - 1)//cols
    im = Image.new("RGB", (cols*CW, r_*CH), PAPER)
    from PIL import ImageDraw
    dr = ImageDraw.Draw(im)
    for i, r in enumerate(rows):
        c = Image.fromarray(r["rgba"], "RGBA")
        c.thumbnail((CW-16, CH-40))
        x, y = (i % cols)*CW, (i//cols)*CH
        bg = Image.new("RGBA", (CW, CH), PAPER + (255,))
        bg.paste(c, ((CW-c.width)//2, (CH-30-c.height)//2 + 8), c)
        im.paste(bg.convert("RGB"), (x, y))
        dr.text((x+8, y+CH-22),
                f'{r["tag"]}   shape {r["shape"]:.2f}  fish {r["ratio"]:.2f}  tot {r["total"]:.3f}',
                fill=(90, 83, 70))
    p = os.path.join(SHEETS, sci.replace(" ", "_") + ".png")
    im.save(p)
    print(f"{sci}: {len(rows)} candidates -> {p}")


def pick(sci, tag):
    fp = os.path.join(IMG, sci.replace(" ", "_"), f"{tag}.jpg")
    if not os.path.exists(fp):
        print(f"no such photo: {fp}"); return
    img, m = isnet_mask(fp)
    rgba = cutout(img, m)
    if rgba is None:
        print("cut-out failed"); return
    out = os.path.join(OUT, sci.replace(" ", "_") + ".png")
    cv2.imwrite(out, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    s, _ = fishiness(img, m)
    ratio, _, _ = netfish(rgba)
    print(f"{sci}: wrote {out} from {tag}.jpg  (shape {s:.2f}, fish-vs-reef {ratio:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", nargs="+", metavar="SPECIES")
    ap.add_argument("--pick", nargs=2, metavar=("SPECIES", "TAG"))
    a = ap.parse_args()
    if a.pick:
        pick(*a.pick)
    for s in (a.sheet or []):
        sheet(s)
