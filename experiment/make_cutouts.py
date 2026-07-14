"""Rebuild the per-species cut-outs with a segmentation pass that actually
knows what a fish looks like.

The v1 pipeline (archive/make_cutouts_v1.py) trusted whatever blob u2netp
returned, so a third of the gallery ended up being coral heads, aquarium walls,
dive gloves and petri dishes. Three changes fix that:

  1. quality:  re-segment with isnet-general-use at 768px (u2netp @400px is
     used only as a cheap prefilter to shortlist candidate photos).
  2. rejection: score the *shape* of the mask, not just its size. A fish is a
     laterally-elongated, fairly-but-not-perfectly convex blob that does not
     hug the frame border. Rectangular crops (whole-tank shots), sprawling
     coral and hand-held fish are killed outright rather than ranked low.
  3. veto:     shape alone cannot separate a pufferfish from a brain coral —
     both are compact convex blobs. So an ImageNet classifier (fishnet.py) is
     asked whether the blob looks more like a fish or like reef furniture, and
     a losing photo is replaced by a *different photo of the same species*
     rather than costing the species its place in the gallery.

Anything that fails the bar is left with no cut-out: an absent fish is better
than a picture of a glove. Scores land in cutout_quality.csv so every pick is
auditable.
"""
import warnings; warnings.filterwarnings("ignore")
import os, glob, sys, json
import numpy as np, cv2
from fish_data import SPECIES
from segment import mask_for, load_resized

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG  = os.path.join(ROOT, "data", "images")
OUT  = os.path.join(ROOT, "data", "cutouts")
MASK2 = os.path.join(ROOT, "data", "masks_isnet")
os.makedirs(OUT, exist_ok=True); os.makedirs(MASK2, exist_ok=True)

SEG_RES  = 768      # resolution for the good masks
SHORTLIST = int(os.environ.get("SHORTLIST", 12))   # photos per species promoted from prefilter to isnet
ACCEPT   = 0.08     # minimum combined (shape x fishness) score to earn a cut-out

_S = None
def isnet():
    global _S
    if _S is None:
        from rembg import new_session
        _S = new_session("isnet-general-use")
    return _S

def load_big(fp):
    img = cv2.imread(fp)
    if img is None: return None
    h, w = img.shape[:2]; sc = SEG_RES / max(h, w)
    if sc < 1: img = cv2.resize(img, (int(w*sc), int(h*sc)), interpolation=cv2.INTER_AREA)
    return img

def isnet_mask(fp):
    """Good-quality alpha at SEG_RES, cached."""
    from rembg import remove
    rel = os.path.relpath(fp, IMG)
    mp = os.path.join(MASK2, os.path.splitext(rel)[0] + ".png")
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    img = load_big(fp)
    if img is None: return None, None
    if os.path.exists(mp):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape[:2] != img.shape[:2]:
                m = cv2.resize(m, (img.shape[1], img.shape[0]))
            return img, m
    out = remove(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), session=isnet())
    m = out[..., 3]
    cv2.imwrite(mp, m)
    return img, m

# ---------------------------------------------------------------- shape score
def _bell(v, mu, sig):
    return float(np.exp(-0.5*((v-mu)/sig)**2))

def fishiness(img, m, verbose=False):
    """How much does this mask look like one lateral fish? 0 = not at all.

    Returns (score, dict-of-parts). Hard-zero for the failure modes that
    wrecked the first gallery."""
    if m is None or img is None: return 0.0, {}
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]))
    binm = (m > 128).astype(np.uint8)
    H, W = binm.shape
    atot = int(binm.sum())
    if atot < 0.01*H*W: return 0.0, {"why": "empty"}

    ncc, lbl, st, _ = cv2.connectedComponentsWithStats(binm, 8)
    if ncc <= 1: return 0.0, {"why": "nocc"}
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    a1 = int(st[k, cv2.CC_STAT_AREA])
    x, y = st[k, cv2.CC_STAT_LEFT], st[k, cv2.CC_STAT_TOP]
    w, h = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
    cc = (lbl == k).astype(np.uint8)

    frac    = a1/(H*W)                       # how much of the frame it eats
    purity  = a1/atot                        # single blob, not a scattered mess
    extent  = a1/max(w*h, 1)                 # fill of its own bbox
    aspect  = w/max(h, 1)

    cnts, _ = cv2.findContours(cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    solidity = a1/hull if hull > 0 else 0.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    rectness = a1/max(rw*rh, 1)              # ~1.0 => it IS a rectangle

    # border contact: a whole-frame / tank-wall mask hugs the edges
    b = 2
    touch = [cc[:b, :].mean(), cc[-b:, :].mean(), cc[:, :b].mean(), cc[:, -b:].mean()]
    nborder = sum(t > 0.30 for t in touch)   # borders it runs along
    border_run = max(touch)

    # ---- hard rejections (the four ways the old pipeline failed) ----
    why = None
    if frac > 0.80:                       why = "eats frame"
    elif rectness > 0.90:                 why = "rectangular crop"
    elif nborder >= 3:                    why = "hugs 3+ borders"
    elif extent > 0.88:                   why = "boxy"
    elif purity < 0.40:                   why = "fragmented"
    elif aspect < 0.55 or aspect > 7.0:   why = "not lateral"
    elif solidity < 0.45:                 why = "sprawling (coral?)"
    if why:
        return 0.0, {"why": why, "frac": frac, "rect": rectness, "sol": solidity}

    # ---- soft shape preferences ----
    # Deliberately broad: angelfish and discus are taller than they are long, and a
    # long-finned betta is nowhere near convex. The hard gates above are what keep
    # coral and tank walls out; these only rank the survivors, so a narrow prior
    # here just throws away good, oddly-shaped fish.
    s_aspect = max(_bell(min(aspect, 5.0), 1.9, 1.25), 0.25)
    s_extent = _bell(extent, 0.60, 0.20)              # fins keep it off 1.0
    s_solid  = max(_bell(min(solidity, 0.99), 0.84, 0.18), 0.25)
    s_size   = _bell(np.clip(frac, 0, .8), 0.30, 0.20)  # a close-up, not a speck
    s_border = 1.0 - 0.55*min(border_run/0.5, 1.0) - 0.20*(nborder == 2)
    s_pure   = purity**2

    # a hand holding the fish: penalise skin inside the blob
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Cr, Cb, Yc = ycc[..., 1], ycc[..., 2], ycc[..., 0]
    ccb = cc.astype(bool)
    skin = ccb & (Cr >= 135) & (Cr <= 178) & (Cb >= 85) & (Cb <= 132) & (Yc > 80)
    skin_frac = skin.sum()/max(a1, 1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = (hsv[..., 1][ccb]/255.0).mean()
    val = (hsv[..., 2][ccb]/255.0).mean()

    # crisp mask = confident segmentation (soft halo edges mean it guessed)
    edge = cv2.morphologyEx(cc, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    conf = float(np.abs(m[ccb].astype(np.float32) - 128).mean()/127.0) if ccb.any() else 0

    score = (s_aspect * s_extent * s_solid * s_size * s_border * s_pure
             * (1 - 0.85*skin_frac)
             * (0.55 + 0.75*sat) * (0.6 + 0.5*min(val*1.4, 1.0))
             * (0.5 + 0.5*conf))
    parts = dict(frac=frac, purity=purity, extent=extent, aspect=aspect,
                 solidity=solidity, rectness=rectness, nborder=nborder,
                 skin=skin_frac, sat=sat, score=score)
    return float(score), parts

# ------------------------------------------------------------------ cut it out
def cutout(img, m, pad=6):
    binm = (m > 128).astype(np.uint8)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    ncc, lbl, st, _ = cv2.connectedComponentsWithStats(binm, 8)
    if ncc <= 1: return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    binm = (lbl == k).astype(np.uint8)

    alpha = (binm*255).astype(np.uint8)
    alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8), 1)   # trim the halo fringe
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)               # feather
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0: return None
    x0, x1 = max(xs.min()-pad, 0), min(xs.max()+pad, img.shape[1]-1)
    y0, y1 = max(ys.min()-pad, 0), min(ys.max()+pad, img.shape[0]-1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.dstack([rgb, alpha])[y0:y1+1, x0:x1+1]

def prefilter(sci, n=SHORTLIST):
    """Cheap pass over the cached u2netp masks to shortlist candidate photos."""
    d = os.path.join(IMG, sci.replace(" ", "_"))
    out = []
    for fp in sorted(glob.glob(os.path.join(d, "*.jpg"))):
        m = mask_for(fp); img = load_resized(fp)
        if m is None or img is None: continue
        s, _ = fishiness(img, m)
        if s > 0: out.append((s, fp))
    out.sort(key=lambda t: -t[0])
    return [fp for _, fp in out[:n]]

def best_for(sci):
    """Shape ranks the candidates; the classifier then vetoes the ones that are
    shaped like a fish but aren't one (anemones, sponges, gorgonians, bait balls).
    Crucially this re-picks a *different photo* of the same species rather than
    dropping the species — there are up to 80 to choose from."""
    from fishnet import fishness as netfish
    cands = prefilter(sci)
    if not cands:   # prefilter rejected everything -> let isnet see a sample anyway
        d = os.path.join(IMG, sci.replace(" ", "_"))
        cands = sorted(glob.glob(os.path.join(d, "*.jpg")))[:SHORTLIST]
    best, bs, bp = None, -1, {}
    for fp in cands:
        img, m = isnet_mask(fp)
        if m is None: continue
        s, parts = fishiness(img, m)
        if s <= 0: continue
        rgba = cutout(img, m)
        if rgba is None: continue
        _, pfish, _ = netfish(rgba)
        total = s * (0.15 + 1.2*pfish)
        parts["shape"], parts["p_fish"] = round(s, 3), round(pfish, 3)
        if total > bs: best, bs, bp = (img, m, fp), total, parts
    return best, bs, bp

def main():
    only = set(sys.argv[1:]) or None
    rows, made, rejected = [], 0, []
    for i, sp in enumerate(SPECIES):
        sci = sp["scientific"]
        if only and sci not in only: continue
        out = os.path.join(OUT, sci.replace(" ", "_")+".png")
        best, s, parts = best_for(sci)
        if best is None or s < ACCEPT:
            rejected.append((sci, sp["common"], round(s, 3), parts.get("why", "low score")))
            print(f"[{i:3d}] REJECT {sp['common']:34s} score={s:.3f} {parts.get('why','')}", flush=True)
            if os.path.exists(out): os.remove(out)
            continue
        img, m, fp = best
        rgba = cutout(img, m)
        if rgba is None:
            rejected.append((sci, sp["common"], round(s, 3), "cutout failed")); continue
        cv2.imwrite(out, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        rows.append(dict(scientific=sci, common=sp["common"], score=round(s, 3),
                         src=os.path.basename(fp), **{k: round(float(v), 3) for k, v in parts.items() if k != "score"}))
        made += 1
        print(f"[{i:3d}] ok     {sp['common']:34s} score={s:.3f} asp={parts.get('aspect',0):.2f} sol={parts.get('solidity',0):.2f}", flush=True)
    import pandas as pd
    if rows: pd.DataFrame(rows).to_csv(os.path.join(ROOT, "cutout_quality.csv"), index=False)
    print(f"\n{made} cut-outs accepted, {len(rejected)} species rejected")
    for r in rejected: print("   reject:", r[1], r[2], r[3])

if __name__ == "__main__":
    main()
