"""rembg (U^2-Net) foreground masks, cached to disk. Correctly isolates the
fish from background so colour/shape metrics are not contaminated. Because we
have up to 80 photos per species, we SELECT the close-up shots (fish fills more
of the frame) which give cleaner colour and shape than distant habitat shots."""
import warnings; warnings.filterwarnings("ignore")
import os, glob
import numpy as np, cv2
from rembg import remove, new_session

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG  = os.path.join(ROOT, "data", "images")
MASK = os.path.join(ROOT, "data", "masks")
os.makedirs(MASK, exist_ok=True)
_SESSION = None
RESIZE = 400

def session():
    global _SESSION
    if _SESSION is None:
        _SESSION = new_session("u2netp")     # light + fast (~0.12s/img on CPU)
    return _SESSION

def load_resized(fp):
    img = cv2.imread(fp)
    if img is None:
        return None
    h, w = img.shape[:2]; sc = RESIZE / max(h, w)
    if sc < 1:
        img = cv2.resize(img, (int(w*sc), int(h*sc)), interpolation=cv2.INTER_AREA)
    return img

def mask_for(fp):
    """Return cached (or freshly computed+cached) alpha mask uint8 for an image."""
    rel = os.path.relpath(fp, IMG)
    mp = os.path.join(MASK, os.path.splitext(rel)[0] + ".png")
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    if os.path.exists(mp):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return m
    img = load_resized(fp)
    if img is None:
        return None
    out = remove(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), session=session())
    alpha = out[..., 3]
    cv2.imwrite(mp, alpha)
    return alpha

def cache_species(sci, limit=80):
    """Pre-compute and cache masks for a species' images."""
    d = os.path.join(IMG, sci.replace(" ", "_"))
    files = sorted(glob.glob(os.path.join(d, "*.jpg")))[:limit]
    n = 0
    for fp in files:
        if mask_for(fp) is not None:
            n += 1
    return n

def select_images(sci, want=25, fg_lo=0.02, fg_hi=0.92, min_px=800):
    """Return list of (image_bgr, fg_bool) for the best close-up shots:
    fish present, not filling the whole frame, ranked by size (largest first)."""
    d = os.path.join(IMG, sci.replace(" ", "_"))
    files = sorted(glob.glob(os.path.join(d, "*.jpg")))
    cands = []
    for fp in files:
        m = mask_for(fp)
        if m is None:
            continue
        img = load_resized(fp)
        if img is None or img.shape[:2] != m.shape[:2]:
            # size mismatch (cache from different resize) -> recompute against img
            if img is None: continue
            m = cv2.resize(m, (img.shape[1], img.shape[0]))
        fg = m > 128
        frac = fg.mean()
        if fg.sum() < min_px or frac < fg_lo or frac > fg_hi:
            continue
        cands.append((frac, img, fg))
    cands.sort(key=lambda t: -t[0])          # biggest fish first
    return [(img, fg) for _, img, fg in cands[:want]]

def select_from_dir(image_dir, mask_dir, want=25, fg_lo=0.02, fg_hi=0.92, min_px=800):
    """Like select_images but for an arbitrary image dir with its own mask cache."""
    os.makedirs(mask_dir, exist_ok=True)
    cands = []
    for fp in sorted(glob.glob(os.path.join(image_dir, "*.jpg"))):
        mp = os.path.join(mask_dir, os.path.splitext(os.path.basename(fp))[0] + ".png")
        if os.path.exists(mp):
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            img = load_resized(fp)
            if m is None or img is None: continue
            if m.shape[:2] != img.shape[:2]: m = cv2.resize(m, (img.shape[1], img.shape[0]))
        else:
            img = load_resized(fp)
            if img is None: continue
            out = remove(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), session=session())
            m = out[..., 3]; cv2.imwrite(mp, m)
        fg = m > 128
        if fg.sum() < min_px or fg.mean() < fg_lo or fg.mean() > fg_hi: continue
        cands.append((fg.mean(), img, fg))
    cands.sort(key=lambda t: -t[0])
    return [(img, fg) for _, img, fg in cands[:want]]

if __name__ == "__main__":
    from fish_data import SPECIES
    for sp in SPECIES:
        n = cache_species(sp["scientific"])
        print(f"cached {n:3d} masks  {sp['common']}")
