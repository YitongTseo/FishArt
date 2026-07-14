"""Generate a clean transparent PNG cutout per species from the best close-up
(largest, most lateral fish) using the cached rembg masks. Used as marks /
side-images in the artsy figures. Cached to data/cutouts/."""
import warnings; warnings.filterwarnings("ignore")
import os, glob
import numpy as np, cv2
from fish_data import SPECIES
from segment import mask_for, load_resized

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG  = os.path.join(ROOT, "data", "images")
OUT  = os.path.join(ROOT, "data", "cutouts")
os.makedirs(OUT, exist_ok=True)

def best_image(sci):
    """Pick the cleanest SINGLE-fish lateral close-up: one dominant, compact,
    solid blob (not two fish / fish+coral), reasonably large and lateral."""
    d = os.path.join(IMG, sci.replace(" ", "_"))
    best = None; best_score = -1
    for fp in sorted(glob.glob(os.path.join(d, "*.jpg"))):
        m = mask_for(fp)
        img = load_resized(fp)
        if m is None or img is None: continue
        if m.shape[:2] != img.shape[:2]: m = cv2.resize(m, (img.shape[1], img.shape[0]))
        binm = (m > 128).astype(np.uint8)
        atot = int(binm.sum())
        if atot < 1500 or binm.mean() > 0.9: continue
        ncc, lbl, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
        if ncc <= 1: continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        a1 = int(stats[k, cv2.CC_STAT_AREA]); w = stats[k, cv2.CC_STAT_WIDTH]; h = stats[k, cv2.CC_STAT_HEIGHT]
        purity = a1/atot                              # one dominant blob?
        cnts,_ = cv2.findContours((lbl==k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(cnts[0])) if cnts else a1
        solidity = a1/hull if hull>0 else 0.5         # compact silhouette (not fused mess)
        lateral = 1.0 + 0.3*((w/max(h,1)) > 1.2)
        size = a1/binm.size
        # penalise human skin (hands holding the fish) and reward saturated subjects
        ccpix = (lbl == k)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Cr, Cb, Yc = ycrcb[...,1], ycrcb[...,2], ycrcb[...,0]
        skin = ccpix & (Cr>=135)&(Cr<=178)&(Cb>=85)&(Cb<=132)&(Yc>80)
        skin_frac = skin.sum()/max(a1,1)
        sat_mean = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[...,1][ccpix]/255.0).mean()
        score = (size * (purity**2) * solidity * lateral
                 * (1 - 0.75*skin_frac) * (0.55 + 0.9*sat_mean))
        if score > best_score:
            best_score = score; best = (img, m)
    return best

def cutout(img, m, pad=8):
    binm = (m > 128).astype(np.uint8)
    # keep only the largest connected component (one clean fish, no 2nd fish/coral)
    ncc, lbl, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
    if ncc > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binm = (lbl == biggest).astype(np.uint8)
    alpha = binm*255
    alpha = cv2.erode(alpha, np.ones((3,3), np.uint8), iterations=1)  # trim fringe
    alpha = cv2.GaussianBlur(alpha, (5,5), 0)                          # feather
    ys, xs = np.where(alpha > 20)
    if len(xs) == 0: return None
    x0,x1 = max(xs.min()-pad,0), min(xs.max()+pad, img.shape[1]-1)
    y0,y1 = max(ys.min()-pad,0), min(ys.max()+pad, img.shape[0]-1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, alpha])[y0:y1+1, x0:x1+1]
    return rgba

def main():
    n = 0
    for sp in SPECIES:
        sci = sp["scientific"]
        out = os.path.join(OUT, sci.replace(" ", "_")+".png")
        if os.path.exists(out):
            n += 1; continue
        bi = best_image(sci)
        if bi is None: continue
        rgba = cutout(*bi)
        if rgba is None: continue
        cv2.imwrite(out, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        n += 1
        print(f"  cutout {sp['common']}")
    print(f"\n{n} cutouts in data/cutouts/")

if __name__ == "__main__":
    main()
