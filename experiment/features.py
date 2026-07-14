"""Stage 2 (v2): extract striking-ness features using cached rembg masks and
close-up selection. Re-runnable without re-downloading or re-segmenting.

COLOUR    colorfulness, hue_entropy, saturation, sat_p90 (vivid patches),
          high_chroma_frac, n_color_clusters
PATTERN   contrast, edge_density
SHAPE     aspect_ratio, body_depth, solidity, silhouette_cplx, extent
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, cv2, pandas as pd
from fish_data import SPECIES
from segment import select_images

ROOT = os.path.dirname(os.path.abspath(__file__))

def colour_feats(img, fg):
    b,g,r = (img[...,0].astype(np.float64), img[...,1].astype(np.float64), img[...,2].astype(np.float64))
    rg, yb = r-g, 0.5*(r+g)-b
    rgf, ybf = rg[fg], yb[fg]
    colorfulness = float(np.sqrt(rgf.std()**2+ybf.std()**2) + 0.3*np.sqrt(rgf.mean()**2+ybf.mean()**2))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = hsv[...,0][fg].astype(np.float64); S = hsv[...,1][fg].astype(np.float64)/255.0
    satmask = S > 0.15
    if satmask.sum() > 30:
        hist,_ = np.histogram(H[satmask], bins=30, range=(0,180)); p = hist/hist.sum(); p = p[p>0]
        hue_entropy = float(-(p*np.log2(p)).sum())
    else:
        hue_entropy = 0.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1,3)[fg.reshape(-1)].astype(np.float32)
    n_clusters = 0
    if len(lab) > 50:
        samp = lab[np.random.default_rng(0).choice(len(lab), min(2000,len(lab)), replace=False)]
        crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(samp, 6, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        frac = np.bincount(labels.flatten(), minlength=6)/len(labels)
        keep, distinct = [i for i in range(6) if frac[i]>0.08], []
        for i in keep:
            if all(np.linalg.norm(centers[i]-centers[j])>18 for j in distinct): distinct.append(i)
        n_clusters = len(distinct)
    return dict(colorfulness=colorfulness, hue_entropy=hue_entropy,
                saturation=float(S.mean()), sat_p90=float(np.percentile(S,90)),
                high_chroma_frac=float((S>0.6).mean()), n_color_clusters=float(n_clusters))

def pattern_feats(img, fg):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = float(gray[fg].astype(np.float64).std())
    edges = cv2.Canny(gray, 60, 160)
    return dict(contrast=contrast, edge_density=float((edges[fg]>0).mean()))

def shape_feats(fg):
    d = dict(aspect_ratio=np.nan, body_depth=np.nan, solidity=np.nan,
             silhouette_cplx=np.nan, extent=np.nan)
    m = fg.astype(np.uint8)
    ncc, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if ncc <= 1: return d
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnts,_ = cv2.findContours((lbl==biggest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return d
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 400: return d
    peri = cv2.arcLength(c, True)
    harea = cv2.contourArea(cv2.convexHull(c))
    x,y,w,h = cv2.boundingRect(c)
    d["solidity"] = float(area/harea) if harea>0 else np.nan
    d["extent"]   = float(area/(w*h)) if w*h>0 else np.nan
    d["silhouette_cplx"] = float(peri*peri/(4*np.pi*area))
    d["body_depth"] = float(h/w) if w>0 else np.nan
    if len(c) >= 5:
        (_,_),(MA,ma),_ = cv2.fitEllipse(c)
        major,minor = max(MA,ma),min(MA,ma)
        d["aspect_ratio"] = float(major/minor) if minor>0 else np.nan
    return d

def process_species(sci, want=25):
    rows = []
    for img, fg in select_images(sci, want=want):
        try:
            rows.append({**colour_feats(img,fg), **pattern_feats(img,fg), **shape_feats(fg)})
        except Exception:
            continue
    return rows

def _worker(sp):
    return sp["scientific"], process_species(sp["scientific"])

def main():
    from multiprocessing import Pool, cpu_count
    per_image, agg = [], []
    results = {}
    with Pool(max(1, cpu_count()-2)) as pool:
        for sci, rows in pool.imap_unordered(_worker, SPECIES):
            results[sci] = rows
            print(f"  done {sci}: {len(rows)} usable")
    for sp in SPECIES:
        rows = results.get(sp["scientific"], [])
        for r in rows: per_image.append({"scientific": sp["scientific"], **r})
        if not rows: continue
        med = pd.DataFrame(rows).median(numeric_only=True).to_dict()
        agg.append({**sp, "n_images": len(rows), **med})
    pd.DataFrame(per_image).to_csv(os.path.join(ROOT,"metrics_per_image.csv"), index=False)
    pd.DataFrame(agg).to_csv(os.path.join(ROOT,"fish_metrics.csv"), index=False)
    print(f"\nWrote fish_metrics.csv ({len(agg)} species)")

if __name__ == "__main__":
    main()
