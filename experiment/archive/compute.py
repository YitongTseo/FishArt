"""
Fetch several photos per species from iNaturalist, segment the fish from
the background (GrabCut), and compute striking-ness metrics on the foreground.

Metrics (all on segmented foreground pixels):
  - colorfulness : Hasler & Susstrunk (2003) perceptual colorfulness
  - hue_entropy  : Shannon entropy of hue histogram (color VARIETY)
  - saturation   : mean HSV saturation
  - contrast     : std of grayscale luminance (bold pattern / markings)

Per species we take the MEDIAN across images to suppress lighting/crop noise.
Writes fish_metrics.csv.
"""
import io, time, json, sys
import numpy as np
import cv2
import requests
from fish_data import SPECIES

UA = {"User-Agent": "fish-art-experiment/1.0 (research; contact yitong.tseo@gmail.com)"}
N_IMAGES = 6            # target photos per species
MIN_IMAGES = 1          # keep even rare species (flagged via n_images)
TIMEOUT = 15

def inat_photo_urls(sciname, n=N_IMAGES):
    """Return up to n medium-size photo URLs for a species from iNaturalist."""
    try:
        r = requests.get("https://api.inaturalist.org/v1/taxa",
                         params={"q": sciname, "rank": "species", "per_page": 5},
                         headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as e:
        print(f"    taxa lookup failed: {e}")
        return []
    # pick the result whose name matches best
    taxon = None
    for res in results:
        if res.get("name", "").lower() == sciname.lower():
            taxon = res; break
    if taxon is None and results:
        taxon = results[0]
    if taxon is None:
        return []
    # the search endpoint truncates taxon_photos; fetch full taxon by id
    try:
        r2 = requests.get(f"https://api.inaturalist.org/v1/taxa/{taxon['id']}",
                          headers=UA, timeout=TIMEOUT)
        r2.raise_for_status()
        full = r2.json().get("results", [])
        if full:
            taxon = full[0]
    except Exception as e:
        print(f"    taxon detail failed: {e}")
    urls = []
    for tp in taxon.get("taxon_photos", []):
        photo = tp.get("photo", {})
        u = photo.get("medium_url") or photo.get("url")
        if u:
            urls.append(u.replace("square", "medium"))
        if len(urls) >= n:
            break
    # fall back to default photo
    if not urls and taxon.get("default_photo"):
        dp = taxon["default_photo"]
        u = dp.get("medium_url") or dp.get("url")
        if u: urls.append(u.replace("square", "medium"))
    return urls

def download_image(url):
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        return img
    except Exception as e:
        return None

def segment_foreground(img):
    """GrabCut with a central-rectangle init. Returns boolean foreground mask."""
    h, w = img.shape[:2]
    # resize for speed/consistency
    scale = 400.0 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    m = int(0.10 * min(h, w))          # margin: assume subject is centered
    rect = (m, m, w - 2*m, h - 2*m)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), True, False)
    except Exception:
        fg = np.zeros((h, w), bool)
    # fallback: if segmentation collapsed, use center 60% crop
    if fg.sum() < 0.05 * h * w:
        fg = np.zeros((h, w), bool)
        fg[int(0.2*h):int(0.8*h), int(0.2*w):int(0.8*w)] = True
    return img, fg

def metrics(img, fg):
    """Compute striking-ness metrics on foreground pixels of a BGR image."""
    b, g, r = img[..., 0].astype(np.float64), img[..., 1].astype(np.float64), img[..., 2].astype(np.float64)
    rg = r - g
    yb = 0.5 * (r + g) - b
    rg_f, yb_f = rg[fg], yb[fg]
    std_rg, mean_rg = rg_f.std(), rg_f.mean()
    std_yb, mean_yb = yb_f.std(), yb_f.mean()
    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0][fg].astype(np.float64)      # 0-179
    S = hsv[..., 1][fg].astype(np.float64) / 255.0
    # hue entropy weighted toward saturated pixels (grey pixels have meaningless hue)
    sat_mask = S > 0.15
    if sat_mask.sum() > 20:
        hist, _ = np.histogram(H[sat_mask], bins=30, range=(0, 180))
        p = hist / hist.sum()
        p = p[p > 0]
        hue_entropy = float(-(p * np.log2(p)).sum())
    else:
        hue_entropy = 0.0
    saturation = float(S.mean())

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)[fg]
    contrast = float(gray.std())
    return dict(colorfulness=float(colorfulness), hue_entropy=hue_entropy,
                saturation=saturation, contrast=contrast)

def main():
    rows = []
    for i, (sci, common, family, ms, mci, hab, note) in enumerate(SPECIES):
        print(f"[{i+1}/{len(SPECIES)}] {common} ({sci})")
        urls = inat_photo_urls(sci)
        print(f"    {len(urls)} photo urls")
        vals = []
        for u in urls:
            img = download_image(u)
            if img is None:
                continue
            img2, fg = segment_foreground(img)
            try:
                vals.append(metrics(img2, fg))
            except Exception as e:
                print(f"    metric fail: {e}")
        if len(vals) < MIN_IMAGES:
            print(f"    SKIP: only {len(vals)} usable images")
            continue
        agg = {k: float(np.median([v[k] for v in vals])) for k in vals[0]}
        rows.append(dict(scientific=sci, common=common, family=family,
                         mating_system=ms, mate_choice_index=mci, habitat=hab,
                         note=note, n_images=len(vals), **agg))
        print(f"    n={len(vals)}  colorfulness={agg['colorfulness']:.1f} "
              f"hue_entropy={agg['hue_entropy']:.2f} contrast={agg['contrast']:.1f}")
        time.sleep(0.5)  # be polite to iNat
    import pandas as pd
    df = pd.DataFrame(rows)
    out = "fish_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out} with {len(df)} species")

if __name__ == "__main__":
    main()
