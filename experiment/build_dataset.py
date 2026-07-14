"""Stage 1: resolve taxa, collect photo URLs, and download images to disk.
Everything is CACHED so re-runs are cheap and feature extraction never
re-downloads. Image CDN downloads are concurrent (CDN is not rate-limited)."""
import os, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from inat import resolve_taxon_id, photo_urls, UA
from fish_data import SPECIES

MAX_PHOTOS = 80
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
IMG  = os.path.join(DATA, "images")
URLS = os.path.join(DATA, "urls")
os.makedirs(IMG, exist_ok=True)
os.makedirs(URLS, exist_ok=True)

def slug(sci):
    return sci.replace(" ", "_")

def get_urls(sp):
    cache = os.path.join(URLS, slug(sp["scientific"]) + ".json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    tid = resolve_taxon_id(sp["scientific"])
    urls = photo_urls(tid, MAX_PHOTOS) if tid else []
    with open(cache, "w") as f:
        json.dump(urls, f)
    return urls

def download_one(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return True
    try:
        r = requests.get(url, headers=UA, timeout=25)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        return False

def main():
    total_imgs = 0
    for i, sp in enumerate(SPECIES):
        sci = sp["scientific"]
        d = os.path.join(IMG, slug(sci))
        os.makedirs(d, exist_ok=True)
        urls = get_urls(sp)
        tasks = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            for j, u in enumerate(urls):
                ext = ".jpg"
                path = os.path.join(d, f"{j:03d}{ext}")
                tasks.append(ex.submit(download_one, u, path))
            ok = sum(1 for t in as_completed(tasks) if t.result())
        n = len([f for f in os.listdir(d) if f.endswith(".jpg")])
        total_imgs += n
        print(f"[{i+1}/{len(SPECIES)}] {sp['common']:30s} urls={len(urls):3d} on_disk={n:3d}")
    print(f"\nTotal images on disk: {total_imgs}")

if __name__ == "__main__":
    main()
