"""Download sex-annotated (male / female) photo sets per species into
data/images_male/ and data/images_female/. Cached."""
import os, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from inat import photo_urls_sex, UA
from fish_data import SPECIES

ROOT = os.path.dirname(os.path.abspath(__file__))
TIDS = json.load(open(os.path.join(ROOT,"data","taxon_ids.json")))
MAXP = 40

def slug(s): return s.replace(" ", "_")

def dl(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1000: return True
    try:
        r = requests.get(url, headers=UA, timeout=25); r.raise_for_status()
        open(path,"wb").write(r.content); return True
    except Exception:
        return False

def fetch(sex_name, sex_val):
    outdir = os.path.join(ROOT, "data", f"images_{sex_name}")
    urldir = os.path.join(ROOT, "data", f"urls_{sex_name}")
    os.makedirs(outdir, exist_ok=True); os.makedirs(urldir, exist_ok=True)
    for i, sp in enumerate(SPECIES):
        sci = sp["scientific"]; tid = TIDS.get(sci)
        if not tid: continue
        uc = os.path.join(urldir, slug(sci)+".json")
        if os.path.exists(uc):
            urls = json.load(open(uc))
        else:
            urls = photo_urls_sex(tid, sex_val, MAXP)
            json.dump(urls, open(uc,"w")); time.sleep(0.4)
        if not urls: continue
        d = os.path.join(outdir, slug(sci)); os.makedirs(d, exist_ok=True)
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(dl, u, os.path.join(d, f"{j:03d}.jpg")) for j,u in enumerate(urls)]
            [f.result() for f in as_completed(futs)]
        n = len([f for f in os.listdir(d) if f.endswith(".jpg")])
        if n: print(f"  [{sex_name}] {sp['common']:28s} {n}")

if __name__ == "__main__":
    print("=== MALE ==="); fetch("male", 10)
    print("=== FEMALE ==="); fetch("female", 11)
    print("done")
