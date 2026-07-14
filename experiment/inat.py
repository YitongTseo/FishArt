"""iNaturalist helpers: resolve a species to a taxon_id and collect many
research-grade photo URLs from the observations endpoint (not the small
curated taxon_photos set)."""
import time, requests

UA = {"User-Agent": "fish-art-experiment/2.0 (research; yitong.tseo@gmail.com)"}
API = "https://api.inaturalist.org/v1"
TIMEOUT = 20

def resolve_taxon_id(sciname):
    r = requests.get(f"{API}/taxa", params={"q": sciname, "rank": "species", "per_page": 8},
                     headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    results = r.json().get("results", [])
    for res in results:                      # prefer exact name match
        if res.get("name", "").lower() == sciname.lower():
            return res["id"]
    return results[0]["id"] if results else None

def photo_urls_sex(taxon_id, sex_value, max_photos=40, size="medium"):
    """Photo URLs restricted to a sex annotation (Sex term_id=9; Male=10, Female=11)."""
    return _photo_urls(taxon_id, max_photos, size, extra={"term_id": 9, "term_value_id": sex_value})

def photo_urls(taxon_id, max_photos=80, size="medium"):
    return _photo_urls(taxon_id, max_photos, size)

def _photo_urls(taxon_id, max_photos, size, extra=None):
    """Page through research-grade observations, returning up to max_photos
    distinct photo URLs (ordered by community votes = better photos first)."""
    urls, page = [], 1
    while len(urls) < max_photos and page <= 5:
        params = {"taxon_id": taxon_id, "photos": "true", "quality_grade": "research",
                  "per_page": 200, "page": page, "order_by": "votes", "order": "desc"}
        if extra:
            params.update(extra)
        try:
            r = requests.get(f"{API}/observations", headers=UA, timeout=TIMEOUT, params=params)
            r.raise_for_status()
            obs = r.json().get("results", [])
        except Exception as e:
            print(f"    obs page {page} failed: {e}")
            break
        if not obs:
            break
        for o in obs:
            for ph in o.get("photos", []):
                u = ph.get("url")
                if u:
                    urls.append(u.replace("square", size))
                    if len(urls) >= max_photos:
                        break
            if len(urls) >= max_photos:
                break
        page += 1
        time.sleep(1.0)                      # be polite to the API
    # dedupe, keep order
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out
