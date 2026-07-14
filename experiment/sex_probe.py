"""Resolve taxon ids (cached) and count male/female sex-annotated research-grade
photos per species. Decides which species can support the male-only and
dichromatism analyses."""
import json, os, time, requests
from fish_data import SPECIES
from inat import resolve_taxon_id, UA, API

ROOT = os.path.dirname(os.path.abspath(__file__))
TIDS = os.path.join(ROOT, "data", "taxon_ids.json")
tids = json.load(open(TIDS)) if os.path.exists(TIDS) else {}

def obs_count(tid, sex_val):
    r = requests.get(f"{API}/observations", headers=UA, timeout=20, params={
        "taxon_id": tid, "photos": "true", "quality_grade": "research",
        "term_id": 9, "term_value_id": sex_val, "per_page": 1})
    return r.json().get("total_results", 0)

rows = []
for i, sp in enumerate(SPECIES):
    sci = sp["scientific"]
    if sci not in tids:
        try:
            tids[sci] = resolve_taxon_id(sci)
        except Exception:
            tids[sci] = None
        time.sleep(0.3)
    tid = tids[sci]
    m = f = 0
    if tid:
        try:
            m, f = obs_count(tid, 10), obs_count(tid, 11)
            time.sleep(0.3)
        except Exception:
            pass
    rows.append((sci, sp["common"], m, f))
    if (i+1) % 20 == 0:
        print(f"  ...{i+1}/{len(SPECIES)}")

json.dump(tids, open(TIDS, "w"))
qual = [r for r in rows if r[2] >= 5 and r[3] >= 5]
print(f"\nspecies with >=5 male AND >=5 female annotated: {len(qual)}/{len(SPECIES)}")
qual.sort(key=lambda r: -(r[2]+r[3]))
for sci, common, m, f in qual:
    print(f"  {common:28s} M={m:4d} F={f:4d}")
