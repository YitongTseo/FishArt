"""Male-only striking-ness and sexual DICHROMATISM analyses.

- Male-only: recompute striking-ness on MALE photos only (undiluted by drab
  females) and correlate with mate choice.
- Dichromatism: per species, distance between the male and female appearance
  (CIELAB DeltaE of mean colour, plus colorfulness/hue differences). Sexual-
  selection theory predicts stronger dichromatism where mate choice is stronger.
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, cv2, pandas as pd
from scipy import stats
from fish_data import SPECIES
from segment import select_from_dir
from features import colour_feats
try:
    from phylo import load_tree_and_vcv, pgls
    HAVE_PHYLO = True
except Exception:
    HAVE_PHYLO = False

ROOT = os.path.dirname(os.path.abspath(__file__))
def slug(s): return s.replace(" ", "_")

def sex_profile(sci, sex):
    idir = os.path.join(ROOT, "data", f"images_{sex}", slug(sci))
    mdir = os.path.join(ROOT, "data", f"masks_{sex}", slug(sci))
    if not os.path.isdir(idir): return None
    imgs = select_from_dir(idir, mdir, want=20)
    if len(imgs) < 5: return None
    rows, labs = [], []
    for img, fg in imgs:
        rows.append(colour_feats(img, fg))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
        labs.append([lab[...,0][fg].mean(), lab[...,1][fg].mean(), lab[...,2][fg].mean()])
    agg = pd.DataFrame(rows).median(numeric_only=True).to_dict()
    agg["meanLab"] = np.median(np.array(labs), axis=0)
    agg["n"] = len(imgs)
    return agg

def main():
    recs = []
    for sp in SPECIES:
        M = sex_profile(sp["scientific"], "male")
        F = sex_profile(sp["scientific"], "female")
        rec = {"scientific": sp["scientific"], "common": sp["common"],
               "family": sp["family"], "mate_choice_index": sp["mate_choice_index"]}
        if M:
            rec.update(male_colorfulness=M["colorfulness"], male_hue=M["hue_entropy"],
                       male_sat=M["saturation"], n_male=M["n"], _Lm=M["meanLab"])
        if F:
            rec.update(female_colorfulness=F["colorfulness"], female_hue=F["hue_entropy"],
                       n_female=F["n"], _Lf=F["meanLab"])
        if M and F:
            rec["dichromatism_dE"] = float(np.linalg.norm(M["meanLab"] - F["meanLab"]))
            rec["male_excess_colorf"] = float(M["colorfulness"] - F["colorfulness"])
            rec["male_excess_hue"] = float(M["hue_entropy"] - F["hue_entropy"])
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.drop(columns=[c for c in ["_Lm","_Lf"] if c in df], errors="ignore").to_csv(
        os.path.join(ROOT,"sexed_metrics.csv"), index=False)

    print("=== MALE-ONLY striking-ness vs mate choice ===")
    dm = df.dropna(subset=["male_hue"])
    dm = dm[dm["n_male"] >= 6]
    print(f"n = {len(dm)} species (>=6 male images)")
    for c in ["male_hue","male_colorfulness","male_sat"]:
        rho,p = stats.spearmanr(dm["mate_choice_index"], dm[c]); print(f"  {c:20s} rho={rho:+.3f} p={p:.3f}")

    print("\n=== SEXUAL DICHROMATISM vs mate choice ===")
    dd = df.dropna(subset=["dichromatism_dE"])
    print(f"n = {len(dd)} species (>=5 male AND >=5 female)")
    for c in ["dichromatism_dE","male_excess_colorf","male_excess_hue"]:
        rho,p = stats.spearmanr(dd["mate_choice_index"], dd[c]); print(f"  {c:20s} rho={rho:+.3f} p={p:.3f}")

    # PGLS on dichromatism if enough species land in the tree
    if HAVE_PHYLO and len(dd) >= 12:
        try:
            keep, C = load_tree_and_vcv(dd["scientific"].tolist())
            sub = dd[dd["scientific"].isin(keep)].set_index("scientific").loc[keep].reset_index()
            if len(sub) >= 10:
                r = pgls(sub["mate_choice_index"].values.astype(float),
                         sub["dichromatism_dE"].values.astype(float), C)
                print(f"\n  PGLS dichromatism (n={r['n']}, lambda={r['lam']:.2f}): "
                      f"slope={r['slope']:+.3f} t={r['t']:+.2f} p={r['p']:.3f}")
        except Exception as e:
            print("  PGLS dichromatism failed:", e)

    print(f"\nwrote sexed_metrics.csv")
    return df

if __name__ == "__main__":
    main()
