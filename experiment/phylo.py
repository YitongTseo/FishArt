"""Phylogenetic Generalised Least Squares (PGLS) with ML-estimated Pagel's lambda.

Uses the Fish Tree of Life chronogram (Rabosky et al. 2018). Tests whether each
striking-ness measure predicts the mate-choice index AFTER accounting for shared
ancestry (Brownian-motion covariance among species).

For a tree, the phylogenetic covariance between species i,j is the shared path
length from the root to their MRCA:  C_ij = (r_i + r_j - d_ij)/2,  C_ii = r_i,
where r = root-to-tip distance and d = patristic distance. Pagel's lambda scales
the off-diagonals (lambda=1 Brownian, lambda=0 == ordinary regression).
"""
import os, numpy as np, pandas as pd, dendropy
from scipy import optimize, stats

ROOT = os.path.dirname(os.path.abspath(__file__))

def load_tree_and_vcv(species):
    tree = dendropy.Tree.get(path=os.path.join(ROOT,"data","fishtree.nwk"),
                             schema="newick", preserve_underscores=True)
    tips = {t.taxon.label for t in tree.leaf_node_iter()}
    keep = [s for s in species if s.replace(" ","_") in tips]     # exact matches only
    labels = [s.replace(" ","_") for s in keep]
    tree.retain_taxa_with_labels(labels)
    tree.encode_bipartitions()
    # root-to-tip distances
    r = {leaf.taxon.label: leaf.distance_from_root() for leaf in tree.leaf_node_iter()}
    pdm = tree.phylogenetic_distance_matrix()
    tax = {t.label: t for t in tree.taxon_namespace}
    n = len(labels)
    C = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            la, lb = labels[a], labels[b]
            if a == b:
                C[a,b] = r[la]
            else:
                d = pdm.patristic_distance(tax[la], tax[lb])
                C[a,b] = C[b,a] = 0.5*(r[la] + r[lb] - d)
    return keep, C

def _gls(X, y, V):
    Vinv = np.linalg.inv(V)
    XtVi = X.T @ Vinv
    beta = np.linalg.solve(XtVi @ X, XtVi @ y)
    resid = y - X @ beta
    return beta, resid, Vinv

def _neg_ll(lam, X, y, C):
    V = C.copy(); off = ~np.eye(len(C), dtype=bool)
    V[off] = lam * C[off]
    try:
        beta, resid, Vinv = _gls(X, y, V)
    except np.linalg.LinAlgError:
        return 1e10
    n = len(y)
    s2 = float(resid.T @ Vinv @ resid) / n
    sign, logdet = np.linalg.slogdet(V)
    return 0.5*(n*np.log(2*np.pi) + n*np.log(s2) + logdet + n)

def pgls(x, y, C):
    """Return dict with ML lambda, slope, se, t, p (df=n-2)."""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    res = optimize.minimize_scalar(_neg_ll, bounds=(0.0, 1.0), method="bounded",
                                   args=(X, y, C))
    lam = float(res.x)
    V = C.copy(); off = ~np.eye(n, dtype=bool); V[off] = lam*C[off]
    beta, resid, Vinv = _gls(X, y, V)
    s2 = float(resid.T @ Vinv @ resid) / (n-2)
    covb = s2 * np.linalg.inv(X.T @ Vinv @ X)
    se = np.sqrt(np.diag(covb))
    t = beta[1]/se[1]
    p = 2*stats.t.sf(abs(t), n-2)
    return dict(n=n, lam=lam, slope=float(beta[1]), se=float(se[1]),
                t=float(t), p=float(p))

def main():
    df = pd.read_csv(os.path.join(ROOT, os.environ.get("METRICS", "fish_metrics.csv")))
    minimg = int(os.environ.get("MINIMG", 1))       # MINIMG=15 -> the quality-filtered subset
    df = df[df.n_images >= minimg].reset_index(drop=True)
    keep, C = load_tree_and_vcv(df["scientific"].tolist())
    d = df[df["scientific"].isin(keep)].set_index("scientific").loc[keep].reset_index()
    x = d["mate_choice_index"].values.astype(float)
    print(f"PGLS on {len(d)} species placed in the fish phylogeny\n")
    print(f"{'measure':18s} {'lambda':>7s} {'slope':>9s} {'t':>7s} {'p':>8s}   {'OLS p':>8s}")
    for c in ["hue_entropy","colorfulness","saturation","aspect_ratio","body_depth",
              "contrast","n_color_clusters"]:
        y = d[c].values.astype(float)
        m = ~np.isnan(y)
        if m.sum() < len(y):
            xi, yi, Ci = x[m], y[m], C[np.ix_(m,m)]
        else:
            xi, yi, Ci = x, y, C
        r = pgls(xi, yi, Ci)
        ols_r, ols_p = stats.spearmanr(xi, yi)
        star = "*" if r["p"] < 0.05 else ("." if r["p"] < 0.10 else "")
        print(f"{c:18s} {r['lam']:7.2f} {r['slope']:+9.4f} {r['t']:+7.2f} {r['p']:8.3f}{star:2s} {ols_p:8.3f}")

if __name__ == "__main__":
    main()
