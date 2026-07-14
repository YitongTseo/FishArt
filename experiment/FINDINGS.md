# Fish-art hypothesis test — findings

## Iteration 6 (current): the gallery's cut-outs were a third wrong
No change to the statistics — this iteration fixed **presentation integrity**.

An audit of the 143 cut-outs behind `fig4_fish_gallery.png` found that **~30% were not
fish**: coral heads, anemones, gorgonians, sea urchins, dive gloves, petri dishes, aquarium
walls and rectangular crops of open water. The v1 cut-out picker ranked photos by mask
*size* and had no way to reject a failed mask, so when segmentation failed it failed
confidently. (The metrics in `fish_metrics.csv` are computed from median statistics over
~25 images per species and are far more robust to this than a single displayed cut-out —
but the figure was, literally, showing pictures of gloves.)

Fixed in `make_cutouts.py` + `fishnet.py` with a three-stage cascade — isnet-general-use at
768px, hard shape gates (rectangular / border-hugging / sprawling → rejected), and an
ImageNet classifier veto that re-picks a different photo of the same species when the blob
looks more like coral than like a fish. Result: **123 species in the gallery, ~95% clean**,
with per-pick scores in `cutout_quality.csv`.

**What did not get fixed, and why:** a few species (Pacific sardine, bigeye scad, harlequin
rasbora) still show fish-in-a-hand / bait-ball / in-tank photos. Widening the search from 12
to 40 candidate photos each did not help — their *entire* iNaturalist photo sets are fishery
and aquarium shots. That is a property of the source data, and worth remembering as a
reminder of what these photo sets actually are.

The gallery figure was also rebuilt: mating behaviour now colours the **background** as a
kernel-density wash with in-plot labels, rather than a halo behind each fish.

---

## Iteration 5: quality-filtered final figures
Restricted to the **126 species with ≥15 clean close-up images** (robust medians;
dropped 16 species resting on 1–14 images). Removing that noise *sharpened* the
result rather than weakening it:

| measure | species-level ρ | PGLS (phylogeny-corrected) |
|---|---|---|
| **colour variety (hue entropy)** | **+0.31, p<0.001** | **p=0.011 (survives)** |
| colorfulness (brightness) | +0.04 (n.s.) | p=0.14 (n.s.) |
| saturation | +0.03 (n.s.) | p=0.088 (weak) |
| body elongation | +0.18, p=0.04 | p=0.53 (artifact) |
| body depth | -0.22, p=0.01 | p=0.74 (artifact) |

**This is the cleanest statement of the result:** on quality data, colour
*variety* significantly increases with mate choice and survives phylogenetic
correction; raw brightness does not; the shape signals are confirmed phylogenetic
artifacts. It still weakens in the small, reef-fish-skewed sex-annotated subset
(see iteration 4) — the one place the effect does not hold. Final figures:
`fig1_colour_variety.png`, `fig2_all_measures.png`, `fig3_robustness.png`
(from `clean_figures.py`). Superseded scripts/figures moved to `archive/`.

---

## Iteration 4: phylogeny + sex-control + dichromatism
Three rigor upgrades — and together they largely DISSOLVE the earlier support.

### 1. PGLS (phylogenetic regression) — `phylo.py`
Fish Tree of Life chronogram (Rabosky 2018); 126/156 species placed; ML Pagel's
lambda; covariance = shared ancestry.
- **hue_entropy (colour variety): slope +0.048, p=0.064** (lambda=0.70) — the ONLY
  measure that survives phylogenetic correction, and only marginally.
- **aspect_ratio p=0.91, body_depth p=0.57** — the "shape striking-ness" signals
  are CONFIRMED phylogenetic artifacts; they vanish entirely.
- colorfulness p=0.16, saturation p=0.08 (weak), contrast p=0.18.

### 2. Male-only images — `fetch_sexed.py`, `sexed_analyze.py`
Only **23-30/156 species** carry enough iNaturalist sex annotations. On that
subset, male-only striking-ness correlates NEGATIVELY with mate choice
(colorfulness rho=-0.37 p=0.047; saturation rho=-0.42 p=0.022; hue rho=-0.11).
Controlled check on the SAME 30 species: mixed-sex is also negative (colorfulness
rho=-0.45), so this is a **biased-subset effect, not a sex-filtering effect** —
sex-annotated species are disproportionately brilliant mid-choice reef fish
(clownfish, anthias, angels) vs cryptic/patchy high-choice fish (seahorses,
stickleback, guppy). Sex-filtering nudged colorfulness toward the hypothesis
(-0.45 -> -0.37) but did not rescue it.

### 3. Sexual dichromatism — male vs female CIELAB distance
n=27 species: dichromatism DECREASES with mate choice (rho=-0.36, p=0.066;
PGLS p=0.13, n.s.). Partly because iconic high-choice species signal via cryptic
bodies + a small patch (stickleback red belly) or are camouflaged (seahorses),
which whole-body colour distance misses.

### Overall verdict
`fig_summary_forest.png` tells it: **the same test, computed many ways.** A
modest colour-VARIETY signal appears in broad samples (rho=+0.19, survives PGLS
at p=0.06) but is fragile — it flattens at family level and reverses in the
sex-controlled subset. Colour INTENSITY never supports it and goes strongly
negative under sex control. **The essay's poetic thesis is not robustly
supported quantitatively.** Sexual selection looks like ONE of several colour
drivers (crypsis, aposematism, species recognition, habitat light all compete),
and its footprint on whole-body striking-ness — if real — is weak and easily
swamped. The intuition fits iconic lineages (cichlids, wrasses, killifish) but
does not generalise into a clean quantitative law.

Files added this iteration: `phylo.py`, `sex_probe.py`, `fetch_sexed.py`,
`sexed_analyze.py`, `summary_figure.py`, `sexed_metrics.csv`,
`fig_summary_forest.png`, `data/fishtree.nwk`, `data/taxon_ids.json`.

---

## Iteration 3: 156 species, ML segmentation
- Species **63 -> 156** (43 families), full mate-choice range (30 low / 71 mid / 55 high).
- Segmentation: GrabCut **-> rembg (U^2-Net)**, masks cached to `data/masks/`.
  Fixed the contamination bug; validated (blue tang high, tuna/sardine low).
- Close-up selection: among up to 80 photos/species, use the shots where the
  fish fills most of the frame (cleaner colour + shape).
- 9,422 images, 142 species with enough usable shots.

### Result — a real but modest, colour-VARIETY effect
| measure | species-level rho (n=142) | family-level rho (n=42) |
|---|---|---|
| **hue_entropy (colour variety)** | **+0.19 (p=0.021)** * | +0.17 (p=0.28) |
| colorfulness (brightness) | +0.02 (n.s.) | +0.12 (n.s.) |
| saturation / sat_p90 / high_chroma | ~0 (n.s.) | — |
| aspect_ratio (slenderness) | +0.20 (p=0.018) * | -0.08 (n.s.) |
| body_depth | -0.21 (p=0.011) * | -0.02 (n.s.) |
| contrast (luminance) | -0.17 (p=0.039) * | — |

**Interpretation**
1. **Colour variety is the one measure that supports the essay** — mate-choosy
   species wear a greater *diversity* of hues. Significant at species level;
   stays positive but underpowered at family level (only 42 families). Replicated
   across 3 runs (clean n=34 p=0.06; rembg n=54 p=0.02; full n=142 p=0.02).
2. **Raw "prettiness" does NOT track mate choice.** Colorfulness, saturation and
   vivid-patch metrics are all ~zero — many low-choice fish are gaudy (tangs,
   tetras, snappers). The naive version of the hypothesis fails.
3. **The shape signals are phylogenetic artifacts.** Slenderness/body-depth look
   significant per-species but VANISH at family level — they just reflect that
   slender livebearer/killifish/wrasse lineages happen to be choosy.
4. **Luminance contrast runs negative** (countershading, not display) in every run.

**Bottom line:** the essay's poetic thesis gets *qualified, partial* support.
Sexual selection appears to diversify the palette (more colours), not simply
brighten it — and even that effect is modest and partly confounded by phylogeny.
A proper test needs phylogenetic comparative methods (PGLS) and sex-specific
images (see below).

### Key remaining threat to validity
iNaturalist photos mix **both sexes / juveniles / feral wild-types**. For
sexually dimorphic species the showy males (the whole point) are diluted by drab
females. E.g. Endler's livebearer still scores low colorfulness because most
photos are not brilliant males. This *systematically weakens* the very effect we
test. Fixes: sex-filtered image sets, sexual-dichromatism as its own axis, or a
vision-LLM striking-ness rating.

---

# Earlier: iteration 2 (63 species, GrabCut) — superseded

## What changed from iteration 1
- Species: 34 -> **63** (29 families), spanning the full mate-choice range.
- Mate-choice score is now **componentised** (`fish_data.py`): four behavioural
  sub-axes (fertilisation, courtship, mating system, who-chooses), 0-3 each,
  summed to 0-10. Auditable, behaviour-only (no colour -> no circularity).
- Photos: 6 -> **up to 80 per species** (3,844 images) from the iNaturalist
  *observations* endpoint, cached to `data/images/` (download paid once).
- Y-axes: colour + pattern + **body-shape** + **distinctiveness** (17 measures).

## Headline result: the earlier signal did NOT survive scaling
Across 59 species with 80 photos each, **every striking-ness measure correlates
near-zero with mate choice** (all |rho| < 0.19, none significant). The colour-
variety signal that looked promising in iteration 1 (rho=+0.32, p=0.06) fell to
+0.18 (n.s.).

## Why — a diagnostic, not just a null
Two things changed at once (more species AND messier photos). Isolating them:

| cut | hue_entropy (colour variety) | note |
|---|---|---|
| iter 1: same 34 sp, 6 CLEAN photos | **+0.32 (p=0.06)** | promising |
| iter 2: same 34 sp, 80 MESSY photos | +0.17 (p=0.35) | signal degraded |
| iter 2: all 59 sp, n_images>=20 | **+0.25 (p=0.08)** | recovers when noise filtered |

**Two real causes, both matter:**
1. **Genuine confound.** Many low-choice fish are brilliantly coloured for
   non-sexual reasons — blue/yellow tang (aggregation spawners), neon & cardinal
   tetra (egg-scatterers), damselfish. So raw colour genuinely does not separate
   on mate choice. This part is a true finding.
2. **Measurement noise now dominates.** Automated GrabCut segmentation fails on
   in-situ / small-fish photos. Smoking gun: **Endler's livebearer** (mate-choice
   10, one of the most vivid fish alive) scored the **lowest** colorfulness — the
   segmenter grabbed tank/background, not fish. More photos != better if the
   segmenter is the weak link.

## What holds up
- **Colour variety (hue entropy)** is the only measure that leans positive across
  every cut (+0.17 to +0.32). Fragile, quality-sensitive, marginal at best.
- **Luminance contrast is consistently negative** (rho ~ -0.13 to -0.37) — it
  tracks anti-predator countershading, not display. Not a striking-ness proxy.
- Shape, saturation, distinctiveness: no reliable relationship as measured.

## The bottleneck is now segmentation quality, not sample size.
Next decisive step: replace GrabCut with ML background removal (e.g. rembg/U2Net)
and re-run on the SAME cached images. That tells us whether the colour-variety
signal is real or noise-limited.

## Files
- `fish_data.py` — 63 species, componentised scores
- `inat.py`, `build_dataset.py` — fetch/cache up to 80 photos/species
- `features.py` — 17 striking-ness measures per image (parallel)
- `analyze.py` — Y-axes, distinctiveness, correlations, figures
- `fig_correlation_summary.png`, `fig_scatter_grid.png`
- `fish_metrics.csv`, `metrics_per_image.csv`, `fish_analysed.csv`
- `BRAINSTORM_Yaxes.md` — full menu of striking-ness measures considered
