# Brainstorm: ways to quantify "visual striking-ness" (the Y-axis)

Legend: [x] implemented in features.py/analyze.py  •  [ ] candidate for next round

## A. Colour intensity / richness
- [x] **Colorfulness** (Hasler-Süsstrunk) — perceptual "how colourful". Confounded:
      many drab-choice and bright-no-choice fish, so weak on its own.
- [x] **Saturation** (mean HSV S)
- [x] **High-chroma fraction** — % of "electric"/near-pure pixels
- [ ] **Chroma in CIELAB/CIECAM02** — perceptually uniform, better than HSV

## B. Colour variety  (the most theory-aligned so far)
- [x] **Hue entropy** — Shannon entropy of hue histogram. Only measure that leans
      positive across every cut. THE candidate to nail down with better segmentation.
- [x] **Number of distinct colour clusters** (k-means in Lab)
- [ ] **Colour "gamut area"** — area of the convex hull the fish occupies in a*b* plane
- [ ] **Palette rarity** — how unusual its colours are vs the whole fish sample

## C. Pattern / markings
- [x] **Luminance contrast** — bold stripes/spots BUT also countershading (runs
      the WRONG way — exclude or model separately).
- [x] **Edge density** (Canny) — busyness
- [ ] **Pattern periodicity** (FFT peaks) — regular stripes/spots vs noise
- [ ] **Pattern-type classifier** — uniform / striped / spotted / ocellated (eyespots)
- [ ] **Eyespot / ocellus detection** — a classic sexually-selected ornament

## D. Body shape / "body-type striking-ness"  (you asked for this)
- [x] **Aspect ratio** (elongation) — ambiguous: eels are elongated but not "striking"
- [x] **Body depth / length** — deep-bodied (angelfish, discus) reads as dramatic
- [x] **Solidity** (area/convex-hull) — flowing fins & filaments lower it
- [x] **Silhouette complexity** (perimeter²/area) — trailing fins, lyretails
- [x] **Extent** (area/bbox)
- [ ] **Fin-area ratio** — needs fin-vs-body segmentation (the real prize; sailfin
      mollies, bettas, lyretails, sailfin displays live here)
- [ ] **Fourier shape descriptors** — full outline signature, rotation-invariant
- [ ] **Deviation from a "canonical fish" template** — shape distinctiveness

## E. Distinctiveness / outlierness  (matches your original "aesthetically distinctive")
- [x] **kNN distance in colour space** — how far the species stands out on colour
- [x] **kNN distance in shape space**
- [x] **kNN distance overall**
- [ ] **Distance to family / clade centroid** — striking *relative to its relatives*
- [ ] **CLIP / deep-embedding distinctiveness** — embed each fish, measure how far
      it sits from the crowd in a learned perceptual space (needs a model)

## F. Perceptual / holistic  (gold standard, not automatable here)
- [ ] **Human striking-ness ratings** — crowd or a vision-LLM rating each species
      1-10. This is the ground truth every metric above is a proxy for.
- [ ] **Neural aesthetic score** (NIMA-style trained model)
- [ ] **Sexual DICHROMATISM** — male-vs-female colour difference. The classic
      signature of sexual selection. KEEP IT A SEPARATE AXIS (folding it into the
      mate-choice score would be circular), but it is arguably the single most
      predictive variable the theory offers.

## Recommended focus for next round
1. Fix segmentation (Section "measurement") so B and D are trustworthy.
2. Add **fin-area ratio** (D) and **sexual dichromatism** (F) — the two measures
   most directly tied to the "self-portrait via mate choice" thesis.
3. Consider a **vision-LLM striking-ness rating** as an independent ground truth.
