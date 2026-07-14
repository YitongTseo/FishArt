# experiment/

All code, data and figures live here. **The dataset, methods and pipeline are documented in
the [top-level README](../README.md)** — that is the canonical description; this file only
exists so the directory is not silently undocumented.

Quick pointers:

- `FINDINGS.md` — the full results, iteration by iteration, including the negative ones and
  the place the effect does not hold. Read this before citing the headline number.
- `BRAINSTORM_Yaxes.md` — every striking-ness measure considered, and why.
- `requirements.txt` — pinned dependencies (note: `torchvision` must match `torch`).
- `archive/` — superseded iteration-1/2 scripts, figures and cut-outs.

Re-run the gallery figure without re-downloading the 1.1 GB image cache:

```bash
python fig4_fish_gallery.py     # data/cutouts/ is committed
```
