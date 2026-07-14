"""Is this cut-out actually a fish, or is it an anemone / sponge / sea urchin /
diver's glove that happens to be fish-shaped?

Shape heuristics cannot answer that: a brain coral and a pufferfish are both
compact convex blobs. So ask an ImageNet classifier, which knows both the fish
classes and — usefully — the exact reef bric-a-brac that keeps sneaking into
the gallery. We only use it comparatively: does this look more like ImageNet's
fish than like ImageNet's coral?
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

# ImageNet-1k indices
FISH = [0, 1, 2, 3, 4, 5, 6,            # tench, goldfish, sharks, rays
        389, 390, 391, 392, 393, 394, 395, 396, 397]  # barracouta..puffer
NOT  = [107, 108, 109, 115, 116, 327, 328, 973,  # jellyfish, anemone, brain coral,
                                                 # sea slug, chiton, starfish,
                                                 # sea urchin, coral reef
        983, 801,                                # scuba diver, snorkel
        326, 314]                                # lycaenid/ant - texture junk

_M = None; _T = None
def model():
    global _M, _T
    if _M is None:
        w = ResNet50_Weights.IMAGENET1K_V2
        _M = resnet50(weights=w).eval()
        _T = w.transforms()
        torch.set_grad_enabled(False)
    return _M, _T

def fishness(rgba, bg=0.5):
    """RGBA cut-out (uint8 or float) -> 0..1. High = ImageNet is confident this
    is a fish rather than reef furniture. Composited onto flat grey so the
    classifier sees the silhouette we actually plot."""
    m, t = model()
    a = rgba[..., 3:4].astype(np.float32)/255.0 if rgba.dtype == np.uint8 else rgba[..., 3:4]
    rgb = rgba[..., :3].astype(np.float32)/255.0 if rgba.dtype == np.uint8 else rgba[..., :3]
    comp = rgb*a + bg*(1-a)
    x = torch.from_numpy(comp).permute(2, 0, 1).float()
    x = t(torch.clamp(x, 0, 1)).unsqueeze(0)
    p = F.softmax(m(x), dim=1)[0].numpy()
    pf, pn = float(p[FISH].sum()), float(p[NOT].sum())
    return pf/(pf + pn + 1e-6), pf, pn

if __name__ == "__main__":
    import sys, glob, os, cv2
    for fp in sys.argv[1:]:
        im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
        r, pf, pn = fishness(im)
        print(f"{os.path.basename(fp)[:34]:36s} fishness={r:.3f}  p_fish={pf:.3f} p_reef={pn:.3f}")
