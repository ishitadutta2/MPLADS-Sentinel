"""
Optional real-world visual classification using CLIP
(openai/clip-vit-base-patch32 via `transformers`) — the exact upgrade path
already documented in vision_geo.py's module docstring and in the README's
"Production upgrade path" section.

Why this is a SEPARATE check from vision_geo.py's RandomForestClassifier:
that classifier is trained only on this demo's synthetic, procedurally
rendered photos (see data/generate_photos.py — flat colour gradients,
simple geometric motifs). It has never seen a real photograph, so its
score on a genuinely real uploaded photo tells you "does this look like
our synthetic renders", not "does this look like a real toilet". Training
it on real photos isn't something this demo can just do — there's no
labeled real-world MPLADS site-photo dataset sitting around to train on.

CLIP sidesteps that: it's pretrained on ~400M real image-text pairs from
the open web, so it already has a working notion of what a real public
toilet, school building, or handpump looks like. This module does CLIP's
standard *zero-shot* classification — no training on our corpus at all —
by embedding the uploaded photo and a set of natural-language category
descriptions, then comparing them by cosine similarity. That's the
textbook-correct way to get open-vocabulary "does this image match this
description" judgments without labeled training data.

Requires `torch` + `transformers` (intentionally NOT in requirements.txt —
see README "Production upgrade path") and ~600MB of model weights fetched
from huggingface.co on first use. Every entry point here is wrapped so the
app degrades gracefully (is_available() == False, with a human-readable
reason) if those aren't installed or the weights can't be downloaded,
rather than crashing anything that imports this module.

Honesty note for whoever reads this next: in the sandbox this file was
originally written in, `torch` and `transformers` installed fine (pure
Python packages, no restricted network needed), but downloading the actual
model weights failed — the sandbox's network egress is allow-listed to a
fixed set of package-registry domains and huggingface.co isn't one of
them. So the "model loads and classify() returns real scores" path in this
file is implemented to the same standard as everything else in this repo,
but could not be exercised end-to-end there. It's exactly the kind of
thing to double check on first real run.
"""
from __future__ import annotations

import numpy as np

_MODEL = None
_PROCESSOR = None
_LOAD_ERROR: str | None = None
_ATTEMPTED = False

MODEL_NAME = "openai/clip-vit-base-patch32"

# Natural-language prompts, one per work category — CLIP compares the
# uploaded photo's embedding against these, not against our synthetic
# corpus. Written as plain descriptions of what the real thing looks like.
CATEGORY_PROMPTS = {
    "Road Construction": "a photograph of a road being built or paved",
    "Community Hall": "a photograph of a community hall or public meeting building",
    "Borewell / Handpump": "a photograph of a borewell or a hand-operated water pump",
    "Street Lighting": "a photograph of a street light or lamp post",
    "School Building Repair": "a photograph of a school building under repair or construction",
    "Drainage System": "a photograph of a drain, sewer line, or drainage construction",
    "Drinking Water Supply": "a photograph of a water storage tank or drinking-water supply infrastructure",
    "Public Toilet Complex": "a photograph of a public toilet block under construction",
    "Sports Infrastructure": "a photograph of a sports field, court, or stadium facility",
    "Library / Reading Room": "a photograph of a library or public reading room building",
}


def is_available() -> bool:
    """True only once the model has actually loaded successfully. Trigger
    the (lazy, one-time) load attempt by calling classify() or
    ensure_loaded() first."""
    return _MODEL is not None


def unavailable_reason() -> str | None:
    """None if never attempted yet, or if it succeeded; a human-readable
    error string if a load was attempted and failed."""
    return _LOAD_ERROR


def ensure_loaded() -> bool:
    """Triggers the load attempt (if not already made this process) and
    returns is_available()."""
    _try_load()
    return is_available()


def _try_load():
    global _MODEL, _PROCESSOR, _LOAD_ERROR, _ATTEMPTED
    if _ATTEMPTED:
        return
    _ATTEMPTED = True
    try:
        import torch  # noqa: F401  (imported for its side effect of confirming availability)
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as e:
        _LOAD_ERROR = (
            "torch/transformers not installed. This is an optional dependency "
            f"(see requirements.txt 'Production upgrade path') — install with "
            f"`pip install torch transformers` to enable. ({e})"
        )
        return
    try:
        _MODEL = CLIPModel.from_pretrained(MODEL_NAME)
        _PROCESSOR = CLIPProcessor.from_pretrained(MODEL_NAME)
        _MODEL.eval()
    except Exception as e:  # noqa: BLE001 - deliberately broad: any failure here should degrade, not crash
        _MODEL, _PROCESSOR = None, None
        _LOAD_ERROR = (
            f"Model weights ({MODEL_NAME}, ~600MB) could not be downloaded from huggingface.co "
            f"on first use — check internet access and try again. ({type(e).__name__}: {e})"
        )


def classify(img_rgb: np.ndarray) -> dict[str, float] | None:
    """Zero-shot classifies img_rgb (HxWx3, RGB, uint8) against every
    category in CATEGORY_PROMPTS. Returns {category: probability}
    (softmax over CLIP cosine similarities, sums to 1.0), or None if CLIP
    isn't available — call unavailable_reason() to find out why."""
    _try_load()
    if _MODEL is None:
        return None

    import torch
    from PIL import Image

    pil_img = Image.fromarray(img_rgb)
    labels = list(CATEGORY_PROMPTS.keys())
    prompts = [CATEGORY_PROMPTS[c] for c in labels]

    inputs = _PROCESSOR(text=prompts, images=pil_img, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = _MODEL(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()

    return dict(zip(labels, probs))
