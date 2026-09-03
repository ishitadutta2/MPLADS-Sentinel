"""
Optional reverse image search: checks whether an uploaded photo appears
elsewhere on the open web (stock photo sites, other articles, previously
published images) — catching the "contractor grabbed a random photo off
Google Images instead of shooting the actual site" fraud pattern, which
is distinct from (and complements) vision_geo.py's duplicate check
against this system's OWN photo corpus.

This genuinely cannot be done for free or "simulated" honestly: real
reverse image search requires actually querying a real web-scale image
index, which means a real third-party API and a real API key someone
pays for. The standard, correct tool for exactly this job is Google Cloud
Vision's Web Detection feature (also what's behind Google Images' own
"search by image"). This module is a real, correct client for it — not a
mock — but it is INERT without an API key, and will say so clearly
rather than pretending to have checked anything.

Setup (production, not this demo sandbox):
  1. Create a Google Cloud project, enable the Cloud Vision API.
  2. Create an API key restricted to the Vision API.
  3. Set it as the GOOGLE_VISION_API_KEY environment variable (or add to
     .streamlit/secrets.toml as GOOGLE_VISION_API_KEY, which Streamlit
     also exposes as an env var automatically).
  4. Cloud Vision is a paid API past a small free tier — see
     https://cloud.google.com/vision/pricing before enabling this for
     any real volume of uploads.

Honesty note for whoever reads this next: vision.googleapis.com isn't
reachable from the sandbox this file was written in (network egress
there is allow-listed to package-registry domains only), so — exactly
like real_world_vision.py's CLIP integration — the "configured and
actually calling the API" path could not be exercised end-to-end before
shipping. The request/response shape follows Google's documented Vision
API v1 contract precisely, but treat it as unverified until tried against
a real key.
"""
from __future__ import annotations

import base64
import os

VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"


def is_configured() -> bool:
    return bool(_get_api_key())


def _get_api_key() -> str | None:
    key = os.environ.get("GOOGLE_VISION_API_KEY")
    if key:
        return key
    # Streamlit surfaces st.secrets values as env vars only if configured
    # that way by the deployer; fall back to checking st.secrets directly
    # if streamlit happens to be importable and secrets.toml exists, so
    # either setup method works without extra code from a deployer.
    try:
        import streamlit as st
        return st.secrets.get("GOOGLE_VISION_API_KEY")
    except Exception:
        return None


def search_web_for_image(image_bytes: bytes, timeout: int = 15) -> dict:
    """Sends image_bytes to Google Cloud Vision's Web Detection endpoint.

    Returns a dict:
      {"configured": False, "reason": "..."}                    — no API key set
      {"configured": True, "ok": False, "reason": "..."}         — request/API error
      {"configured": True, "ok": True,
       "full_matches": [...], "partial_matches": [...],
       "pages_with_matches": [...], "best_guess_labels": [...]}  — real result

    full_matches / partial_matches / pages_with_matches are lists of URLs;
    an uploaded photo with any full_matches is very likely lifted from the
    web rather than actually shot on site.
    """
    api_key = _get_api_key()
    if not api_key:
        return {
            "configured": False,
            "reason": (
                "GOOGLE_VISION_API_KEY is not set. Reverse image search requires a real "
                "Google Cloud Vision API key — see this module's docstring for setup steps."
            ),
        }

    import requests

    payload = {
        "requests": [
            {
                "image": {"content": base64.b64encode(image_bytes).decode("ascii")},
                "features": [{"type": "WEB_DETECTION", "maxResults": 10}],
            }
        ]
    }

    try:
        resp = requests.post(
            VISION_API_URL, params={"key": api_key}, json=payload, timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001 - any failure here should degrade cleanly, not crash the page
        return {"configured": True, "ok": False, "reason": f"{type(e).__name__}: {e}"}

    try:
        web_detection = data["responses"][0].get("webDetection", {})
    except (KeyError, IndexError):
        return {"configured": True, "ok": False, "reason": f"Unexpected API response shape: {data}"}

    return {
        "configured": True,
        "ok": True,
        "full_matches": [m.get("url") for m in web_detection.get("fullMatchingImages", []) if m.get("url")],
        "partial_matches": [m.get("url") for m in web_detection.get("partialMatchingImages", []) if m.get("url")],
        "pages_with_matches": [p.get("url") for p in web_detection.get("pagesWithMatchingImages", []) if p.get("url")],
        "best_guess_labels": [g.get("label") for g in web_detection.get("bestGuessLabels", []) if g.get("label")],
    }
