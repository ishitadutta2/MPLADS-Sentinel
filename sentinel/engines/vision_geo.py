"""
Geo & visual-consistency engine.

Two real, independently-verifiable checks on the evidence photos
contractors upload as proof of work:

1. GEO CHECK — pure math, no ML needed: does the photo's GPS EXIF/metadata
   fall within a sane radius of the project's declared site coordinates?
   (Haversine distance.) This alone catches a large share of real-world
   MPLADS photo fraud, where a contractor uploads a picture taken
   somewhere else entirely.

2. VISUAL-CONSISTENCY CHECK — two complementary real techniques:
     a. Perceptual hashing (via a DCT-based average-hash implemented on
        top of OpenCV/NumPy) + strict pixel-level confirmation, to catch
        *duplicate or near-duplicate* images reused across different
        projects/claims — this is a fingerprinting/matching task, where
        a hash-based approach is the textbook-correct tool (not a weaker
        substitute for a learned model — this is what production
        duplicate-detection systems actually use).
     b. A genuinely TRAINED multi-class image classifier
        (`sklearn.ensemble.RandomForestClassifier` on HOG + colour-
        histogram + edge-density features) that learns what each work
        category actually looks like from the photo corpus itself,
        rather than comparing against a fixed hand-built prototype. The
        classifier is evaluated with stratified k-fold cross-validation
        so every "visual consistency" score is an out-of-fold prediction
        — the model never scores a photo it was trained on. Validation
        metrics (accuracy, F1, confusion matrix) are computed by
        `evaluate_vision_classifier()` and asserted in the test suite.

  --- Upgrade path (documented, not run in this offline sandbox) ---
  Swap `_extract_features()` for a CLIP (openai/clip-vit-base-patch32 via
  `transformers`) embedding + fine-tuned linear head. It's a drop-in
  replacement behind the same function signature — see README "Production
  upgrade path".
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sentinel.config import GEO_TOLERANCE_METERS, PHOTO_DIR
from sentinel.utils.geo import haversine_meters

_HOG = cv2.HOGDescriptor(_winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8),
                          _cellSize=(8, 8), _nbins=9)


# --------------------------------------------------------------------------
# Perceptual hash (average hash over DCT-reduced grayscale image)
# --------------------------------------------------------------------------
def _phash(img_gray: np.ndarray, hash_size: int = 8) -> int:
    resized = cv2.resize(img_gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    bits = (dct_low > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------
# Feature extraction for the TRAINED category classifier
# --------------------------------------------------------------------------
def _extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """Colour histogram + Canny edge density + HOG shape descriptor — a
    classic, well-understood hand-engineered feature set fed into a real
    learned classifier below, rather than compared by a fixed formula."""
    hist = cv2.calcHist([img_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.array([edges.mean() / 255.0])

    gray_64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    hog_feat = _HOG.compute(gray_64).flatten()

    return np.concatenate([hist, edge_density, hog_feat])


def _build_feature_matrix(photos_df: pd.DataFrame):
    """Extracts features for every photo that has a readable file on
    disk. Returns (feature_matrix, labels, photo_ids) aligned by row."""
    features, labels, photo_ids = [], [], []
    for _, row in photos_df.iterrows():
        fpath = PHOTO_DIR / row["filename"]
        if not fpath.exists():
            continue
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        features.append(_extract_features(img))
        labels.append(row["category"])
        photo_ids.append(row["photo_id"])
    return np.array(features), np.array(labels), photo_ids


def _train_and_score_visual_consistency(photos_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42):
    """Trains a RandomForestClassifier to predict a photo's declared
    category from its visual features, using stratified k-fold cross-
    validation so every photo is scored by a model that never saw it
    during training (no leakage). Returns a dict {photo_id: consistency}
    where consistency is the out-of-fold predicted probability the model
    assigns to the photo's OWN declared category — low for a photo that
    doesn't visually look like what it claims to be."""
    X, y, photo_ids = _build_feature_matrix(photos_df)
    if len(X) < n_splits * 2:
        return {}

    clf = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=random_state, n_jobs=-1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
    classes = np.unique(y)
    class_index = {c: i for i, c in enumerate(classes)}

    consistency = {}
    for i, pid in enumerate(photo_ids):
        true_class_idx = class_index[y[i]]
        consistency[pid] = float(proba[i, true_class_idx])
    return consistency


def evaluate_vision_classifier(photos_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> dict:
    """Standalone validation report for the trained category classifier —
    cross-validated accuracy, weighted F1, and confusion matrix. This is
    what the test suite asserts against and what the README's ML metrics
    table is generated from; it is not used in the scoring pipeline
    itself (which only needs the per-photo probabilities above)."""
    X, y, photo_ids = _build_feature_matrix(photos_df)
    clf = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=random_state, n_jobs=-1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(clf, X, y, cv=skf)
    return {
        "n_photos": len(y),
        "n_categories": len(np.unique(y)),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "f1_weighted": round(f1_score(y, y_pred, average="weighted"), 4),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=np.unique(y)).tolist(),
        "class_labels": np.unique(y).tolist(),
    }



def _build_hash_corpus(photos_df: pd.DataFrame):
    """Perceptual hash + small grayscale thumbnail for every readable photo
    on disk. Shared by the main engine (duplicate detection across the
    corpus) and by score_new_photo() (checking one new upload against the
    corpus) so the two paths can't drift out of sync."""
    hashes, thumbnails = {}, {}
    THUMB_SIZE = (64, 64)
    for _, row in photos_df.iterrows():
        fpath = PHOTO_DIR / row["filename"]
        if not fpath.exists():
            continue
        img_gray = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        hashes[row["photo_id"]] = _phash(img_gray)
        thumbnails[row["photo_id"]] = cv2.resize(img_gray, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    return hashes, thumbnails


def _is_pixel_near_identical(thumb_a, thumb_b, max_mean_abs_diff=2.0) -> bool:
    if thumb_a is None or thumb_b is None:
        return False
    return float(np.mean(np.abs(thumb_a - thumb_b))) <= max_mean_abs_diff


def build_vision_corpus(photos_df: pd.DataFrame) -> dict:
    """Builds everything needed to score a brand-new (e.g. user-uploaded)
    photo against the existing corpus, once: a classifier fit on ALL
    existing photos (a plain single fit is correct and honest here —
    unlike the main engine's cross-validated scoring of corpus photos
    against themselves, a genuinely new photo was never part of training,
    so there's no leakage to guard against), plus the phash/thumbnail
    corpus for duplicate matching. Expensive (same cost as the main
    engine's feature extraction) — callers should cache the result."""
    X, y, photo_ids = _build_feature_matrix(photos_df)
    clf = None
    if len(X) > 0:
        clf = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
        clf.fit(X, y)
    hashes, thumbnails = _build_hash_corpus(photos_df)
    proj_lookup = photos_df.set_index("photo_id")["project_id"].to_dict()
    return {"clf": clf, "hashes": hashes, "thumbnails": thumbnails, "project_lookup": proj_lookup}


def score_new_photo(img_bgr: np.ndarray, declared_category: str, corpus: dict,
                     project_lat=None, project_lon=None, photo_lat=None, photo_lon=None) -> dict:
    """Scores one new (not-in-corpus) photo the same way the main engine
    scores existing ones: visual-consistency probability for its declared
    category from the trained classifier, a duplicate check against every
    photo already in the system, and — if coordinates were supplied — a
    geo-distance check. Coordinates are optional and taken as given rather
    than read from image EXIF: this demo's own sample photos don't carry
    real GPS EXIF (their geotag is a visible pixel stamp, not metadata),
    so pretending to auto-extract it from an upload would be dishonest."""
    result = {
        "visual_consistency": None, "consistency_fail": None,
        "is_duplicate": False, "duplicate_of_project": None, "duplicate_of_photo": None,
        "geo_distance_m": None, "geo_fail": None,
    }

    clf = corpus.get("clf")
    if clf is not None and declared_category in getattr(clf, "classes_", []):
        feats = _extract_features(img_bgr).reshape(1, -1)
        proba = clf.predict_proba(feats)[0]
        class_index = {c: i for i, c in enumerate(clf.classes_)}
        consistency = float(proba[class_index[declared_category]])
        result["visual_consistency"] = consistency
        result["consistency_fail"] = consistency < 0.35

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h = _phash(gray)
    thumb = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    for pid, corpus_hash in corpus.get("hashes", {}).items():
        if _hamming(h, corpus_hash) <= 4 and _is_pixel_near_identical(thumb, corpus.get("thumbnails", {}).get(pid)):
            result["is_duplicate"] = True
            result["duplicate_of_photo"] = pid
            result["duplicate_of_project"] = corpus.get("project_lookup", {}).get(pid)
            break

    if project_lat is not None and project_lon is not None and photo_lat is not None and photo_lon is not None:
        dist = haversine_meters(project_lat, project_lon, photo_lat, photo_lon)
        result["geo_distance_m"] = round(dist, 1)
        result["geo_fail"] = dist > GEO_TOLERANCE_METERS

    return result


def run_vision_geo_engine(projects_df: pd.DataFrame, photos_df: pd.DataFrame) -> pd.DataFrame:
    # ---- 1. GEO CHECK (per photo, rolled up to per project) ----
    photos = photos_df.merge(
        projects_df[["project_id", "site_latitude", "site_longitude"]],
        on="project_id", how="left",
    )
    photos["geo_distance_m"] = photos.apply(
        lambda r: haversine_meters(r.site_latitude, r.site_longitude, r.latitude, r.longitude),
        axis=1,
    )
    photos["geo_fail"] = photos["geo_distance_m"] > GEO_TOLERANCE_METERS

    # ---- 2. PERCEPTUAL HASH candidate generation + PIXEL-LEVEL confirmation ----
    # Rationale: average/DCT perceptual hashing deliberately ignores
    # high-frequency detail so it's robust to recompression — but that
    # means visually-similar-but-genuinely-different photos of the same
    # work category (same composition, different instance) can collide
    # within a loose Hamming threshold, producing false positives. A
    # *reused* photo is not just perceptually similar, it is the same
    # image re-saved — so we use phash purely to generate cheap
    # candidate pairs, then confirm each candidate with a strict
    # pixel-level near-identity check before calling it a duplicate.
    hashes, thumbnails = _build_hash_corpus(photos)

    def _is_pixel_near_identical(pid_a, pid_b, max_mean_abs_diff=2.0) -> bool:
        """Confirms a phash candidate is a true reused/duplicate image by
        checking mean absolute pixel difference on a small grayscale
        thumbnail — near-zero for a genuinely reused file, materially
        higher for two independently generated photos of the same
        category."""
        a, b = thumbnails.get(pid_a), thumbnails.get(pid_b)
        if a is None or b is None:
            return False
        return float(np.mean(np.abs(a - b))) <= max_mean_abs_diff

    # Group near-identical hashes (Hamming distance <= 4) across DIFFERENT
    # projects as CANDIDATES — genuine duplicates within the *same*
    # project's multiple angles are normal and not flagged.
    photo_ids = list(hashes.keys())
    dup_flags = {pid: False for pid in photo_ids}
    dup_partner = {pid: None for pid in photo_ids}
    proj_lookup = photos.set_index("photo_id")["project_id"].to_dict()

    # Bucket by coarse hash prefix to keep this roughly O(n log n) instead
    # of full O(n^2) on a couple thousand photos.
    from collections import defaultdict
    buckets = defaultdict(list)
    for pid, h in hashes.items():
        buckets[h >> 48].append(pid)  # top 16 bits as bucket key

    for bucket_ids in buckets.values():
        for i in range(len(bucket_ids)):
            for j in range(i + 1, len(bucket_ids)):
                a, b = bucket_ids[i], bucket_ids[j]
                if proj_lookup[a] == proj_lookup[b]:
                    continue
                if _hamming(hashes[a], hashes[b]) <= 4 and _is_pixel_near_identical(a, b):
                    # Both projects are flagged, symmetrically. We
                    # deliberately do NOT try to infer which project is
                    # the "original" and which "reused" the photo —
                    # self-reported capture dates can't be trusted to
                    # establish that (a fraudulent claim can just as
                    # easily backdate itself), so a real investigator
                    # would need to look at both claims by hand. Sentinel
                    # surfaces the pair; it doesn't pre-judge which side
                    # is at fault.
                    dup_flags[a] = True
                    dup_flags[b] = True
                    dup_partner[a] = proj_lookup[b]
                    dup_partner[b] = proj_lookup[a]

    photos["is_duplicate"] = photos["photo_id"].map(dup_flags).fillna(False)
    photos["duplicate_of_project"] = photos["photo_id"].map(dup_partner)

    # ---- 3. VISUAL CONSISTENCY — trained classifier, cross-validated ----
    consistency_scores = _train_and_score_visual_consistency(photos)
    photos["visual_consistency"] = photos["photo_id"].map(consistency_scores)
    photos["consistency_fail"] = photos["visual_consistency"] < 0.35

    # ---- Roll up to project level ----
    def rollup(group):
        n = len(group)
        geo_fail_rate = group["geo_fail"].mean()
        dup_any = group["is_duplicate"].any()
        consistency_fail_rate = group["consistency_fail"].mean()
        max_geo_dist = group["geo_distance_m"].max()

        score = 0.0
        flags = []
        if geo_fail_rate > 0:
            score += min(geo_fail_rate * 60, 60)
            flags.append("geo_location_mismatch")
        if dup_any:
            score += 30
            partners = group.loc[group["is_duplicate"], "duplicate_of_project"].dropna().unique()
            # Symmetric, non-directional wording — we cannot reliably tell
            # which of the two claims is the "original" photo from
            # self-reported metadata alone (see comment above), so this
            # flags the pairing for manual review rather than accusing a
            # specific direction.
            flags.append(f"identical_photo_also_used_in:{','.join(partners)}" if len(partners) else "duplicate_photo")
        if consistency_fail_rate > 0.5:
            score += min(consistency_fail_rate * 25, 25)
            flags.append("photo_does_not_match_declared_work_type")

        return pd.Series({
            "geo_photo_score": round(min(score, 100), 1),
            "max_geo_distance_m": round(max_geo_dist, 1),
            "has_duplicate_photo": bool(dup_any),
            "avg_visual_consistency": round(group["visual_consistency"].dropna().mean(), 3) if group["visual_consistency"].notna().any() else None,
            "geo_vision_flags": flags,
        })

    result = photos.groupby("project_id").apply(rollup, include_groups=False)
    return result
