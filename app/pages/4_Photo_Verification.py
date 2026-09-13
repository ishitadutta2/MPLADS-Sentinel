"""
Photo Verification — a dedicated screen for reviewing the vision/geo
engine's output: geo-mismatches, duplicate/reused photos, and
visual-category-mismatch flags, with the actual images shown side by
side so a reviewer can eyeball the evidence directly.
"""
import datetime
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from common import (
    get_scored_dataset, get_raw_tables, inject_base_style, require_login, show_user_badge,
    get_vision_geo_results, get_vision_corpus, page_header,
)
from sentinel.config import PHOTO_DIR, UPLOADED_PHOTO_DIR
from sentinel.engines.vision_geo import score_new_photo
from sentinel.engines import real_world_vision as rwv
from sentinel.engines import reverse_image_search as ris
from sentinel.utils.geo import extract_gps_from_exif
from sentinel.db import insert_uploaded_evidence, read_uploaded_evidence, delete_uploaded_evidence
from sentinel.audit.hash_chain import append_event
from i18n import t

st.set_page_config(page_title="Photo Verification — MPLADS Sentinel", page_icon=":material/photo_camera:", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()
page_header("camera", t("photo_title"), t("photo_sub"))

df = get_scored_dataset()
raw = get_raw_tables()
photos = raw["photos"]

with st.spinner(t("vision_geo_spinner")):
    photo_geo = get_vision_geo_results(df, photos)

flagged_projects = photo_geo[
    (photo_geo["has_duplicate_photo"]) | (photo_geo["max_geo_distance_m"] > 500)
    | (photo_geo["geo_photo_score"] > 20)
].sort_values("geo_photo_score", ascending=False)

tab1, tab2, tab3 = st.tabs([t("tab_flagged_evidence"), t("tab_lookup_project"), t("tab_upload_check")])

with tab1:
    st.metric(t("kpi_photo_geo_issues"), len(flagged_projects))
    for project_id, prow in flagged_projects.head(15).iterrows():
        meta = df[df["project_id"] == project_id]
        if meta.empty:
            continue
        meta = meta.iloc[0]
        with st.expander(f"{project_id} — {meta.category} — {t('kpi_geo_photo_score')} {prow.geo_photo_score:.0f}"):
            st.markdown(f"**{t('district')}:** {meta.district}, {meta.state}  |  **{t('col_contractor')}:** {meta.contractor_name}")
            for flag in prow.geo_vision_flags:
                st.warning(flag.replace("_", " "))

            proj_photos = photos[photos["project_id"] == project_id]
            cols = st.columns(min(max(len(proj_photos), 1), 4))
            for i, (_, ph) in enumerate(proj_photos.iterrows()):
                fpath = PHOTO_DIR / ph["filename"]
                with cols[i % len(cols)]:
                    if fpath.exists():
                        st.image(str(fpath), width='stretch', caption=ph["photo_id"])

with tab2:
    query_id = st.text_input(t("project_id_lbl"), key="lookup2")
    if query_id:
        pid = query_id.strip().upper()
        if pid in photo_geo.index:
            prow = photo_geo.loc[pid]
            st.metric(t("kpi_geo_photo_score"), f"{prow.geo_photo_score:.0f}")
            st.metric(t("kpi_max_distance"), f"{prow.max_geo_distance_m:,.0f} m")
            st.metric(t("kpi_duplicate_detected"), t("yes") if prow.has_duplicate_photo else t("no"))
            proj_photos = photos[photos["project_id"] == pid]
            cols = st.columns(min(max(len(proj_photos), 1), 4))
            for i, (_, ph) in enumerate(proj_photos.iterrows()):
                fpath = PHOTO_DIR / ph["filename"]
                with cols[i % len(cols)]:
                    if fpath.exists():
                        st.image(str(fpath), width='stretch', caption=ph["photo_id"])
        else:
            st.error(t("project_not_found"))

with tab3:
    st.caption(t("upload_check_caption"))
    lookup_id = st.text_input(t("upload_lookup_lbl"), key="upload_lookup").strip().upper()
    matches = df[df["project_id"] == lookup_id] if lookup_id else pd.DataFrame()

    if lookup_id and matches.empty:
        st.error(t("no_project_found", id=lookup_id))
    elif lookup_id:
        meta = matches.iloc[0]
        st.markdown(f"**{t('declared_category_site')}:** {meta.category}  |  **{t('site_lbl')}:** {meta.district}, {meta.state}")

        uploaded = st.file_uploader(t("upload_site_photo"), type=["jpg", "jpeg", "png"], key="evidence_uploader")

        if uploaded is not None and st.button(t("check_save_photo"), type="primary"):
            raw_bytes = uploaded.getvalue()
            file_bytes = np.frombuffer(raw_bytes, np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error(t("couldnt_read_image"))
            else:
                # Real GPS EXIF — not a manually-typed number. See
                # extract_gps_from_exif()'s docstring for why: a
                # self-reported coordinate isn't a location check, since a
                # fraudulent upload could just type in the right answer.
                # If this photo has no GPS EXIF (common for anything
                # downloaded, forwarded, or screenshotted rather than shot
                # fresh with location services on), we say so plainly
                # rather than pretending to verify something we can't.
                pil_img = Image.open(io.BytesIO(raw_bytes))
                photo_lat, photo_lon = extract_gps_from_exif(pil_img)

                with st.spinner(t("scoring_photo_spinner")):
                    corpus = get_vision_corpus()
                    res = score_new_photo(
                        img_bgr, meta.category, corpus,
                        project_lat=meta.site_latitude, project_lon=meta.site_longitude,
                        photo_lat=photo_lat, photo_lon=photo_lon,
                    )

                # Persist: save the actual file + a DB row, so this upload
                # is real evidence from now on, not a one-off in-memory
                # check that vanishes when the page reruns.
                UPLOADED_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
                upload_id = f"UPL{int(time.time() * 1000)}"
                ext = Path(uploaded.name).suffix.lower() or ".jpg"
                dest_path = UPLOADED_PHOTO_DIR / f"{upload_id}{ext}"
                dest_path.write_bytes(raw_bytes)
                now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                insert_uploaded_evidence({
                    "upload_id": upload_id, "project_id": lookup_id, "declared_category": meta.category,
                    "filename": dest_path.name, "uploaded_by": user["display_name"], "uploaded_at": now_iso,
                    "visual_consistency": res["visual_consistency"], "consistency_fail": res["consistency_fail"],
                    "is_duplicate": res["is_duplicate"], "duplicate_of_project": res["duplicate_of_project"],
                    "duplicate_of_photo": res["duplicate_of_photo"],
                    "geo_source": "exif" if photo_lat is not None else "none",
                    "geo_lat": photo_lat, "geo_lon": photo_lon,
                    "geo_distance_m": res["geo_distance_m"], "geo_fail": res["geo_fail"],
                })
                append_event(
                    event_type="EVIDENCE_UPLOAD", project_id=lookup_id, actor=user["display_name"],
                    details={"upload_id": upload_id, "declared_category": meta.category,
                              "is_duplicate": res["is_duplicate"], "geo_source": "exif" if photo_lat is not None else "none"},
                )

                cimg, cres = st.columns([1, 1.3])
                with cimg:
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width='stretch', caption=t("your_upload_saved"))
                with cres:
                    if res["visual_consistency"] is not None:
                        st.metric(t("match_declared_category"),
                                  f"{res['visual_consistency'] * 100:.0f}%")
                        if res["consistency_fail"]:
                            st.warning(t("doesnt_look_like", category=meta.category))
                        else:
                            st.success(t("visually_consistent"))
                    else:
                        st.info(t("category_not_covered"))

                    if res["is_duplicate"]:
                        st.error(
                            f"This photo is pixel-identical to one already on file — {res['duplicate_of_photo']}, "
                            f"used for project {res['duplicate_of_project']}."
                        )
                    else:
                        st.success(t("no_matching_photo"))

                    if photo_lat is not None:
                        st.metric(t("distance_from_site_metric"), f"{res['geo_distance_m']:,.0f} m")
                        if res["geo_fail"]:
                            st.warning(t("farther_than_tolerance"))
                        else:
                            st.success(t("within_expected_distance"))
                    else:
                        st.info(
                            "This photo has no GPS location embedded in it, so its location can't be verified "
                            "automatically. Real smartphone photos carry this automatically when location "
                            "services are on at the moment of capture — it's lost when an image is "
                            "downloaded, forwarded (e.g. via WhatsApp), or screenshotted instead of shot fresh. "
                            "This is exactly why real field-verification apps (like the eSAKSHI process this "
                            "app is modelled on) require evidence photos to be captured through the app's own "
                            "in-app camera at submission time, rather than uploaded from a gallery — it makes "
                            "this workaround impossible instead of just detecting it after the fact. "
                            "(This explainer is currently only available in English.)"
                        )

                st.divider()
                st.markdown(f"**{t('clip_section_header')}**")
                if rwv.ensure_loaded():
                    clip_scores = rwv.classify(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    if clip_scores:
                        declared_pct = clip_scores.get(meta.category, 0.0) * 100
                        top_cat = max(clip_scores, key=clip_scores.get)
                        st.metric(f"CLIP match to '{meta.category}'", f"{declared_pct:.0f}%")
                        if top_cat != meta.category:
                            st.warning(
                                f"A general-purpose pretrained vision model's best guess for what this photo "
                                f"actually shows is **{top_cat}** ({clip_scores[top_cat] * 100:.0f}%), not the "
                                f"declared **{meta.category}**. Unlike the score above, this model was never "
                                f"trained on this demo's data at all — it's checking against real-world visual "
                                f"knowledge from ~400M web images."
                            )
                        else:
                            st.success("A general-purpose pretrained vision model also agrees with the declared category.")
                else:
                    st.info(
                        f"Real-world check unavailable in this environment: {rwv.unavailable_reason() or 'not yet attempted.'} "
                        "This is an optional upgrade (see README 'Production upgrade path') — the score above, "
                        "from a classifier trained only on this demo's synthetic photos, is what's active without it."
                    )

                st.markdown(f"**{t('reverse_image_header')}**")
                web_result = ris.search_web_for_image(raw_bytes)
                if not web_result["configured"]:
                    st.info(
                        f"{web_result['reason']} This checks whether a photo was copied from Google Images "
                        "or a stock site rather than actually shot on location — a different, complementary "
                        "check from the duplicate-detection above, which only catches reuse *within this "
                        "system's own* photo corpus."
                    )
                elif not web_result["ok"]:
                    st.warning(f"Reverse image search request failed: {web_result['reason']}")
                else:
                    if web_result["full_matches"]:
                        st.error(
                            f"This photo (or an exact copy of it) was found on {len(web_result['full_matches'])} "
                            f"page(s) elsewhere on the web — it may not be an original site photo."
                        )
                        for url in web_result["full_matches"][:5]:
                            st.caption(url)
                    elif web_result["partial_matches"] or web_result["pages_with_matches"]:
                        st.warning("Visually similar (not identical) images were found elsewhere on the web.")
                        for url in web_result["pages_with_matches"][:5]:
                            st.caption(url)
                    else:
                        st.success("No matches found elsewhere on the open web.")

        st.divider()
        st.subheader(t("evidence_for_project"))
        existing = read_uploaded_evidence(lookup_id)
        if existing.empty:
            st.caption(t("no_evidence_yet"))
        else:
            for _, erow in existing.iterrows():
                with st.container(border=True):
                    ec1, ec2 = st.columns([1, 2.4])
                    with ec1:
                        epath = UPLOADED_PHOTO_DIR / erow["filename"]
                        if epath.exists():
                            st.image(str(epath), width='stretch')
                    with ec2:
                        uploaded_at_display = str(erow["uploaded_at"])[:19].replace("T", " ")
                        st.markdown(f"Uploaded by **{erow['uploaded_by']}** on {uploaded_at_display}")
                        if pd.notna(erow["visual_consistency"]):
                            st.caption(f"Category match: {erow['visual_consistency'] * 100:.0f}%")
                        if erow["is_duplicate"]:
                            st.error(f"Duplicate of {erow['duplicate_of_photo']} (project {erow['duplicate_of_project']})")
                        if erow["geo_source"] == "exif":
                            mismatch = " :material/warning: farther than tolerance" if erow["geo_fail"] else " :material/check: within tolerance"
                            st.caption(f"GPS: {erow['geo_lat']:.5f}, {erow['geo_lon']:.5f} — {erow['geo_distance_m']:.0f} m from site{mismatch}")
                        else:
                            st.caption(t("no_gps_metadata"))
                        if st.button(t("delete_evidence_btn"), key=f"del_{erow['upload_id']}"):
                            delete_uploaded_evidence(erow["upload_id"])
                            if epath.exists():
                                epath.unlink()
                            append_event(
                                event_type="EVIDENCE_DELETE", project_id=lookup_id, actor=user["display_name"],
                                details={"upload_id": erow["upload_id"]},
                            )
                            st.rerun()
