"""
Sanity/regression tests for every engine, run against the real generated
dataset. Uses stdlib `unittest` so it runs with zero extra dependencies;
also fully compatible with `pytest tests/` once pytest is installed.

Run: python -m unittest discover tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from sentinel.utils.geo import haversine_meters
from sentinel.engines.financial import run_financial_engine
from sentinel.engines.cost import run_cost_benchmark_engine
from sentinel.engines.vision_geo import run_vision_geo_engine, evaluate_vision_classifier
from sentinel.engines.contractor_graph import run_contractor_network_engine, build_contractor_graph
from sentinel.engines.document_intel import run_document_engine
from sentinel.engines.ml_risk_classifier import evaluate_ml_classifier, train_and_score
from sentinel.engines.document_ocr import evaluate_ocr_pipeline, extract_text, parse_sanction_order
from sentinel.engines.risk_engine import compute_composite_risk
from sentinel.services.esakshi_ingestion import (
    ingest_allocation_export, validate_allocation_export, IngestionValidationError,
)
from sentinel.audit.hash_chain import append_event, verify_chain, GENESIS_HASH
from sentinel.config import DATA_DIR, REPORT_AS_OF_DATE
from sentinel.auth import create_user, authenticate, scoped_query


class TestGeoUtils(unittest.TestCase):
    def test_zero_distance(self):
        self.assertAlmostEqual(haversine_meters(28.6, 77.2, 28.6, 77.2), 0.0, places=3)

    def test_known_distance_delhi_mumbai_roughly(self):
        # Delhi to Mumbai is ~1150km — sanity check order of magnitude
        d = haversine_meters(28.6139, 77.2090, 19.0760, 72.8777)
        self.assertGreater(d, 1_000_000)
        self.assertLess(d, 1_300_000)


class TestEnginesOnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projects = pd.read_csv(DATA_DIR / "projects.csv")
        cls.transactions = pd.read_csv(DATA_DIR / "transactions.csv")
        cls.photos = pd.read_csv(DATA_DIR / "photos.csv")
        cls.contractors = pd.read_csv(DATA_DIR / "contractors.csv")

    def test_financial_engine_scores_seeded_anomalies_higher(self):
        result = run_financial_engine(self.projects, self.transactions)
        merged = self.projects.set_index("project_id").join(result)
        anomalous_mean = merged.loc[merged.is_seeded_anomalous, "financial_score"].mean()
        clean_mean = merged.loc[~merged.is_seeded_anomalous, "financial_score"].mean()
        self.assertGreater(anomalous_mean, clean_mean)

    def test_cost_benchmark_scores_in_range(self):
        result = run_cost_benchmark_engine(self.projects)
        self.assertTrue((result["cost_benchmark_score"] >= 0).all())
        self.assertTrue((result["cost_benchmark_score"] <= 100).all())

    def test_vision_geo_catches_all_seeded_duplicates(self):
        result = run_vision_geo_engine(self.projects, self.photos)
        dup_projects = self.projects[
            self.projects["seeded_anomaly_types"].str.contains("duplicate_photo", na=False)
        ]["project_id"]
        for pid in dup_projects:
            self.assertTrue(result.loc[pid, "has_duplicate_photo"],
                             f"{pid} was seeded with a duplicate photo but engine missed it")

    def test_vision_geo_flags_geo_mismatches(self):
        result = run_vision_geo_engine(self.projects, self.photos)
        geo_projects = self.projects[
            self.projects["seeded_anomaly_types"].str.contains("geo_mismatch", na=False)
        ]["project_id"]
        for pid in geo_projects:
            self.assertGreater(result.loc[pid, "max_geo_distance_m"], 500,
                                f"{pid} was seeded with a geo mismatch but engine missed it")

    def test_contractor_network_finds_cartel_clusters(self):
        result, G = run_contractor_network_engine(self.projects, self.contractors)
        cartel_ids = self.contractors[self.contractors.is_cartel_seed]["contractor_id"]
        # every seeded cartel contractor should have at least one graph edge
        for cid in cartel_ids:
            self.assertGreater(G.degree(cid), 0, f"{cid} was seeded as cartel but has no graph links")

    def test_contractor_network_has_no_accidental_links_among_clean_contractors(self):
        """Regression test: earlier versions of the synthetic data
        generator drew director names / bank prefixes from a small enough
        pool that non-cartel contractors collided by chance, creating
        false cartel-link edges. Guards against that regressing."""
        result, G = run_contractor_network_engine(self.projects, self.contractors)
        clean_ids = set(self.contractors[~self.contractors.is_cartel_seed]["contractor_id"])
        for cid in clean_ids:
            if cid in G:
                self.assertEqual(G.degree(cid), 0,
                                  f"{cid} is not a seeded cartel contractor but has graph links — "
                                  f"likely an accidental name/address/bank collision in the generator")

    def test_vision_geo_duplicate_detection_precision(self):
        """Regression test: an earlier version of the visual-consistency
        engine flagged hundreds of unrelated same-category photos as
        'duplicates' because the perceptual-hash threshold alone was too
        loose for visually-similar-but-distinct procedural images. Every
        flagged project must trace back to an actual seeded duplicate
        photo pairing (either side of the pair)."""
        result = run_vision_geo_engine(self.projects, self.photos)
        dup_project_ids = set(self.projects[
            self.projects["seeded_anomaly_types"].str.contains("duplicate_photo", na=False)
        ]["project_id"])

        flagged = result[result["has_duplicate_photo"]]
        # every flagged project's partner list must include at least one
        # genuinely seeded duplicate-photo project (itself or the project
        # whose photo it shares)
        for pid, row in flagged.iterrows():
            partner_projects = set()
            for flag in row["geo_vision_flags"]:
                if flag.startswith("identical_photo_also_used_in:"):
                    partner_projects.update(flag.split(":", 1)[1].split(","))
            implicated = {pid} | partner_projects
            self.assertTrue(
                implicated & dup_project_ids,
                f"{pid} flagged as duplicate but neither it nor its partners "
                f"({partner_projects}) were seeded as duplicate_photo anomalies",
            )

    def test_document_engine_runs_without_error(self):
        result = run_document_engine(self.projects, self.transactions)
        self.assertEqual(len(result), len(self.projects))
        self.assertTrue((result["document_score"] >= 0).all())

    def test_document_engine_overdue_rule_grounded_in_guideline(self):
        """Projects sanctioned long ago (per REPORT_AS_OF_DATE) and not
        marked complete by the Implementing Agency must be flagged —
        this rule is directly grounded in the eSAKSHI process guideline
        that sanctioned works are generally completed within one year."""
        result = run_document_engine(self.projects, self.transactions)
        merged = self.projects.set_index("project_id").join(result)
        merged["sanction_date"] = pd.to_datetime(merged["sanction_date"])
        as_of = pd.Timestamp(REPORT_AS_OF_DATE)
        days_since = (as_of - merged["sanction_date"]).dt.days

        clearly_overdue = (days_since > 400) & (~merged["marked_complete_by_ia"].astype(bool))
        clearly_not_overdue = (days_since < 300) | (merged["marked_complete_by_ia"].astype(bool))

        overdue_flagged = merged.loc[clearly_overdue, "document_flags"].apply(
            lambda l: "overdue_beyond_one_year_sanction_guideline" in l
        )
        not_overdue_flagged = merged.loc[clearly_not_overdue, "document_flags"].apply(
            lambda l: "overdue_beyond_one_year_sanction_guideline" in l
        )
        self.assertTrue(overdue_flagged.all(), "Some clearly-overdue projects were not flagged")
        self.assertFalse(not_overdue_flagged.any(), "Some clearly-not-overdue projects were incorrectly flagged")

    def test_composite_risk_end_to_end(self):
        fin = run_financial_engine(self.projects, self.transactions)
        cost = run_cost_benchmark_engine(self.projects)
        geo = run_vision_geo_engine(self.projects, self.photos)
        net, _ = run_contractor_network_engine(self.projects, self.contractors)
        doc = run_document_engine(self.projects, self.transactions)
        composite = compute_composite_risk(fin, cost, geo, net, doc)

        self.assertEqual(len(composite), len(self.projects))
        self.assertTrue((composite["composite_score"] >= 0).all())
        self.assertTrue((composite["composite_score"] <= 100).all())
        self.assertTrue(composite["risk_band"].isin(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).all())

        merged = self.projects.set_index("project_id").join(composite.set_index("project_id"))
        anomalous_mean = merged.loc[merged.is_seeded_anomalous, "composite_score"].mean()
        clean_mean = merged.loc[~merged.is_seeded_anomalous, "composite_score"].mean()
        self.assertGreater(anomalous_mean, clean_mean,
                            "Seeded anomalous projects should score higher on average than clean ones")


class TestTrainedMLModels(unittest.TestCase):
    """These are the genuinely LEARNED-model components (as opposed to
    fixed rules or unsupervised-only detectors): a trained image
    classifier and a supervised risk classifier, both evaluated with
    proper cross-validation. Every metric asserted here is what actually
    backs the model-card numbers in the README — if these thresholds
    ever fail, the README numbers are stale and need regenerating too."""

    @classmethod
    def setUpClass(cls):
        cls.projects = pd.read_csv(DATA_DIR / "projects.csv")
        cls.transactions = pd.read_csv(DATA_DIR / "transactions.csv")
        cls.photos = pd.read_csv(DATA_DIR / "photos.csv")
        cls.contractors = pd.read_csv(DATA_DIR / "contractors.csv")

    def test_vision_classifier_beats_chance_by_a_wide_margin(self):
        metrics = evaluate_vision_classifier(self.photos)
        chance_level = 1.0 / metrics["n_categories"]
        self.assertGreater(metrics["accuracy"], chance_level * 4,
                            "Trained category classifier should clearly beat random guessing")
        # Sanity ceiling too — a suspiciously perfect classifier on
        # procedurally generated images would suggest a data leak, not a
        # good model.
        self.assertLess(metrics["accuracy"], 1.0)

    def test_ml_risk_classifier_beats_chance_by_a_wide_margin(self):
        fin = run_financial_engine(self.projects, self.transactions)
        cost = run_cost_benchmark_engine(self.projects)
        geo = run_vision_geo_engine(self.projects, self.photos)
        net, _ = run_contractor_network_engine(self.projects, self.contractors)
        doc = run_document_engine(self.projects, self.transactions)
        engine_outputs = {"financial": fin, "cost": cost, "geo": geo, "network": net, "document": doc}

        metrics = evaluate_ml_classifier(engine_outputs, self.projects)
        # Cross-validated, out-of-fold — these are not training-set numbers.
        self.assertGreater(metrics["roc_auc"], 0.85, "ROC-AUC should indicate strong discrimination")
        self.assertGreater(metrics["pr_auc"], 0.6, "PR-AUC should be well above the positive-class base rate")
        self.assertGreater(metrics["f1"], 0.6)

    def test_ml_risk_classifier_output_shape_and_range(self):
        fin = run_financial_engine(self.projects, self.transactions)
        cost = run_cost_benchmark_engine(self.projects)
        geo = run_vision_geo_engine(self.projects, self.photos)
        net, _ = run_contractor_network_engine(self.projects, self.contractors)
        doc = run_document_engine(self.projects, self.transactions)
        engine_outputs = {"financial": fin, "cost": cost, "geo": geo, "network": net, "document": doc}

        result = train_and_score(engine_outputs, self.projects)
        self.assertEqual(len(result), len(self.projects))
        self.assertTrue((result["ml_risk_probability"] >= 0).all())
        self.assertTrue((result["ml_risk_probability"] <= 1).all())


class TestAuditChain(unittest.TestCase):
    def test_chain_verifies_after_appends(self):
        append_event("TEST_EVENT", "PRJ_TEST", "unit_test", {"note": "chain test"})
        valid, broken_at = verify_chain()
        self.assertTrue(valid, f"Chain broke at entry {broken_at}")

    def test_genesis_hash_format(self):
        self.assertEqual(len(GENESIS_HASH), 64)
        self.assertTrue(all(c == "0" for c in GENESIS_HASH))


class TestDateRealism(unittest.TestCase):
    """Regression test: anchoring sanction_date to REPORT_AS_OF_DATE
    (today) without bounding downstream event-date offsets caused
    transactions/photos/citizen-reports to be dated in the future
    (up to ~2028). Every event date must be on or before today."""

    def test_no_event_dates_in_the_future(self):
        as_of = pd.Timestamp(REPORT_AS_OF_DATE)

        txns = pd.read_csv(DATA_DIR / "transactions.csv")
        txns["txn_date"] = pd.to_datetime(txns["txn_date"])
        self.assertTrue((txns["txn_date"] <= as_of).all(), "Some transactions are dated in the future")

        photos = pd.read_csv(DATA_DIR / "photos.csv")
        photos["capture_date"] = pd.to_datetime(photos["capture_date"])
        self.assertTrue((photos["capture_date"] <= as_of).all(), "Some photos are dated in the future")

        citizen = pd.read_csv(DATA_DIR / "citizen_reports.csv")
        citizen["report_date"] = pd.to_datetime(citizen["report_date"])
        self.assertTrue((citizen["report_date"] <= as_of).all(), "Some citizen reports are dated in the future")


class TestDocumentOCR(unittest.TestCase):
    """Real OCR (pytesseract over the actual Tesseract binary, not
    mocked) against real rendered document images. This is the
    previously-documented-but-not-implemented 'OCR upgrade path' —
    these tests are what make it fair to call it implemented now."""

    @classmethod
    def setUpClass(cls):
        cls.docs = pd.read_csv(DATA_DIR / "sanction_order_documents.csv")
        cls.projects = pd.read_csv(DATA_DIR / "projects.csv")

    def test_ocr_extracts_correct_amount_from_a_known_document(self):
        doc = self.docs.iloc[0]
        doc_path = DATA_DIR.parent / "sample_documents" / doc["filename"]
        text = extract_text(doc_path)
        parsed = parse_sanction_order(text)
        self.assertIsNotNone(parsed["extracted_amount"])
        self.assertAlmostEqual(parsed["extracted_amount"], doc["printed_amount"], delta=1.0)

    def test_ocr_pipeline_catches_seeded_amount_mismatches(self):
        metrics = evaluate_ocr_pipeline(self.docs, self.projects)
        self.assertGreater(metrics["ocr_success_rate"], 0.9,
                            "OCR should reliably read these clean, computer-rendered documents")
        self.assertGreater(metrics["precision"], 0.8)
        self.assertGreater(metrics["recall"], 0.8)


class TestAuthAndRBAC(unittest.TestCase):
    """Password hashing correctness and the actual scope-enforcement
    behavior of scoped_query() — this is the part that matters most:
    a role check that only hides a UI element but still returns
    unscoped data underneath would be worse than no auth at all."""

    def setUp(self):
        create_user("test_admin_u", "Test Admin", "central_admin", "pw1")
        create_user("test_mp_u", "Test MP", "mp", "pw2", scope_mp_id="MP001")
        create_user("test_officer_u", "Test Officer", "district_officer", "pw3", scope_district="District X")

    def test_wrong_password_rejected(self):
        self.assertIsNone(authenticate("test_admin_u", "wrong_password"))

    def test_correct_password_accepted(self):
        user = authenticate("test_admin_u", "pw1")
        self.assertIsNotNone(user)
        self.assertEqual(user["role"], "central_admin")

    def test_nonexistent_user_rejected(self):
        self.assertIsNone(authenticate("nobody_at_all", "anything"))

    def test_mp_scope_actually_restricts_rows(self):
        df = pd.DataFrame({
            "project_id": ["A", "B", "C"],
            "mp_id": ["MP001", "MP002", "MP001"],
            "district": ["D1", "D2", "D1"],
        })
        user = authenticate("test_mp_u", "pw2")
        scoped = scoped_query(df, user)
        self.assertEqual(len(scoped), 2)
        self.assertTrue((scoped["mp_id"] == "MP001").all())

    def test_district_officer_scope_actually_restricts_rows(self):
        df = pd.DataFrame({
            "project_id": ["A", "B", "C"],
            "mp_id": ["MP001", "MP002", "MP003"],
            "district": ["District X", "District Y", "District X"],
        })
        user = authenticate("test_officer_u", "pw3")
        scoped = scoped_query(df, user)
        self.assertEqual(len(scoped), 2)
        self.assertTrue((scoped["district"] == "District X").all())

    def test_central_admin_sees_everything(self):
        df = pd.DataFrame({"project_id": ["A", "B"], "mp_id": ["MP001", "MP002"], "district": ["D1", "D2"]})
        user = authenticate("test_admin_u", "pw1")
        scoped = scoped_query(df, user)
        self.assertEqual(len(scoped), len(df))

    def test_no_session_returns_zero_rows(self):
        df = pd.DataFrame({"project_id": ["A", "B"], "mp_id": ["MP001", "MP002"], "district": ["D1", "D2"]})
        scoped = scoped_query(df, None)
        self.assertEqual(len(scoped), 0)


class TestEsakshiIngestion(unittest.TestCase):
    """The ingestion pipeline actually writes to
    data/real_mp_allocated_limits.csv, so every test here backs it up in
    setUp and restores it in tearDown — these tests must never leave the
    shipped real dataset mutated for the next test run or for the app."""

    def setUp(self):
        self.real_data_path = DATA_DIR / "real_mp_allocated_limits.csv"
        self.backup = pd.read_csv(self.real_data_path)

    def tearDown(self):
        self.backup.to_csv(self.real_data_path, index=False)

    def test_dry_run_never_writes(self):
        before = pd.read_csv(self.real_data_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        pd.DataFrame({
            "state": ["Kerala"], "mp_name": ["DRY RUN TEST MP"],
            "constituency": ["TEST SEAT"], "allocated_amount": [147000000],
        }).to_csv(tmp.name, index=False)

        report = ingest_allocation_export(tmp.name, dry_run=True)
        after = pd.read_csv(self.real_data_path)
        self.assertEqual(len(before), len(after), "dry_run must never modify the target file")
        self.assertFalse((after["mp_name"] == "DRY RUN TEST MP").any())
        self.assertEqual(report["problems"], [])

    def test_validation_rejects_bad_data_without_writing(self):
        before = pd.read_csv(self.real_data_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        pd.DataFrame({
            "state": ["Kerala", "Kerala"],
            "mp_name": ["", "DUP MP"],  # missing name
            "constituency": ["SEAT A", "SEAT A"],
            "allocated_amount": [147000000, -999],  # negative amount
        }).to_csv(tmp.name, index=False)

        with self.assertRaises(IngestionValidationError) as ctx:
            ingest_allocation_export(tmp.name)
        self.assertGreater(len(ctx.exception.problems), 0)

        after = pd.read_csv(self.real_data_path)
        self.assertEqual(len(before), len(after), "a failed validation must not touch the target file")

    def test_incremental_update_adds_new_and_updates_existing(self):
        existing_mp = pd.read_csv(self.real_data_path).iloc[0]
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        pd.DataFrame({
            "state": [existing_mp["state"], "Kerala"],
            "mp_name": [existing_mp["mp_name"], "BRAND NEW TEST MP"],
            "constituency": [existing_mp["constituency"], "NEW TEST SEAT"],
            "allocated_amount": [999999999, 147000000],  # deliberately changed value for the existing MP
        }).to_csv(tmp.name, index=False)

        report = ingest_allocation_export(tmp.name)
        self.assertEqual(report["rows_added"], 1)
        self.assertEqual(report["rows_updated"], 1)

        after = pd.read_csv(self.real_data_path)
        updated_row = after[after["mp_name"] == existing_mp["mp_name"]].iloc[0]
        self.assertEqual(updated_row["allocated_amount"], 999999999)
        self.assertTrue((after["mp_name"] == "BRAND NEW TEST MP").any())

    def test_idempotent_reingestion_produces_no_duplicates(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        pd.DataFrame({
            "state": ["Kerala"], "mp_name": ["IDEMPOTENCY TEST MP"],
            "constituency": ["TEST SEAT"], "allocated_amount": [147000000],
        }).to_csv(tmp.name, index=False)

        ingest_allocation_export(tmp.name)
        report2 = ingest_allocation_export(tmp.name)
        self.assertEqual(report2["rows_added"], 0, "re-ingesting the same file must not add a duplicate row")

        after = pd.read_csv(self.real_data_path)
        self.assertEqual((after["mp_name"] == "IDEMPOTENCY TEST MP").sum(), 1)

    def test_grand_total_row_is_detected_and_excluded(self):
        """Regression test: real government exports commonly end with a
        'Grand Total' summary row (no MP name/state/constituency, just a
        sum). This must be stripped, not ingested as a fake MP record."""
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        pd.DataFrame({
            "state": ["Kerala", None],
            "mp_name": ["REAL MP", None],
            "constituency": ["REAL SEAT", None],
            "allocated_amount": [147000000, 83180553326],  # second row is the fake "grand total"
        }).to_csv(tmp.name, index=False)

        report = ingest_allocation_export(tmp.name, dry_run=True)
        self.assertEqual(report["summary_rows_dropped"], 1)
        self.assertEqual(report["rows_in_file"], 1)
        self.assertEqual(report["problems"], [])


class TestRealAllocationData(unittest.TestCase):
    """The one dataset in this repo that is REAL, published eSAKSHI data
    (not synthetic) — sanity-checks it loaded and parsed correctly."""

    def test_real_mp_allocation_file_loads_and_is_plausible(self):
        df = pd.read_csv(DATA_DIR / "real_mp_allocated_limits.csv")
        self.assertGreater(len(df), 500)  # Lok Sabha has 543 seats
        self.assertTrue(df["mp_name"].notna().all())
        # One row (Nanded) genuinely has no allocated_amount published on
        # the portal as of this snapshot — preserved as-is (not fabricated)
        # rather than silently dropped or zero-filled. Everything else
        # must be a plausible positive entitlement.
        known = df["allocated_amount"].dropna()
        self.assertGreater(len(known), 500)
        self.assertTrue((known > 0).all())
        self.assertTrue((known >= 1e7).all())
        self.assertTrue((known <= 5e8).all())


if __name__ == "__main__":
    unittest.main()
