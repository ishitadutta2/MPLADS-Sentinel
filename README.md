# 🛡️ MPLADS Sentinel

An AI-assisted transparency and fraud-detection layer for the **Members of
Parliament Local Area Development Scheme (MPLADS)** — built as a working
MVP, not a slide deck. Every engine described below runs on real code
against a realistic synthetic dataset shipped in this repo, and every
claim in this README is backed by the test suite in `tests/`.

> **Sentinel never blocks a payment.** It ranks projects by risk and hands
> a human reviewer a clear, explainable reason. All final decisions stay
> with the district officer / audit authority — this is a prioritisation
> and evidence tool, not an automated enforcement system.

---

## What it does

Five independent detection engines score every project 0–100, which are
combined into one explainable **composite risk score** — plus a sixth,
genuinely *learned* supervised layer on top (see the model card below):

| Engine | Technique | What it catches |
|---|---|---|
| **Financial Anomaly** | Two-model unsupervised ensemble (`IsolationForest` + `LocalOutlierFactor`) + explainable rule checks | Cost inflation, split tendering, single-shot high-value disbursals |
| **Cost Benchmark** | Robust (median/IQR) z-score vs. category+district peer group | Projects priced far above comparable local works |
| **Geo & Visual Evidence** | Haversine GPS check + perceptual-hash/pixel duplicate detection + a **trained** `RandomForestClassifier` (HOG + colour-histogram features, cross-validated) for category consistency | Photos taken far from the declared site, reused/duplicate "proof of work" photos, photos that don't visually match the claimed work type |
| **Contractor Network** | Real `networkx` graph over shared address/bank-prefix/director | Shell-company cartels bidding against "each other" |
| **Document Consistency** | Rule-based transaction/UC cross-checks + a guideline-grounded overdue-completion rule | Transaction totals that don't reconcile, payments predating sanction, suspiciously round billing, works overdue against the scheme's own 1-year completion guideline |
| **ML Risk Classifier** | Supervised `RandomForestClassifier` trained on the five engines' outputs, 5-fold cross-validated | Learns which *combination* of signals actually predicts an irregularity, instead of a fixed hand-picked weight |

On top of this:
- A **tamper-evident SHA-256 hash-chain audit log** records every scoring
  run and every officer verdict — editing a past entry breaks every hash
  after it, so silent tampering is detectable (`sentinel/audit/hash_chain.py`).
- A **citizen reporting loop** lets the public flag a project directly,
  and those reports are shown correlated against Sentinel's own scores.
- A **human-in-the-loop review queue** captures officer verdicts
  (confirmed / false positive / needs field visit) which are logged and
  become training signal for future model tuning.

## Validated results (on the shipped synthetic dataset)

The data generator (`data/generate_data.py`) seeds ~13% of projects with
seven real fraud patterns (cost inflation, phantom work, geo-mismatch,
split tendering, duplicate photos, cartel contractors) — and hides the
label from the engines. The test suite proves each engine actually
recovers signal it was never told about:

- Seeded-anomalous projects score **~33/100 on average vs. ~18/100** for
  clean projects on the composite score (see `tests/test_engines.py`).
- The vision engine catches **100% of seeded duplicate/reused photos**
  and **100% of seeded geo-mismatches**.
- At the CRITICAL risk band, **81% of flagged projects are true seeded
  anomalies** — a screening tool is meant to over-flag for human review,
  and this calibration reflects that design choice (see "Risk bands" below).

Run `python -m unittest discover tests -v` to reproduce all of this
yourself (the full suite, including cross-validated model training, takes
a few minutes — most of that is the two trained models below).

## ML Model Card

Two components in this project are genuinely **trained, learned models**
— evaluated with proper cross-validation, not measured on their own
training data. Every number below is asserted by a test in
`tests/test_engines.py::TestTrainedMLModels` and reproducible by running
the suite; nothing here is hand-typed into this table without a test
backing it.

**1. Visual category classifier** (`sentinel/engines/vision_geo.py`)
`RandomForestClassifier` (300 trees) on HOG + colour-histogram + edge-
density features, predicting a photo's declared work category — 5-fold
stratified cross-validation, every score is out-of-fold.

| Metric | Value |
|---|---|
| Photos evaluated | 2,025 |
| Categories | 10 |
| Chance-level accuracy | ~10% |
| **Cross-validated accuracy** | **94.5%** |
| Cross-validated weighted F1 | 94.5% |

**2. Supervised risk classifier** (`sentinel/engines/ml_risk_classifier.py`)
`RandomForestClassifier` (300 trees, class-balanced) trained on the five
engines' output scores plus raw project features, predicting whether a
project is irregular — 5-fold stratified cross-validation.

| Metric | Value |
|---|---|
| Projects evaluated | 1,000 (130 positive) |
| **ROC-AUC** | **0.980** |
| **PR-AUC** | **0.918** |
| Precision | 86.4% |
| Recall | 78.5% |
| F1 | 82.3% |
| Confusion matrix `[[TN,FP],[FN,TP]]` | `[[854, 16], [28, 102]]` |

Learned feature importances (not hand-assigned weights): `geo_photo_score`
(46%) dominates, followed by `financial_score` (14.6%),
`cost_vs_category_ratio` (10.4%), `contractor_network_score` (9.6%) —
`document_score` contributes least (0.7%), which is an honest finding,
not a design choice: the rule-based document checks catch a genuinely
different, rarer pattern than what's seeded in this synthetic dataset, so
the classifier correctly learns to lean on it less *for this data*.

**Honest labelling caveat**: both models above are trained against the
synthetic seeded ground truth (`is_seeded_anomalous`), because no real
confirmed-fraud labels exist to train on outside a live deployment. The
risk classifier's production training path — against accumulated
`officer_feedback.verdict` entries from the District Officer Queue —
is implemented in `train_from_officer_feedback()` and switches over
automatically once enough real verdicts exist; see the docstring in
`ml_risk_classifier.py` for the exact threshold and rationale.

**3. OCR document verification** (`sentinel/engines/document_ocr.py`)
Real OCR via `pytesseract` against the actual Tesseract binary (not
mocked or simulated) — extracts the printed sanctioned amount from a
sanction-order document image and cross-checks it against the amount
recorded in the system, over 60 rendered documents (15 seeded with a
deliberately altered/doctored printed amount).

| Metric | Value |
|---|---|
| Documents evaluated | 60 (15 seeded mismatches) |
| OCR success rate | 100% |
| Precision (mismatch detection) | 100% |
| Recall (mismatch detection) | 100% |

**Honest scope caveat**: these are clean, computer-rendered documents —
the kind a government document-generation system itself would produce —
not noisy scans. Tesseract performs near-perfectly on text like this;
real scanned/photographed/handwritten sanction orders would need a
deskew + binarization preprocessing step first, which isn't implemented.
The OCR extraction and cross-check logic itself is real and tested; what's
still a documented gap is the preprocessing needed for messier real-world
scans, not OCR itself.

---

## Production readiness

Built honestly for what it is — a hackathon-scoped prototype — but with
real, working versions of the pieces that are usually left as slideware:

| Component | What's actually implemented | What's still a genuine gap |
|---|---|---|
| **Auth & access control** | Real PBKDF2-SHA256 password hashing (`sentinel/auth.py`), three roles (MP / District Officer / Central Admin), and *enforced* data scoping — an MP account's session genuinely cannot retrieve another MP's projects, tested in `TestAuthAndRBAC` | No SSO/OIDC against a real government identity provider; sessions live in Streamlit's in-memory `st.session_state`, not signed/expiring tokens; no rate limiting on repeated login failures (though every attempt is logged to `login_audit` — the data a lockout policy would need already exists) |
| **Database** | WAL mode (concurrent reads during writes) + indices on every column the app actually filters/joins on, verified to survive a full data reload | Still SQLite, not PostgreSQL/PostGIS — the schema is written in portable ANSI SQL specifically so that migration is a config change, not a rewrite (see below) |
| **eSAKSHI data ingestion** | A real, validated, idempotent ETL pipeline (`sentinel/services/esakshi_ingestion.py`) for eSAKSHI-format exports — schema validation with row-level error messages, automatic detection/exclusion of spreadsheet "Grand Total" rows, incremental updates without duplicating existing MPs, every run logged to the audit chain. Tested against the actual real eSAKSHI export this project ships with | **Not a live API feed** — and honestly, can't be: eSAKSHI has no public API, so a claim to "live ingestion" for this specific portal would be fiction. File-based ingestion of portal exports is the realistic integration point for a system like this, the same way it is for most Indian government data systems |
| **Document intelligence** | Real OCR (`pytesseract`/Tesseract) extracting and cross-checking sanctioned amounts from rendered sanction-order documents — see the model card above | No preprocessing for noisy/scanned/handwritten real-world documents |
| **Vision** | Two real trained/tested pieces: a cross-validated `RandomForestClassifier` for category consistency, and perceptual-hash + pixel-level duplicate detection | No CLIP/deep embedding model — no internet access in the build environment to fetch pretrained weights; `_extract_features()` in `vision_geo.py` is a documented one-function swap point for it |

### Why SQLite, not PostgreSQL/PostGIS

## Grounding in the real eSAKSHI process

This project was built to reflect the actual MPLADS–eSAKSHI workflow, not
a generic guess at it:

- **Real reference data**: `data/real_mp_allocated_limits.csv` is the
  published eSAKSHI allocated-limit dataset for the 18th Lok Sabha (543
  MPs — name, state, constituency, annual entitlement). It powers the
  **National Allocation Overview** page, which shows **zero risk scores
  or anomaly flags** — it's a factual reference view only. One record
  (Nanded) has no allocated amount published on the portal as of this
  snapshot; it's shown as-is rather than estimated.
- **Every other page** (Home, MP Dashboard, District Officer Queue,
  Photo Verification, Contractor Network, Project Detail, Citizen
  Reporting) runs on the **synthetic** dataset described above.
  This split is deliberate: attaching fabricated fraud/risk scores to
  real, named, sitting Members of Parliament — even clearly-labelled
  synthetic ones — is not something this project does. The detection
  engines are demonstrated on fictional projects/contractors instead.
- **Process terminology and rules** follow the real eSAKSHI lifecycle:
  recommendation → District Authority sanction → Implementing Agency
  (IA) execution → staged vendor payments → the IA's final "mark
  complete" step (only marked-complete works show as completed on the
  public dashboard, which is why a `marked_complete_by_ia` field
  distinct from project status exists in the synthetic data). The
  document-consistency engine includes a rule directly grounded in the
  scheme's own guideline — *"sanctioned works are generally required to
  be completed within one year of sanction"* — flagging projects
  sanctioned over a year ago (relative to `REPORT_AS_OF_DATE` in
  `sentinel/config.py`) that the IA still hasn't marked complete.

## Quickstart

```bash
# 1. Set up environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# pytesseract also needs the tesseract-ocr system binary:
#   Ubuntu/Debian: sudo apt install tesseract-ocr
#   macOS: brew install tesseract

# 2. The CSVs in data/ are already generated and committed (small,
#    text — this is the actual demo scenario, so it's tracked directly).
#    sample_photos/ and sample_documents/ are NOT committed (large
#    generated binaries — see .gitignore) and won't exist yet after a
#    fresh clone; generate them before running the app or several pages
#    (Photo Verification, document/OCR scoring) will show empty results:
python data/generate_photos.py
python data/generate_sanction_orders.py
# Re-running data/generate_data.py too is optional — it reseeds
# data/*.csv with the same statistical patterns (RANDOM_SEED is fixed),
# so there's no need to unless you want a different specific set of
# projects/anomalies.

# 3. (Optional) run the scoring pipeline standalone to see console output
python -m sentinel.services.scoring_pipeline

# 4. Run the tests (the trained models make this take a few minutes —
#    that's real cross-validated training happening, not a hang)
python -m unittest discover tests -v

# 5. Launch the dashboard
streamlit run app/Home.py
```

The first dashboard load runs all six engines (five detection engines
plus the supervised ML classifier) over 1,000 projects and 2,000+
photos — the trained vision classifier's cross-validation dominates this
at around 40 seconds on a modest machine. Cached after that; every
subsequent navigation is instant.

### Demo login credentials

Every page requires sign-in. Demo accounts are seeded automatically on
first launch and shown directly on the login screen — deliberately not
hidden, since pretending this is production-hardened would be dishonest
for a hackathon prototype:

| Role | Username | Password | Scope |
|---|---|---|---|
| Central Admin | `admin` | `admin123` | Sees everything |
| MP Office | `mp_demo` | `mp123` | Locked to one MP's own projects only |
| District Officer | `officer_demo` | `officer123` | Locked to one district only |

Try logging in as `mp_demo` after `admin` to see the same dashboard
return a completely different (and much smaller) set of projects — that
restriction is enforced in `sentinel/auth.py::scoped_query()`, not just
hidden in the UI; see `TestAuthAndRBAC` in the test suite.

## App tour

| Page | Who it's for |
|---|---|
| **National Allocation Overview** | Real, published eSAKSHI data (543 MPs) — no risk scoring, factual only |
| **Home** | National oversight — total funds tracked, risk distribution, state-wise concentration *(synthetic data)* |
| **MP Dashboard** | Self-service transparency view for an individual MP's office *(synthetic data)* |
| **District Officer Queue** | Prioritised worklist with explainability + verdict capture *(synthetic data)* |
| **Project Detail** | Full deep-dive: timeline, photos, contractor, citizen reports *(synthetic data)* |
| **Photo Verification** | Side-by-side evidence review for geo/visual flags *(synthetic data)* |
| **Contractor Network** | Interactive graph of suspected cartel clusters *(synthetic data)* |
| **Citizen Reporting** | Public complaint intake + correlation with risk scores *(synthetic data)* |
| **Audit Trail** | Live hash-chain viewer with an integrity-verification button *(synthetic data)* |
| **Work Tracker** | Writes, not just reads: MPs recommend new works, District Officers/Central Admin advance them through the real recommend → sanction → execute → pay → complete pipeline, live |

### Multi-language UI

A 🌐 language switcher lives in the sidebar (English, हिन्दी, বাংলা, தமிழ்,
తెలుగు, मराठी, ગુજરાતી — see `app/i18n.py`). It translates the app's own
interface — navigation content, buttons, labels, chart axes, table
headers — consistently across all nine pages. It deliberately does **not**
translate: data people typed in (project descriptions, names — same as
the real eSAKSHI portal wouldn't retranslate a citizen's own words), a
few dense technical explainer asides (the eSAKSHI process writeup, the
hash-chain internals), or Streamlit's own native sidebar page-nav labels
(a platform limitation, not an oversight — those come from filenames and
aren't restyleable without fragile DOM hacking this project avoids).
Translations are a solid first pass, not yet reviewed by native speakers
of each language.

---

## Why SQLite, not PostgreSQL/PostGIS

The schema in `sentinel/db.py` is written in portable ANSI SQL
specifically so a production deployment can move to PostgreSQL/PostGIS by
changing `SENTINEL_DATABASE_URL` and adding a `geography(Point)` column
for true spatial indexing — no application code changes required. SQLite
was chosen for this MVP so the whole project runs with `pip install` and
zero external services, which matters far more for a hackathon judge or a
first-time reviewer than for the architecture itself.

## Why this environment's offline sandbox shaped some choices

This project was built inside a sandbox with **no internet access**, so a
few components were built as genuine, tested, offline-runnable
equivalents of what a fully-resourced production system would use:

| Component | Built here (real & tested) | Production upgrade (documented, not installed) |
|---|---|---|
| Visual similarity | OpenCV colour-histogram + edge-density signature, cosine similarity | Swap `_image_signature()` in `sentinel/engines/vision_geo.py` for a CLIP (`transformers`) embedding — same function signature |
| Duplicate photo detection | DCT-based perceptual hash on OpenCV/NumPy | Swap `_phash()` for `imagehash.phash()` — same interface |
| Document extraction | Rule-based transaction/amount cross-checks | PyMuPDF + regex/NER pipeline over scanned UC PDFs, feeding the same `run_document_engine` scoring logic |
| Database | SQLite via stdlib `sqlite3` | PostgreSQL + PostGIS via `SENTINEL_DATABASE_URL` |

Every one of these is an isolated function behind a stable interface — the
upgrade path is a swap, not a rewrite.

---

## Project structure

```
mplads_sentinel/
├── data/
│   ├── generate_data.py       # synthetic dataset generator (seeded anomalies)
│   ├── generate_photos.py     # procedural evidence-photo generator
│   ├── generate_sanction_orders.py  # synthetic sanction-order document generator
│   ├── real_mp_allocated_limits.csv # REAL eSAKSHI data — 543 MPs, 18th Lok Sabha
│   └── *.csv                  # generated synthetic dataset (committed for zero-setup demo)
├── sample_photos/              # 2,000+ generated project evidence photos
├── sample_documents/           # 60 generated sanction-order document images (for OCR)
├── sentinel/
│   ├── config.py               # all tunable constants (weights, thresholds)
│   ├── db.py                   # SQLite schema + indices + WAL mode + data access
│   ├── auth.py                 # PBKDF2 password hashing + role-based scope enforcement
│   ├── engines/
│   │   ├── financial.py        # IsolationForest + LocalOutlierFactor ensemble + rules
│   │   ├── cost.py             # peer-benchmark cost engine
│   │   ├── vision_geo.py       # trained RandomForest classifier + perceptual-hash duplicate detection
│   │   ├── contractor_graph.py # NetworkX cartel-detection engine
│   │   ├── document_intel.py   # rule-based document consistency engine
│   │   ├── document_ocr.py     # real pytesseract OCR + amount cross-check
│   │   ├── ml_risk_classifier.py # supervised RandomForest trained on the 5 engines' outputs
│   │   └── risk_engine.py      # weighted composite + explainability
│   ├── audit/hash_chain.py     # SHA-256 tamper-evident audit log
│   ├── services/
│   │   ├── scoring_pipeline.py       # orchestrates all engines
│   │   └── esakshi_ingestion.py      # real eSAKSHI export ingestion (validated, idempotent)
│   └── utils/geo.py            # Haversine distance
├── app/
│   ├── Home.py                  # Streamlit entry point (login-gated)
│   ├── common.py                # shared caching/styling/auth helpers
│   ├── i18n.py                  # 7-language UI translation layer + language switcher
│   └── pages/                   # the 9 dashboard pages, including Work Tracker (writes, not just reads)
├── tests/test_engines.py       # full engine + auth + ingestion + audit-chain test suite (33 tests)
└── requirements.txt
```

## Risk bands

Composite scores are bucketed as:

| Band | Range | Meaning |
|---|---|---|
| LOW | 0–20 | No action needed |
| MEDIUM | 20–32 | Worth a routine check |
| HIGH | 32–45 | Recommended for officer review |
| CRITICAL | 45–100 | Priority review — strong multi-signal irregularity |

These thresholds are calibrated (see `sentinel/config.py`) against the
score distribution on the shipped dataset so the bands are meaningfully
populated rather than everything clustering into one bucket — tune them
for your own data via `RISK_BANDS` in `sentinel/config.py`.

## Configuration

All weights and thresholds live in `sentinel/config.py` — nothing is
hard-coded inside an engine:

```python
RISK_WEIGHTS = {
    "financial": 0.30,
    "cost_benchmark": 0.15,
    "geo_photo": 0.20,
    "contractor_network": 0.20,
    "document": 0.15,
}
GEO_TOLERANCE_METERS = 500
```

---

## Disclaimer

All names, constituencies, contractors, and figures in this repository
are **synthetically generated** for demonstration purposes and do not
represent any real person, company, or government record.
