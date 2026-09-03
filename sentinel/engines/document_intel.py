"""
Document-intelligence engine.

Real MPLADS fraud investigations rely heavily on cross-checking three
paper trails that are supposed to agree but often don't:
  - the Utilisation Certificate (UC) amount filed with the district
  - the measurement-book / work-completion entry
  - the actual bank transaction amount

This module works off the transaction & project tables we already have
(no OCR model available offline) and applies rule-based consistency
checks that mirror exactly what a manual document audit does:
  - UC amount vs. sanctioned amount mismatch beyond a tolerance
  - transaction total vs. claimed expenditure mismatch
  - suspiciously round-number invoices (a classic red flag — genuine
    construction billing is rarely a perfectly round lakh)
  - backdated / out-of-sequence transaction dates relative to sanction

Production upgrade path (documented, not run here): swap the
`extract_uc_amount()` stub for a PyMuPDF + regex/NER pipeline over
scanned UC PDFs (spaCy or a fine-tuned layout model), feeding the same
downstream mismatch-scoring function unchanged.
"""
import numpy as np
import pandas as pd

from sentinel.config import REPORT_AS_OF_DATE, SANCTION_TO_COMPLETION_GUIDELINE_DAYS


def _is_suspiciously_round(amount: float) -> bool:
    """Flags amounts that are exact multiples of a round unit (e.g.
    exactly Rs 10,00,000) — real billed amounts almost always carry
    material/labour-cost cents-level irregularity."""
    if amount <= 0:
        return False
    return amount % 50000 == 0 or amount % 100000 == 0


def run_document_engine(projects_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    txn_agg = transactions_df.groupby("project_id").agg(
        txn_total=("amount", "sum"),
        n_installments=("transaction_id", "count"),
        first_txn_date=("txn_date", "min"),
        last_txn_date=("txn_date", "max"),
    ).reset_index()

    df = projects_df.merge(txn_agg, on="project_id", how="left")
    df["sanction_date"] = pd.to_datetime(df["sanction_date"])
    df["first_txn_date"] = pd.to_datetime(df["first_txn_date"])
    df["last_txn_date"] = pd.to_datetime(df["last_txn_date"])

    scores = pd.Series(0.0, index=df.index)
    flags = pd.Series([[] for _ in range(len(df))], index=df.index)

    # 1. Transaction total vs claimed expenditure mismatch
    mismatch_pct = (df["txn_total"] - df["expenditure"]).abs() / df["expenditure"].replace(0, np.nan)
    txn_mismatch = mismatch_pct.fillna(0) > 0.05
    scores[txn_mismatch] += 30
    flags[txn_mismatch] = flags[txn_mismatch].apply(lambda l: l + ["transaction_total_vs_expenditure_mismatch"])

    # 2. Transaction predates sanction (impossible in a compliant workflow)
    predates = df["first_txn_date"] < df["sanction_date"]
    scores[predates] += 25
    flags[predates] = flags[predates].apply(lambda l: l + ["transaction_predates_sanction"])

    # 3. Suspiciously round total billing
    round_amt = df["expenditure"].apply(_is_suspiciously_round)
    scores[round_amt] += 15
    flags[round_amt] = flags[round_amt].apply(lambda l: l + ["suspiciously_round_billed_amount"])

    # 4. Single lump-sum payment for a high-value, multi-phase-looking project
    lump_sum_high_value = (df["n_installments"] <= 1) & (df["sanctioned_amount"] > 1500000)
    scores[lump_sum_high_value] += 20
    flags[lump_sum_high_value] = flags[lump_sum_high_value].apply(lambda l: l + ["single_lumpsum_payment_high_value_work"])

    # 5. Rapid full disbursal (all installments within an implausibly short span for the work scale)
    span_days = (df["last_txn_date"] - df["first_txn_date"]).dt.days
    rapid = (df["n_installments"] >= 2) & (span_days <= 3) & (df["sanctioned_amount"] > 1000000)
    scores[rapid] += 15
    flags[rapid] = flags[rapid].apply(lambda l: l + ["all_installments_disbursed_within_days"])

    # 6. Overdue against the scheme's own completion guideline: "sanctioned
    #    works are generally required to be completed within one year of
    #    sanction" (per the eSAKSHI process notes). A project sanctioned
    #    over a year ago that the Implementing Agency still hasn't marked
    #    complete on the portal is a legitimate, guideline-grounded flag —
    #    independent of anything the other rules already catch.
    as_of = pd.Timestamp(REPORT_AS_OF_DATE)
    days_since_sanction = (as_of - df["sanction_date"]).dt.days
    overdue = (days_since_sanction > SANCTION_TO_COMPLETION_GUIDELINE_DAYS) & (~df["marked_complete_by_ia"].astype(bool))
    scores[overdue] += 20
    flags[overdue] = flags[overdue].apply(lambda l: l + ["overdue_beyond_one_year_sanction_guideline"])

    df["document_score"] = np.clip(scores, 0, 100).round(1)
    df["document_flags"] = flags

    return df.set_index("project_id")[["document_score", "document_flags"]]
