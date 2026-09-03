"""
Composite risk engine.

Combines the five independent engines into one explainable 0-100
`composite_score` per project, using the weights declared in
`sentinel.config.RISK_WEIGHTS` (transparent & tunable — not a black box).

Every score ships with a plain-English `explanation` string built
directly from the flags each sub-engine raised, so a district officer
opening a HIGH-risk project sees *why*, not just a number.
"""
import datetime

import pandas as pd

from sentinel.config import RISK_WEIGHTS, risk_band


FLAG_EXPLANATIONS = {
    "expenditure_exceeds_sanction": "Amount spent exceeds the sanctioned budget",
    "possible_split_tendering": "Cost sits just below the scrutiny threshold — possible split tendering",
    "single_installment_high_value": "High-value work paid in a single installment with no milestones",
    "cost_far_above_category_median": "Cost is far above the median for this category",
    "geo_location_mismatch": "Evidence photo GPS location does not match the declared project site",
    "photo_does_not_match_declared_work_type": "Evidence photo does not visually match the declared work type",
    "transaction_total_vs_expenditure_mismatch": "Recorded transactions don't add up to the claimed expenditure",
    "transaction_predates_sanction": "A payment was recorded before the project was even sanctioned",
    "suspiciously_round_billed_amount": "Billed amount is a suspiciously round figure",
    "single_lumpsum_payment_high_value_work": "Large project paid out as a single lump sum",
    "all_installments_disbursed_within_days": "All installments disbursed within an implausibly short window",
    "overdue_beyond_one_year_sanction_guideline": "Sanctioned over a year ago and still not marked complete by the Implementing Agency, beyond the scheme's own completion guideline",
    "active_cartel_cluster_with_multiple_awards": "Contractor belongs to a cluster of linked firms winning multiple awards",
    "high_repeat_award_concentration_single_mp": "Contractor wins an unusually high share of work from a single MP",
}


def _explain_flag(flag: str) -> str:
    if flag in FLAG_EXPLANATIONS:
        return FLAG_EXPLANATIONS[flag]
    if flag.startswith("linked_cluster_of_"):
        n = flag.split("_")[-2]
        return f"Contractor is part of a network of {n} firms sharing address/bank/director details"
    if flag.startswith("shared_identity_links:"):
        n = flag.split(":")[-1]
        return f"Contractor shares identity details with {n} other registered firms"
    if flag.startswith("identical_photo_also_used_in:") or flag == "duplicate_photo":
        return "This project's evidence photo is pixel-identical to a photo used in another project's claim — needs manual review to establish which claim is genuine"
    return flag.replace("_", " ").capitalize()


def compute_composite_risk(
    financial_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    network_df: pd.DataFrame,
    document_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = (
        financial_df[["financial_score", "financial_flags"]]
        .join(cost_df[["cost_benchmark_score"]], how="outer")
        .join(geo_df[["geo_photo_score", "geo_vision_flags"]], how="outer")
        .join(network_df[["contractor_network_score", "network_flags"]], how="outer")
        .join(document_df[["document_score", "document_flags"]], how="outer")
    )

    for col in ["financial_score", "cost_benchmark_score", "geo_photo_score",
                "contractor_network_score", "document_score"]:
        merged[col] = merged[col].fillna(0)
    for col in ["financial_flags", "geo_vision_flags", "network_flags", "document_flags"]:
        merged[col] = merged[col].apply(lambda v: v if isinstance(v, list) else [])

    merged["composite_score"] = (
        merged["financial_score"] * RISK_WEIGHTS["financial"]
        + merged["cost_benchmark_score"] * RISK_WEIGHTS["cost_benchmark"]
        + merged["geo_photo_score"] * RISK_WEIGHTS["geo_photo"]
        + merged["contractor_network_score"] * RISK_WEIGHTS["contractor_network"]
        + merged["document_score"] * RISK_WEIGHTS["document"]
    ).round(1)

    bands = merged["composite_score"].apply(risk_band)
    merged["risk_band"] = bands.apply(lambda t: t[0])
    merged["risk_color"] = bands.apply(lambda t: t[1])

    def build_explanation(row):
        all_flags = (
            row["financial_flags"] + row["geo_vision_flags"]
            + row["network_flags"] + row["document_flags"]
        )
        if not all_flags:
            return "No significant irregularities detected across financial, geo-visual, contractor-network or document checks."
        bullets = [f"- {_explain_flag(f)}" for f in all_flags]
        return "\n".join(bullets)

    merged["explanation"] = merged.apply(build_explanation, axis=1)
    merged["scored_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    merged.index.name = "project_id"
    return merged.reset_index()
