"""
Financial anomaly engine.

Combines three real, complementary techniques — a genuine two-model
unsupervised ENSEMBLE plus an explainable rule layer, not a single model
carrying the whole signal:

1. `sklearn.ensemble.IsolationForest` — isolates outliers by how few
   random partitions it takes to separate a point from the rest of the
   data. Good at catching points that are extreme on a few dimensions.

2. `sklearn.neighbors.LocalOutlierFactor` — a density-based detector that
   instead asks "how isolated is this point from its own local
   neighbourhood", which catches a different failure mode: a project that
   isn't extreme in the global distribution but is anomalous *relative to
   its closest peers* (e.g. unusual for its specific category/district
   even if unremarkable nationally). IsolationForest and LOF disagreeing
   on a project is itself informative — this engine averages their
   percentile ranks rather than picking one, so a project has to look
   unusual by at least one genuinely different definition of "unusual"
   to score high.

3. A small set of explainable rule-based checks (split-tendering just
   under scrutiny thresholds, expenditure exceeding sanction, single-shot
   full disbursal with no milestone installments) — these give the
   officer a human-readable *reason*, which a bare anomaly score can't.

The final `financial_score` (0-100) blends the two-model ensemble outlier
percentile with fixed increments from each triggered rule. This mirrors
how real fraud-detection stacks combine ML with hard compliance rules (ML
finds the unknown unknowns; rules encode known regulatory red flags).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

SPLIT_TENDER_THRESHOLD = 500000  # works above this typically need extra approval layers
SPLIT_TENDER_MARGIN = 0.98  # flag anything sitting within 2% below the threshold


def _build_features(projects_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    txn_agg = transactions_df.groupby("project_id").agg(
        n_installments=("transaction_id", "count"),
        txn_amount_std=("amount", "std"),
    ).reset_index()

    df = projects_df.merge(txn_agg, on="project_id", how="left")
    df["n_installments"] = df["n_installments"].fillna(1)
    df["txn_amount_std"] = df["txn_amount_std"].fillna(0)

    df["spend_ratio"] = df["expenditure"] / df["sanctioned_amount"].replace(0, np.nan)
    df["spend_ratio"] = df["spend_ratio"].fillna(1.0)

    # cost normalised within its own category (z-score-like) so a 40L
    # community hall isn't compared apples-to-oranges against a 1L borewell
    df["category_median_cost"] = df.groupby("category")["sanctioned_amount"].transform("median")
    df["cost_vs_category_ratio"] = df["sanctioned_amount"] / df["category_median_cost"]

    feature_cols = [
        "sanctioned_amount", "expenditure", "spend_ratio",
        "n_installments", "txn_amount_std", "cost_vs_category_ratio",
    ]
    return df, feature_cols


def run_financial_engine(projects_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame indexed by project_id with financial_score (0-100)
    and a list of triggered rule flags."""
    df, feature_cols = _build_features(projects_df, transactions_df)

    X = df[feature_cols].fillna(0).values
    # Standardise manually (mean/std) to keep both detectors well-behaved
    # across features on very different scales (rupees vs. counts).
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_norm = (X - mu) / sigma

    iso = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)
    iso.fit(X_norm)
    # decision_function: higher = more normal. Flip & rescale to a 0..1 percentile (1 = most anomalous).
    iso_raw = iso.decision_function(X_norm)
    iso_pct = 1 - pd.Series(iso_raw).rank(pct=True)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.15)
    lof.fit_predict(X_norm)
    # negative_outlier_factor_: closer to 0 = more normal, more negative = more anomalous.
    lof_raw = lof.negative_outlier_factor_
    lof_pct = 1 - pd.Series(lof_raw).rank(pct=True)

    ensemble_score = ((iso_pct + lof_pct) / 2 * 100)

    # --- explainable rule layer -------------------------------------------------
    flags = pd.Series([[] for _ in range(len(df))], index=df.index)
    rule_bonus = pd.Series(0.0, index=df.index)

    over_spend = df["expenditure"] > df["sanctioned_amount"] * 1.05
    flags[over_spend] = flags[over_spend].apply(lambda l: l + ["expenditure_exceeds_sanction"])
    rule_bonus[over_spend] += 15

    split_tender = (df["sanctioned_amount"] >= SPLIT_TENDER_THRESHOLD * SPLIT_TENDER_MARGIN) & \
                    (df["sanctioned_amount"] < SPLIT_TENDER_THRESHOLD)
    flags[split_tender] = flags[split_tender].apply(lambda l: l + ["possible_split_tendering"])
    rule_bonus[split_tender] += 20

    single_shot_high_value = (df["n_installments"] <= 1) & (df["sanctioned_amount"] > df["category_median_cost"] * 1.5)
    flags[single_shot_high_value] = flags[single_shot_high_value].apply(lambda l: l + ["single_installment_high_value"])
    rule_bonus[single_shot_high_value] += 10

    cost_outlier = df["cost_vs_category_ratio"] > 1.8
    flags[cost_outlier] = flags[cost_outlier].apply(lambda l: l + ["cost_far_above_category_median"])
    rule_bonus[cost_outlier] += 15

    final_score = np.clip(ensemble_score * 0.6 + rule_bonus, 0, 100)

    result = pd.DataFrame({
        "project_id": df["project_id"],
        "financial_score": final_score.round(1),
        "financial_flags": flags,
        "cost_vs_category_ratio": df["cost_vs_category_ratio"].round(2),
        "spend_ratio": df["spend_ratio"].round(2),
        "isolation_forest_percentile": iso_pct.round(3),
        "lof_percentile": lof_pct.round(3),
    })
    return result.set_index("project_id")
