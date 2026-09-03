"""
Cost-benchmark engine.

Independent of the multivariate financial engine, this asks a narrower
and more interpretable question an auditor would ask by hand: "does this
project's cost look right *for this category, in this district*, compared
to its peers?" Uses a robust (median/IQR-based) z-score rather than
mean/std so a handful of genuinely expensive large projects don't distort
the benchmark for everyone else.
"""
import numpy as np
import pandas as pd


def _robust_z(series: pd.Series) -> pd.Series:
    median = series.median()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = max(q3 - q1, 1e-6)
    # 0.7413 scales IQR to be comparable to a normal std-dev
    robust_std = iqr * 0.7413
    return (series - median) / max(robust_std, 1e-6)


def run_cost_benchmark_engine(projects_df: pd.DataFrame) -> pd.DataFrame:
    df = projects_df.copy()
    df["peer_group"] = df["category"] + "|" + df["district"]

    # Fall back to category-only benchmark where a district has too few
    # peer projects for a stable statistic.
    peer_counts = df.groupby("peer_group")["project_id"].transform("count")
    df.loc[peer_counts < 5, "peer_group"] = df["category"]

    df["robust_z"] = df.groupby("peer_group")["sanctioned_amount"].transform(_robust_z)
    df["peer_median_cost"] = df.groupby("peer_group")["sanctioned_amount"].transform("median")
    df["pct_above_peer_median"] = (
        (df["sanctioned_amount"] - df["peer_median_cost"]) / df["peer_median_cost"] * 100
    )

    # Map |z| onto 0-100: z=0 -> 0, z>=4 -> 100, only positive (over-cost) deviations penalised
    score = np.clip(df["robust_z"], 0, None) / 4.0 * 100
    df["cost_benchmark_score"] = np.clip(score, 0, 100).round(1)

    return df.set_index("project_id")[[
        "cost_benchmark_score", "robust_z", "peer_median_cost", "pct_above_peer_median", "peer_group",
    ]]
