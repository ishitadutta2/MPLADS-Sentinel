"""
Contractor network engine.

Builds a real graph (NetworkX) linking contractors that share a
registration address, bank account prefix, or director name — the
standard proxy signals investigators use to spot shell-company cartels
bidding against "each other" to simulate competitive tendering while
actually being the same beneficial owner.

Graph metrics computed (all real, on the actual graph object):
  - connected component size (how big is the suspected cartel cluster)
  - degree centrality (how many shared-identity links does this
    contractor have)
  - share of that cluster's total project value awarded to a single MP's
    constituency (concentration risk — one MP repeatedly funnelling work
    to one cluster)

These roll up into a per-project `contractor_network_score`.
"""
import networkx as nx
import pandas as pd


def build_contractor_graph(contractors_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in contractors_df.iterrows():
        G.add_node(row["contractor_id"], name=row["contractor_name"])

    # Link contractors sharing a registration address, bank prefix, or director
    by_address = contractors_df.groupby("registration_address")["contractor_id"].apply(list)
    by_director = contractors_df.groupby("director_name")["contractor_id"].apply(list)
    # bank account "prefix" = everything before the last 6 digits (branch/product code)
    contractors_df = contractors_df.copy()
    contractors_df["bank_prefix"] = contractors_df["bank_account"].str[:-6]
    by_bank = contractors_df.groupby("bank_prefix")["contractor_id"].apply(list)

    for grouping, reason in [(by_address, "shared_address"), (by_director, "shared_director"), (by_bank, "shared_bank_prefix")]:
        for ids in grouping:
            if len(ids) < 2:
                continue
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if G.has_edge(ids[i], ids[j]):
                        G[ids[i]][ids[j]]["reasons"].add(reason)
                    else:
                        G.add_edge(ids[i], ids[j], reasons={reason})

    return G


def run_contractor_network_engine(
    projects_df: pd.DataFrame, contractors_df: pd.DataFrame
) -> pd.DataFrame:
    G = build_contractor_graph(contractors_df)

    components = list(nx.connected_components(G))
    cluster_of = {}
    for comp in components:
        if len(comp) < 2:
            continue
        for node in comp:
            cluster_of[node] = comp

    degree = dict(G.degree())

    # per-contractor project value + constituency concentration
    proj_by_contractor = projects_df.groupby("contractor_id").agg(
        n_projects=("project_id", "count"),
        total_value=("sanctioned_amount", "sum"),
    )
    mp_concentration = projects_df.groupby(["contractor_id", "mp_id"]).size().reset_index(name="n")
    top_mp_share = mp_concentration.sort_values("n", ascending=False).drop_duplicates("contractor_id").set_index("contractor_id")

    contractor_scores = {}
    for cid in contractors_df["contractor_id"]:
        cluster = cluster_of.get(cid)
        cluster_size = len(cluster) if cluster else 1
        deg = degree.get(cid, 0)

        score = 0.0
        reasons = []
        if cluster_size >= 3:
            score += min((cluster_size - 1) * 12, 55)
            reasons.append(f"linked_cluster_of_{cluster_size}_contractors")
        if deg >= 2:
            score += min(deg * 5, 20)
            reasons.append(f"shared_identity_links:{deg}")

        if cluster and cid in proj_by_contractor.index:
            cluster_value = proj_by_contractor.reindex(list(cluster)).total_value.sum()
            if cluster_value > 0:
                own_share = proj_by_contractor.loc[cid, "total_value"] / cluster_value
                if len(cluster) >= 3 and proj_by_contractor.reindex(list(cluster)).n_projects.sum() >= 5:
                    score += 15
                    reasons.append("active_cartel_cluster_with_multiple_awards")

        if cid in top_mp_share.index and cid in proj_by_contractor.index:
            n_total = proj_by_contractor.loc[cid, "n_projects"]
            n_top_mp = top_mp_share.loc[cid, "n"]
            if n_total >= 3 and (n_top_mp / n_total) > 0.8:
                score += 10
                reasons.append("high_repeat_award_concentration_single_mp")

        contractor_scores[cid] = {
            "contractor_network_score": round(min(score, 100), 1),
            "cluster_size": cluster_size,
            "graph_degree": deg,
            "network_flags": reasons,
        }

    contractor_score_df = pd.DataFrame.from_dict(contractor_scores, orient="index")

    result = projects_df[["project_id", "contractor_id"]].merge(
        contractor_score_df, left_on="contractor_id", right_index=True, how="left"
    ).set_index("project_id")

    return result[["contractor_network_score", "cluster_size", "graph_degree", "network_flags"]], G
