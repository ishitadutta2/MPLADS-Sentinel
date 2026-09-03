"""
Contractor Network — visualises the real NetworkX graph of contractors
linked by shared registration address, bank account prefix, or director
name, so a reviewer can see suspected cartel clusters directly rather
than trusting a single score.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from common import (
    get_scored_dataset, get_raw_tables, inject_base_style, require_login, show_user_badge, apply_plot_theme,
    get_contractor_graph, get_contractor_network_scores,
)

st.set_page_config(page_title="Contractor Network — MPLADS Sentinel", page_icon="🕸️", layout="wide")
inject_base_style()
user = require_login()
show_user_badge()
st.title("🕸️ Contractor Network Analysis")
st.caption(
    "Contractors sharing a registration address, bank-account prefix, or director name are linked — "
    "the classic proxy signals for shell-company cartels bidding against 'each other' to fake competition. "
    "Built with a real NetworkX graph over the contractor registry."
)

df = get_scored_dataset()
raw = get_raw_tables()
contractors = raw["contractors"]

G = get_contractor_graph(contractors)
network_scores, _ = get_contractor_network_scores(df, contractors)

components = [c for c in nx.connected_components(G) if len(c) >= 2]
components.sort(key=len, reverse=True)

c1, c2, c3 = st.columns(3)
c1.metric("Contractors Tracked", len(contractors))
c2.metric("Linked Clusters Found", len(components))
c3.metric("Contractors in a Cluster", sum(len(c) for c in components))

st.divider()
st.subheader("Suspected Cartel Clusters")

if not components:
    st.success("No linked contractor clusters detected.")
else:
    cluster_choice = st.selectbox(
        "Select a cluster to inspect",
        range(len(components)),
        format_func=lambda i: f"Cluster {i+1} — {len(components[i])} contractors",
    )
    cluster_nodes = list(components[cluster_choice])
    subG = G.subgraph(cluster_nodes)

    # Layout + Plotly network render
    pos = nx.spring_layout(subG, seed=42, k=0.9)
    edge_x, edge_y = [], []
    for a, b in subG.edges():
        x0, y0 = pos[a]; x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color="#888"), mode="lines", hoverinfo="none")

    node_x, node_y, node_text, node_size = [], [], [], []
    name_lookup = contractors.set_index("contractor_id")["contractor_name"].to_dict()
    for node in subG.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        deg = subG.degree(node)
        node_size.append(18 + deg * 6)
        reasons = set()
        for _, _, edata in subG.edges(node, data=True):
            reasons |= edata.get("reasons", set())
        node_text.append(f"{name_lookup.get(node, node)}<br>{node}<br>Links: {', '.join(reasons)}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=[name_lookup.get(n, n)[:18] for n in subG.nodes()],
        textposition="top center",
        hovertext=node_text, hoverinfo="text",
        marker=dict(size=node_size, color="#c62828", line=dict(width=2, color="white")),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, height=500, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(showgrid=False, zeroline=False, visible=False),
                       yaxis=dict(showgrid=False, zeroline=False, visible=False))
    fig = apply_plot_theme(fig)
    st.plotly_chart(fig, width='stretch')

    st.markdown("**Contractors in this cluster:**")
    cluster_df = contractors[contractors["contractor_id"].isin(cluster_nodes)][
        ["contractor_id", "contractor_name", "registration_address", "bank_account", "director_name"]
    ]
    st.dataframe(cluster_df, width='stretch', hide_index=True)

    st.markdown("**Projects awarded to this cluster:**")
    cluster_projects = df[df["contractor_id"].isin(cluster_nodes)][
        ["project_id", "mp_name", "category", "sanctioned_amount", "composite_score", "risk_band"]
    ].sort_values("composite_score", ascending=False)
    st.dataframe(cluster_projects, width='stretch', hide_index=True)
    st.metric("Total Value Awarded to This Cluster", f"₹{cluster_projects['sanctioned_amount'].sum():,.0f}")
