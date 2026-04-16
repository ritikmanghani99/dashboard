import os
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google import genai
from google.genai import types
# import google.generativeai as genai

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Affinius Capital — Lease-Up Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        border-left: 4px solid #1565C0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1565C0; }
    .metric-label { font-size: 0.85rem; color: #555; margin-top: 4px; }
    .insight-box {
        background: #e8f5e9;
        border-left: 4px solid #2E7D32;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 12px;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .query-box {
        background: #e3f2fd;
        border-left: 4px solid #1565C0;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 12px;
        font-size: 0.9rem;
        font-family: monospace;
    }
    .warning-box {
        background: #fff8e1;
    border-left: 4px solid #F9A825;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.85rem;
    word-wrap: break-word;
    overflow-wrap: break-word;
    overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ── Gemini setup ──────────────────────────────────────────────────────────────
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCdb_1rXtwvCwQTCRe56dSiuw9mULMO5h8")
# genai.configure(api_key=GEMINI_API_KEY)
# gemini = genai.GenerativeModel("gemini-2.0-flash-lite")

from dotenv import load_dotenv
load_dotenv()  # loads .env file automatically

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")

if not GEMINI_API_KEY:
    st.error("No Gemini API key found. Add GEMINI_API_KEY to your .env file.")
    st.stop()
    
gemini = genai.Client(api_key=GEMINI_API_KEY)
# genai.configure(api_key=GEMINI_API_KEY)
# gemini = genai.GenerativeModel(GEMINI_MODEL)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = "dashboard_data"
    af   = pd.read_csv(f"{base}/all_features.csv")
    lu   = pd.read_csv(f"{base}/leaseup_results.csv")
    em   = pd.read_csv(f"{base}/embed_df.csv")
    nr   = pd.read_csv(f"{base}/neg_rent_results.csv")
    # Ensure delivery_month is datetime in leaseup
    lu["delivery_month"] = pd.to_datetime(lu["delivery_month"], errors="coerce")
    lu["stabilization_month"] = pd.to_datetime(lu["stabilization_month"], errors="coerce")
    return af, lu, em, nr

all_features, leaseup_df, embed_df, neg_rent_df = load_data()
stab = all_features[all_features["stabilized"] == True].copy()

CLUSTER_LABELS = {
    0: "Premium Submarkets (A/A- grade)",
    1: "Established Premium",
    2: "Mid-Tier Steady (B grade)",
    3: "Akron / Small Market",
}
CLUSTER_COLORS = {0: "#E91E63", 1: "#1565C0", 2: "#2E7D32", 3: "#E65100"}
ERA_COLORS = {
    "Pre-GFC":  "#9C27B0",
    "Recovery": "#FF9800",
    "Expansion":"#1565C0",
    "COVID":    "#F44336",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/building.png", width=60)
    st.markdown("## Affinius Capital\n### Lease-Up Dashboard")
    st.markdown("---")
    st.markdown("**Period:** Apr 2008 – Sep 2020  \n**Markets:** Austin TX · Akron OH")
    st.markdown("---")

    # Global market filter
    market_filter = st.multiselect(
        "Filter by Market",
        options=all_features["market"].unique().tolist(),
        default=all_features["market"].unique().tolist(),
    )
    era_filter = st.multiselect(
        "Filter by Delivery Era",
        options=["Pre-GFC", "Recovery", "Expansion", "COVID"],
        default=["Pre-GFC", "Recovery", "Expansion", "COVID"],
    )
    # st.markdown("---")
    # st.caption("Powered by Google Gemini 2.0 Flash")

# Apply global filters
af_f = all_features[
    all_features["market"].isin(market_filter) &
    all_features["delivery_era"].isin(era_filter)
]
stab_f = af_f[af_f["stabilized"] == True].copy()
em_f   = embed_df[
    embed_df["market"].isin(market_filter) &
    embed_df["delivery_era"].isin(era_filter)
]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Overview",
    "🔍 Feature Explorer",
    "🗂 Clustering",
    "🏠 Property Lookup",
    "💬 Ask the Data",
])

# =============================================================================
# TAB 1 — MARKET OVERVIEW
# =============================================================================
with tab1:
    st.markdown("## Market Overview")
    st.caption("Key lease-up metrics across delivered properties (2008–2020).")

    # KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)
    delivered   = len(af_f)
    stabilized  = len(stab_f)
    avg_lu      = stab_f["leaseup_months"].mean()
    med_lu      = stab_f["leaseup_months"].median()
    neg_pct     = (neg_rent_df[neg_rent_df["market"].isin(market_filter)]["negative_growth"].sum() /
                   max(len(neg_rent_df[neg_rent_df["market"].isin(market_filter)]), 1) * 100)

    for col, val, label in [
        (col1, f"{delivered}",       "Delivered Properties"),
        (col2, f"{stabilized}",      "Stabilized (>=90% Occ)"),
        (col3, f"{avg_lu:.1f} mo",   "Avg Lease-Up Time"),
        (col4, f"{med_lu:.0f} mo",   "Median Lease-Up Time"),
        (col5, f"{neg_pct:.0f}%",    "Negative Rent Growth"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("###")

    col_a, col_b = st.columns(2)

    # Delivery volume by year
    with col_a:
        lu_f = leaseup_df[leaseup_df["market"].isin(market_filter)].copy()
        lu_f = lu_f.dropna(subset=["delivery_month"])
        lu_f["delivery_year"] = lu_f["delivery_month"].dt.year
        vol = lu_f.groupby(["delivery_year", "market"]).size().reset_index(name="count")
        fig_vol = px.bar(
            vol, x="delivery_year", y="count", color="market",
            barmode="group",
            color_discrete_map={
                "Austin-Round Rock, TX": "#1565C0",
                "Akron, OH": "#E65100",
            },
            labels={"delivery_year": "Year", "count": "Properties Delivered", "market": "Market"},
            title="Annual Delivery Volume by Market",
        )
        fig_vol.update_layout(legend=dict(orientation="h", y=-0.2), height=350)
        st.plotly_chart(fig_vol, use_container_width=True)

    # Lease-up distribution
    with col_b:
        fig_dist = px.histogram(
            stab_f, x="leaseup_months", color="market",
            barmode="overlay", nbins=20, opacity=0.72,
            color_discrete_map={
                "Austin-Round Rock, TX": "#1565C0",
                "Akron, OH": "#E65100",
            },
            labels={"leaseup_months": "Months to Stabilization", "market": "Market"},
            title="Lease-Up Duration Distribution",
        )
        # Add mean lines
        for mkt, color in [("Austin-Round Rock, TX", "#1565C0"), ("Akron, OH", "#E65100")]:
            sub = stab_f[stab_f["market"] == mkt]["leaseup_months"].dropna()
            if len(sub) > 0:
                fig_dist.add_vline(
                    x=sub.mean(), line_dash="dash", line_color=color,
                    annotation_text=f"{mkt.split(',')[0]} mean={sub.mean():.1f}mo",
                    annotation_position="top right", annotation_font_size=10,
                )
        fig_dist.update_layout(legend=dict(orientation="h", y=-0.2), height=350)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Lease-up over time (Austin only — Akron too few)
    st.markdown("### Average Lease-Up Time by Delivery Year")
    st.caption("Reveals how the supply cycle affected absorption. Buildings delivered at the 2015-2017 peak faced the most competition.")
    austin_stab = stab_f[stab_f["market"] == "Austin-Round Rock, TX"].copy()
    if len(austin_stab) > 0:
        austin_stab["delivery_year"] = pd.to_datetime(
            leaseup_df[leaseup_df["ProjID"].isin(austin_stab["ProjID"])]["delivery_month"].values,
            errors="coerce"
        )
        # Merge delivery year from leaseup_df
        lu_austin = leaseup_df[leaseup_df["market"] == "Austin-Round Rock, TX"].copy()
        lu_austin["delivery_year"] = lu_austin["delivery_month"].dt.year
        lu_austin_stab = lu_austin[lu_austin["stabilized"] == True].dropna(subset=["delivery_year", "leaseup_months"])
        yearly = lu_austin_stab.groupby("delivery_year")["leaseup_months"].agg(
            mean="mean", median="median", count="count"
        ).reset_index()

        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=yearly["delivery_year"], y=yearly["mean"],
            mode="lines+markers", name="Mean lease-up",
            line=dict(color="#1565C0", width=2.5),
            marker=dict(size=8),
        ))
        fig_time.add_trace(go.Scatter(
            x=yearly["delivery_year"], y=yearly["median"],
            mode="lines+markers", name="Median lease-up",
            line=dict(color="#1565C0", width=1.5, dash="dash"),
            marker=dict(size=6),
        ))
        # Era shading
        for era, x0, x1, color in [
            ("Pre-GFC", 2007.5, 2008.5, "rgba(156,39,176,0.07)"),
            ("Recovery", 2008.5, 2013.5, "rgba(255,152,0,0.07)"),
            ("Expansion", 2013.5, 2019.5, "rgba(21,101,192,0.07)"),
            ("COVID", 2019.5, 2020.5, "rgba(244,67,54,0.07)"),
        ]:
            fig_time.add_vrect(
                x0=x0, x1=x1, fillcolor=color, line_width=0,
                annotation_text=era, annotation_position="top left",
                annotation_font_size=10,
            )
        fig_time.update_layout(
            height=350, xaxis_title="Delivery Year",
            yaxis_title="Avg Lease-Up (months)",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_time, use_container_width=True)

# =============================================================================
# TAB 2 — FEATURE EXPLORER
# =============================================================================
with tab2:
    st.markdown("## Feature Explorer")
    st.caption("Explore how each engineered feature relates to lease-up time. Use the controls to filter and compare.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        x_feature = st.selectbox("X-axis feature", options=[
            "AreaPerUnit", "occ_velocity", "rent_gap_vs_submarket",
            "log_quantity", "concession_burn_rate",
        ], format_func=lambda x: {
            "AreaPerUnit":            "Unit Size (sq ft)",
            "occ_velocity":           "Occupancy Velocity (first 3mo)",
            "rent_gap_vs_submarket":  "Rent Gap vs Submarket ($)",
            "log_quantity":           "Project Size (log units)",
            "concession_burn_rate":   "Concession Burn Rate",
        }[x])
    with col_ctrl2:
        color_by = st.selectbox("Color by", ["delivery_era", "market", "delivery_season"])
    with col_ctrl3:
        size_by = st.selectbox("Dot size by", ["uniform", "Quantity", "leaseup_months"])

    plot_df = stab_f.dropna(subset=[x_feature, "leaseup_months"]).copy()
    size_col = None if size_by == "uniform" else size_by

    # Correlation
    r = plot_df[[x_feature, "leaseup_months"]].corr().iloc[0, 1]
    st.markdown(f"**Pearson r = {r:.3f}** between `{x_feature}` and lease-up months  "
                f"({'negative = faster lease-up as feature increases' if r < 0 else 'positive = slower lease-up as feature increases'})")

    fig_feat = px.scatter(
        plot_df, x=x_feature, y="leaseup_months",
        color=color_by,
        color_discrete_map=ERA_COLORS if color_by == "delivery_era" else None,
        size=size_col,
        hover_data=["Name", "Submarket", "market", "leaseup_months", x_feature],
        # trendline="ols",
        labels={
            x_feature: {
                "AreaPerUnit":           "Avg Unit Size (sq ft)",
                "occ_velocity":          "Occ Velocity (avg monthly gain, first 3mo)",
                "rent_gap_vs_submarket": "Rent Gap vs Submarket at Delivery ($)",
                "log_quantity":          "Project Size (log scale)",
                "concession_burn_rate":  "Concession Burn Rate (MoM change)",
            }[x_feature],
            "leaseup_months": "Lease-Up Time (months)",
        },
        title=f"{x_feature} vs Lease-Up Time",
        height=480,
    )
    fig_feat.update_traces(marker=dict(opacity=0.72, line=dict(width=0.4, color="white")))
    st.plotly_chart(fig_feat, use_container_width=True)

    # Summary stats by color group
    st.markdown("### Group Summary")
    grp = (
        plot_df.groupby(color_by)["leaseup_months"]
        .agg(Count="count", Mean="mean", Median="median", Std="std")
        .round(1)
        .reset_index()
        .sort_values("Mean")
    )
    st.dataframe(grp, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3 — CLUSTERING
# =============================================================================
with tab3:
    st.markdown("## Property Clusters")
    st.caption(
        "KMeans k=4 on the scaled feature matrix. t-SNE reduces to 2D for visualization. "
        "Silhouette score = 0.15 (typical for real-world multifamily data — lease-up is a continuum, not discrete buckets)."
    )

    col_t1, col_t2 = st.columns([3, 2])

    with col_t1:
        color_tsne = st.radio(
            "Color t-SNE by",
            ["Cluster", "Delivery Era", "Lease-Up Duration"],
            horizontal=True,
        )

        if color_tsne == "Cluster":
            em_f["cluster_label"] = em_f["cluster"].map(CLUSTER_LABELS)
            fig_tsne = px.scatter(
                em_f, x="tsne_x", y="tsne_y",
                color="cluster_label",
                color_discrete_sequence=list(CLUSTER_COLORS.values()),
                hover_data=["Name", "market", "delivery_era", "leaseup_months"],
                labels={"tsne_x": "t-SNE 1", "tsne_y": "t-SNE 2", "cluster_label": "Cluster"},
                title="t-SNE — Cluster Membership",
                height=450,
            )
        elif color_tsne == "Delivery Era":
            fig_tsne = px.scatter(
                em_f, x="tsne_x", y="tsne_y",
                color="delivery_era",
                color_discrete_map=ERA_COLORS,
                hover_data=["Name", "market", "leaseup_months"],
                labels={"tsne_x": "t-SNE 1", "tsne_y": "t-SNE 2"},
                title="t-SNE — Delivery Era Overlay",
                height=450,
            )
        else:
            stab_em = em_f[em_f["stabilized"] == True]
            nonstab_em = em_f[em_f["stabilized"] == False]
            fig_tsne = go.Figure()
            fig_tsne.add_trace(go.Scatter(
                x=nonstab_em["tsne_x"], y=nonstab_em["tsne_y"],
                mode="markers",
                marker=dict(color="lightgray", size=6, symbol="x"),
                name="Not stabilized",
                hovertemplate="%{text}<extra></extra>",
                text=nonstab_em["Name"],
            ))
            fig_tsne.add_trace(go.Scatter(
                x=stab_em["tsne_x"], y=stab_em["tsne_y"],
                mode="markers",
                marker=dict(
                    color=stab_em["leaseup_months"],
                    colorscale="RdYlGn_r",
                    cmin=1, cmax=36,
                    size=8,
                    colorbar=dict(title="Months"),
                    line=dict(width=0.4, color="white"),
                ),
                name="Stabilized",
                hovertemplate="%{text}: %{marker.color:.0f} mo<extra></extra>",
                text=stab_em["Name"],
            ))
            fig_tsne.update_layout(
                title="t-SNE — Lease-Up Duration Heatmap",
                xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
                height=450,
            )

        fig_tsne.update_traces(
            marker=dict(opacity=0.82, line=dict(width=0.4, color="white"))
            if color_tsne != "Lease-Up Duration" else {}
        )
        st.plotly_chart(fig_tsne, use_container_width=True)

    with col_t2:
        st.markdown("### Cluster Profiles")
        profile_cols = ["occ_velocity", "rent_gap_vs_submarket", "grade_score", "log_quantity"]
        c_means = em_f.groupby("cluster")[profile_cols].mean().round(3)
        c_means.index = [CLUSTER_LABELS.get(i, f"C{i}") for i in c_means.index]
        c_means.columns = ["Occ Velocity", "Rent Gap ($)", "Grade Score", "Log Units"]
        st.dataframe(c_means, use_container_width=True)

        st.markdown("### Lease-Up by Cluster")
        lu_cluster = (
            em_f[em_f["stabilized"] == True]
            .groupby("cluster")["leaseup_months"]
            .agg(n="count", mean="mean", median="median")
            .round(1)
            .reset_index()
        )
        lu_cluster["Cluster"] = lu_cluster["cluster"].map(CLUSTER_LABELS)
        lu_cluster = lu_cluster[["Cluster", "n", "mean", "median"]]
        lu_cluster.columns = ["Cluster", "n", "Avg (mo)", "Median (mo)"]
        st.dataframe(lu_cluster, use_container_width=True, hide_index=True)

    # Properties table filtered by cluster click
    st.markdown("### Browse Properties by Cluster")
    selected_cluster = st.selectbox(
        "Select cluster",
        options=list(CLUSTER_LABELS.keys()),
        format_func=lambda x: f"Cluster {x}: {CLUSTER_LABELS[x]}",
    )
    cluster_props = em_f[em_f["cluster"] == selected_cluster][
        ["Name", "market", "Submarket", "delivery_era", "leaseup_months",
         "occ_velocity", "rent_gap_vs_submarket", "AreaPerUnit"]
    ].rename(columns={
        "leaseup_months":         "Lease-Up (mo)",
        "occ_velocity":           "Occ Velocity",
        "rent_gap_vs_submarket":  "Rent Gap ($)",
        "AreaPerUnit":            "Unit Size (sqft)",
    }).sort_values("Lease-Up (mo)")
    st.dataframe(cluster_props, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4 — PROPERTY LOOKUP  (GenAI Feature 1)
# =============================================================================
with tab4:
    st.markdown("## Property Lookup & AI Insight")
    st.markdown(
        '<div class="warning-box">🤖 <b>GenAI Feature 1:</b> Select any delivered property '
        'and Gemini 2.5 Flash Lite generates a 3-sentence analyst narrative explaining its '
        'lease-up performance based on its feature profile.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("###")

    # Property selector
    prop_names = sorted(all_features["Name"].dropna().unique().tolist())
    selected_prop = st.selectbox("Select a property", prop_names)

    prop_row = all_features[all_features["Name"] == selected_prop].iloc[0]
    em_row   = embed_df[embed_df["Name"] == selected_prop]
    cluster_id = int(em_row["cluster"].values[0]) if len(em_row) > 0 else None

    # Property metrics
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    metrics = [
        (col_p1, "Market",         str(prop_row.get("market", "N/A")).split(",")[0]),
        (col_p2, "Submarket",      str(prop_row.get("Submarket", "N/A"))),
        (col_p3, "Lease-Up Time",  f"{prop_row.get('leaseup_months', 'N/A'):.0f} mo"
                                   if pd.notna(prop_row.get("leaseup_months")) else "Censored"),
        (col_p4, "Delivery Era",   str(prop_row.get("delivery_era", "N/A"))),
    ]
    for col, label, val in metrics:
        col.metric(label, val)

    col_p5, col_p6, col_p7, col_p8 = st.columns(4)
    metrics2 = [
        (col_p5, "Occ Velocity",    f"{prop_row.get('occ_velocity', np.nan):.3f}"
                                    if pd.notna(prop_row.get("occ_velocity")) else "N/A"),
        (col_p6, "Rent Gap vs Submarket", f"${prop_row.get('rent_gap_vs_submarket', np.nan):.0f}"
                                          if pd.notna(prop_row.get("rent_gap_vs_submarket")) else "N/A"),
        (col_p7, "Unit Size",       f"{prop_row.get('AreaPerUnit', np.nan):.0f} sqft"
                                    if pd.notna(prop_row.get("AreaPerUnit")) else "N/A"),
        (col_p8, "Cluster",         CLUSTER_LABELS.get(cluster_id, "N/A") if cluster_id is not None else "N/A"),
    ]
    for col, label, val in metrics2:
        col.metric(label, val)

    st.markdown("---")

    # Radar chart — property vs cluster average
    if cluster_id is not None:
        radar_features = ["occ_velocity", "rent_gap_vs_submarket", "grade_score",
                          "log_quantity", "AreaPerUnit"]
        radar_labels   = ["Occ Velocity", "Rent Gap ($)", "Grade Score",
                          "Log Units", "Unit Size"]
        cluster_avg = embed_df[embed_df["cluster"] == cluster_id][radar_features].mean()
        prop_vals   = prop_row[radar_features]

        # Normalize 0-1 across the full dataset for fair comparison
        feat_min = all_features[radar_features].min()
        feat_max = all_features[radar_features].max()
        prop_norm    = ((prop_vals - feat_min) / (feat_max - feat_min + 1e-9)).fillna(0).tolist()
        cluster_norm = ((cluster_avg - feat_min) / (feat_max - feat_min + 1e-9)).fillna(0).tolist()

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=prop_norm + [prop_norm[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself", name=selected_prop[:30],
            line=dict(color="#1565C0"), fillcolor="rgba(21,101,192,0.15)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=cluster_norm + [cluster_norm[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself", name=f"Cluster avg",
            line=dict(color="#E91E63", dash="dash"), fillcolor="rgba(233,30,99,0.08)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"{selected_prop[:40]} vs Cluster Average (normalized)",
            height=380, showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # AI Insight generation
    st.markdown("### AI-Generated Analyst Insight")
    if st.button("Generate Insight with Gemini", type="primary"):
        with st.spinner("Gemini is analyzing this property..."):
            # Build prompt
            lu_time = (f"{prop_row['leaseup_months']:.0f} months"
                       if pd.notna(prop_row.get("leaseup_months")) else "did not stabilize within the observation window")
            occ_vel = (f"{prop_row['occ_velocity']:.3f} (monthly avg occupancy gain in first 3 months)"
                       if pd.notna(prop_row.get("occ_velocity")) else "unavailable")
            rent_gap = (f"${prop_row['rent_gap_vs_submarket']:.0f} {'above' if prop_row['rent_gap_vs_submarket'] > 0 else 'below'} submarket median"
                        if pd.notna(prop_row.get("rent_gap_vs_submarket")) else "unavailable")
            unit_sz  = (f"{prop_row['AreaPerUnit']:.0f} sqft"
                        if pd.notna(prop_row.get("AreaPerUnit")) else "unavailable")
            units    = (f"{prop_row['Quantity']:.0f} units"
                        if pd.notna(prop_row.get("Quantity")) else "unavailable")

            prompt = f"""You are a senior real estate analyst at a multifamily investment firm.
Write exactly 3 sentences analyzing this property's lease-up performance.
Be specific, reference the numbers, and draw on real estate fundamentals.
Do not use bullet points. Do not hedge excessively. Sound like a confident analyst, not a chatbot.

Property: {selected_prop}
Market: {prop_row.get('market', 'N/A')}
Submarket: {prop_row.get('Submarket', 'N/A')}
Delivered: {prop_row.get('delivery_era', 'N/A')} era, {prop_row.get('delivery_season', 'N/A')} season
Project size: {units}
Avg unit size: {unit_sz}
Lease-up time: {lu_time}
Occupancy velocity: {occ_vel}
Rent gap vs submarket at delivery: {rent_gap}
Cluster: {CLUSTER_LABELS.get(cluster_id, 'N/A')}

Market context: Austin average lease-up is 12.9 months. 42% of Austin buildings
reached 90% occupancy by cutting effective rent. The Expansion era (2014-2019)
had the most supply competition. Early occupancy velocity is the strongest
predictor of final stabilization speed (r = -0.29).

Write 3 sentences. No headers, no bullet points."""

            try:
                response = gemini.models.generate_content(
    model=GEMINI_MODEL,
    contents=prompt
)
                # response = gemini.generate_content(prompt)
                insight  = response.text.strip()
                st.markdown(
                    f'<div class="insight-box">🏢 <b>Gemini Analysis:</b><br><br>{insight}</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Gemini API error: {e}")

    # Prompt engineering documentation (collapsible)
    with st.expander("Prompt Engineering Documentation"):
        st.markdown("""
**Model:** `gemini-2.5-flash`

**Prompt design decisions:**
1. **Role priming** ("senior real estate analyst") — constrains the tone and prevents generic chatbot hedging
2. **Explicit format constraint** ("exactly 3 sentences, no bullet points") — ensures the output is concise and embeds cleanly in the dashboard
3. **All numeric context is injected** — the model has no access to the raw data; every number it needs is explicitly passed in the prompt
4. **Market context block** — includes the Austin averages and key correlations from Task 1 so the model can benchmark the property correctly
5. **Negative instruction** ("Do not hedge excessively, sound like a confident analyst") — without this, LLMs default to excessive qualifiers that reduce readability

**What was iterated:**
- First version had no role priming → outputs were generic and over-qualified
- Added "no markdown, no bullet points" after first run returned formatted output
- Added the market context block after early outputs failed to benchmark against Austin average
        """)

# =============================================================================
# TAB 5 — NATURAL LANGUAGE QUERY  (GenAI Feature 2)
# =============================================================================
with tab5:
    st.markdown("## Ask the Data")
    st.markdown(
        '<div class="warning-box">🤖 <b>GenAI Feature 2:</b> Type a question in plain English. '
        'Gemini translates it into a pandas filter, executes it on the dataset, '
        'and shows you the results. The generated code is always displayed so you can verify it.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("###")

    # Column schema for the prompt
    SCHEMA = """
DataFrame name: df
Columns and types:
- Name (str): property name
- market (str): 'Austin-Round Rock, TX' or 'Akron, OH'
- Submarket (str): submarket name e.g. 'North Central Austin', 'Round Rock/Georgetown'
- leaseup_months (float): months from delivery to 90% occupancy. NaN = not stabilized
- stabilized (bool): True if property reached 90% occupancy
- occ_velocity (float): avg monthly occupancy gain in first 3 months (0.0 to ~0.2)
- rent_gap_vs_submarket (float): property effective rent minus submarket median in $ at delivery
- AreaPerUnit (float): average unit size in sq ft
- Quantity (float): number of units
- log_quantity (float): log of Quantity
- grade_score (float): submarket grade 1=A+ best, 12=D- worst
- delivery_era (str): 'Pre-GFC', 'Recovery', 'Expansion', or 'COVID'
- delivery_season (str): 'Spring', 'Summer', 'Fall', or 'Winter'
- delivery_year: not a column — use delivery_era instead
- cluster (int): 0, 1, 2, or 3
"""

    # Example questions
    st.markdown("**Example questions you can try:**")
    ex_cols = st.columns(3)
    examples = [
        "Show properties that took more than 20 months to stabilize",
        "Show Expansion era properties with occ velocity above 0.08",
        "Find properties priced more than $200 above their submarket",
        "Show all properties in North Central Austin",
        "Which properties did not stabilize?",
        "Show large projects (more than 400 units) that stabilized in under 10 months",
    ]
    for i, (col, ex) in enumerate(zip(ex_cols * 2, examples)):
        if col.button(ex, key=f"ex_{i}"):
            st.session_state["nl_query"] = ex

    nl_query = st.text_input(
        "Your question:",
        value=st.session_state.get("nl_query", ""),
        placeholder="e.g. Show properties that took more than 18 months in the Expansion era",
    )

    if st.button("Run Query", type="primary") and nl_query.strip():
        with st.spinner("Gemini is translating your question..."):

            prompt = f"""You are a pandas data analyst. Convert the user question into a single
Python expression that filters the DataFrame `df`.

{SCHEMA}

Rules:
1. Return ONLY the Python filter expression. No explanation, no markdown, no code fences.
2. The expression must be a complete statement that can be evaluated with eval().
3. Use df[condition] syntax.
4. For string comparisons use case-insensitive matching with .str.contains() or ==
5. For 'not stabilized' use df[df['stabilized'] == False]
6. Never reference columns that do not exist in the schema above.
7. If the question mentions a submarket, use: df[df['Submarket'].str.contains('keyword', case=False, na=False)]

Examples:
Question: Show properties that took more than 20 months
Answer: df[df['leaseup_months'] > 20]

Question: Show Expansion era properties with fast early leasing
Answer: df[(df['delivery_era'] == 'Expansion') & (df['occ_velocity'] > 0.08)]

Question: Find properties priced above their submarket
Answer: df[df['rent_gap_vs_submarket'] > 0]

Now answer this question:
Question: {nl_query}
Answer:"""

            try:
                # response   = gemini.generate_content(prompt)
                response = gemini.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=prompt
                            )
                raw_code   = response.text.strip()

                # Clean up any accidental markdown fences
                raw_code = re.sub(r"```python\n?", "", raw_code)
                raw_code = re.sub(r"```\n?", "", raw_code).strip()

                # Safety: only allow filter expressions on df, no exec or imports
                safe = (
                    raw_code.startswith("df[") and
                    "import" not in raw_code and
                    "exec" not in raw_code and
                    "eval" not in raw_code and
                    "__" not in raw_code
                )

                st.markdown("**Generated filter code:**")
                st.markdown(
                    f'<div class="query-box">{raw_code}</div>',
                    unsafe_allow_html=True,
                )

                if not safe:
                    st.error("Safety check failed — the generated code looks unsafe. Try rephrasing.")
                else:
                    df = all_features.copy()   # expose as 'df' for eval
                    result = eval(raw_code)

                    if len(result) == 0:
                        st.warning("No properties matched your query. Try rephrasing or broadening the criteria.")
                    else:
                        st.success(f"Found **{len(result)} properties** matching your query.")

                        # Show results table
                        display_cols = [c for c in [
                            "Name", "market", "Submarket", "delivery_era",
                            "leaseup_months", "occ_velocity",
                            "rent_gap_vs_submarket", "AreaPerUnit", "Quantity",
                        ] if c in result.columns]
                        st.dataframe(
                            result[display_cols].round(2).reset_index(drop=True),
                            use_container_width=True,
                        )

                        # Quick chart of results
                        if "leaseup_months" in result.columns and result["leaseup_months"].notna().sum() > 1:
                            fig_res = px.histogram(
                                result.dropna(subset=["leaseup_months"]),
                                x="leaseup_months", color="delivery_era",
                                color_discrete_map=ERA_COLORS,
                                nbins=15,
                                title=f"Lease-Up Distribution for Query Results (n={len(result)})",
                                labels={"leaseup_months": "Lease-Up Months"},
                            )
                            fig_res.update_layout(height=320)
                            st.plotly_chart(fig_res, use_container_width=True)

            except Exception as e:
                st.error(f"Error executing query: {e}")
                st.info("Try rephrasing your question. Example: 'Show properties that took more than 15 months'")

    

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Affinius Capital Research Intern Assessment- Ritik Manghani | Texas A&M University "
    
)
