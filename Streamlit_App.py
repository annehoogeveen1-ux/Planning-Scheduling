from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

# Force the script to restart via Streamlit if started directly with python
if "streamlit" not in sys.modules and "-m" not in sys.argv:
    if not os.environ.get("STREAMLIT_ALREADY_RUNNING"):
        os.environ["STREAMLIT_ALREADY_RUNNING"] = "1"
        print("Starting dashboard in browser via Streamlit...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", __file__,
            "--browser.gatherUsageStats=False",
            "--server.headless=false",
        ])
        sys.exit()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pydeck as pdk
import copy

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from Rolling_Horizon_Allocation import (
    Patient,
    Provider,
    generate_weeks,
    haversine_km,
    rolling_horizon_assignment,
)

# =========================
# Config
# =========================

st.set_page_config(page_title="Isala Tactical Dashboard", layout="wide")
px.defaults.template = "plotly_white"

DEFAULT_PATIENTS_FILE = PROJECT_DIR / "patients.csv"
DEFAULT_PROVIDERS_FILE = PROJECT_DIR / "providers.csv"

STRATEGY_OPTIONS = {
    "Greedy - Heaviest Load First": "greedy",
    "Nearest - Closest Provider": "nearest",
    "Round-Robin": "round_robin",
    "EDD - Earliest Discharge Date": "edd",
}

# =========================
# Data Loading
# =========================

def load_patient_df(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype={"patient_id": str}, parse_dates=["discharge_date"])
    except Exception as e:
        raise ValueError(f"Error reading patients file: {e}")
    
    required = {"patient_id", "discharge_date", "length_of_stay", "visit_hours", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Patients CSV is missing mandatory columns: {sorted(missing)}")
    return df


def load_provider_df(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype={"provider_id": str})
    except Exception as e:
        raise ValueError(f"Error reading providers file: {e}")
        
    required = {"provider_id", "latitude", "longitude", "capacity_hrs_per_week"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Providers CSV is missing mandatory columns: {sorted(missing)}")
    if "initial_load_hrs_per_week" not in df.columns:
        df["initial_load_hrs_per_week"] = 0.0
    return df


def dataframe_to_objects(patient_df: pd.DataFrame, provider_df: pd.DataFrame) -> tuple[list[Patient], list[Provider]]:
    patients = [
        Patient(
            patient_id=str(r.patient_id),
            discharge_date=pd.Timestamp(r.discharge_date).date(),
            length_of_stay=int(r.length_of_stay),
            visit_hours=float(r.visit_hours),
            home_coords=(float(r.latitude), float(r.longitude)),
        )
        for r in patient_df.itertuples(index=False)
    ]

    providers = [
        Provider(
            provider_id=str(r.provider_id),
            coords=(float(r.latitude), float(r.longitude)),
            capacity_hrs_per_week=float(r.capacity_hrs_per_week),
            initial_load_hrs_per_week=float(getattr(r, "initial_load_hrs_per_week", 0.0)),
        )
        for r in provider_df.itertuples(index=False)
    ]

    return patients, providers


# =========================
# Calculations Helper functions
# =========================

def week_floor(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return (ts - pd.Timedelta(days=ts.weekday())).normalize()


def active_chart_weeks(discharge_date, length_of_stay) -> list[pd.Timestamp]:
    if pd.isna(discharge_date) or pd.isna(length_of_stay):
        return []
    try:
        los = int(length_of_stay)
        if los <= 0:
            return []
        start = week_floor(discharge_date)
        care_end = pd.Timestamp(discharge_date) + pd.Timedelta(days=los)
        last_week = week_floor(care_end - pd.Timedelta(days=1))
        if last_week < start:
            return [start]
        return list(pd.date_range(start, last_week, freq="W-MON"))
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def run_dashboard_allocation(
    patient_df: pd.DataFrame,
    provider_df: pd.DataFrame,
    alpha: float,
    lookahead_days: int,
    method: str,
    penalty_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    import Rolling_Horizon_Allocation as rha
    # Set penalty weight globally inside the module
    rha.OVERCAPACITY_PENALTY_WEIGHT = penalty_weight

    patients, providers = dataframe_to_objects(patient_df, provider_df)

    result = rolling_horizon_assignment(
        patients=patients,
        providers=providers,
        alpha=alpha,
        lookahead_days=lookahead_days,
        method=method,
    )

    assignments = result["assignments"]
    remaining_capacity = result["remaining_capacity"]
    rh_kpis = result["kpis"]

    # Reconstruct provider utilization per week from remaining capacity
    utilization_rows = []
    provider_capacity = {p.provider_id: p.capacity_hrs_per_week for p in providers}

    for provider_id, weeks_dict in remaining_capacity.items():
        capacity = provider_capacity[provider_id]
        for week, remaining in weeks_dict.items():
            used = capacity - remaining
            utilization_rows.append({
                "provider_id": provider_id,
                "week_start": pd.Timestamp(week),
                "utilization_pct": (used / capacity * 100) if capacity > 0 else 0.0,
            })

    utilization_df = pd.DataFrame(utilization_rows)

    # Add assignments to patient dataframe copy
    assignment_df = patient_df.copy()
    assignment_df["assigned_provider"] = assignment_df["patient_id"].map(assignments)

    # Calculate travel distance
    provider_coord_map = {p.provider_id: p.coords for p in providers}

    def calc_travel_km(row) -> float:
        provider_id = row["assigned_provider"]
        if pd.isna(provider_id) or provider_id not in provider_coord_map:
            return 0.0
        return haversine_km(
            (float(row["latitude"]), float(row["longitude"])),
            provider_coord_map[provider_id],
        )

    assignment_df["travel_km"] = assignment_df.apply(calc_travel_km, axis=1)
    assignment_df["travel_hours"] = assignment_df["travel_km"].apply(rha.travel_hours)

    # Reconstruct weekly activity for patient distribution chart
    activity_rows = []
    for row in assignment_df.itertuples(index=False):
        if pd.isna(row.assigned_provider):
            continue

        for week_start in active_chart_weeks(row.discharge_date, row.length_of_stay):
            activity_rows.append({
                "week_start": pd.Timestamp(week_start),
                "assigned_provider": row.assigned_provider,
                "patients": 1,
            })

    activity_df = pd.DataFrame(activity_rows)
    if activity_df.empty:
        activity_df = pd.DataFrame(columns=["week_start", "assigned_provider", "patients"])

    avg_util_per_provider = utilization_df.groupby("provider_id")["utilization_pct"].mean() if not utilization_df.empty else pd.Series()

    overcapacity_weeks_total = 0
    if isinstance(rh_kpis.get("overcapacity_weeks"), dict):
        overcapacity_weeks_total = int(sum(rh_kpis["overcapacity_weeks"].values()))

    kpis = {
        "total_patients": int(len(assignment_df)),
        "total_assigned": int(rh_kpis.get("total_assigned", assignment_df["assigned_provider"].notna().sum())),
        "peak_occupancy_pct": float(utilization_df["utilization_pct"].max()) if not utilization_df.empty else 0.0,
        "avg_occupancy_pct": float(utilization_df["utilization_pct"].mean()) if not utilization_df.empty else 0.0,
        "avg_travel_km": float(assignment_df["travel_km"].mean()) if not assignment_df.empty else 0.0,
        "avg_travel_hours": float(rh_kpis.get("avg_travel_hours", 0.0)),
        "balance_std_pct": float(rh_kpis.get("utilization_std_dev_%", avg_util_per_provider.std(ddof=0) if not avg_util_per_provider.empty else 0.0)),
        "overcapacity_weeks": overcapacity_weeks_total,
    }

    return assignment_df, utilization_df, activity_df, kpis


# =========================
# Visualizations
# =========================

def format_pct(value: float) -> str:
    return f"{value:,.1f}%"


def format_num(value: float) -> str:
    return f"{value:,.1f}"


def build_distribution_chart(activity_df: pd.DataFrame, min_date, max_date):
    plot_df = (
        activity_df
        .groupby(["week_start", "assigned_provider"], as_index=False)["patients"]
        .sum()
        .sort_values("week_start")
    )

    fig = px.bar(
        plot_df,
        x="week_start",
        y="patients",
        color="assigned_provider",
        barmode="group",
        height=400,
        labels={"week_start": "Week", "patients": "Number of Patients", "assigned_provider": "Care Provider"},
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Number of Patients",
    )
    if pd.notna(min_date) and pd.notna(max_date):
        fig.update_xaxes(range=[min_date, max_date])
    return fig


def build_occupancy_chart(utilization_df: pd.DataFrame, min_date, max_date):
    fig = px.line(
        utilization_df,
        x="week_start",
        y="utilization_pct",
        color="provider_id",
        height=400,
        line_group="provider_id",
        labels={"week_start": "Week", "utilization_pct": "Occupancy Rate (%)", "provider_id": "Care Provider"},
    )

    fig.add_hline(y=100, line_dash="dot", line_color="red", annotation_text="Capacity Limit (100%)")
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Occupancy Rate (%)",
    )
    if pd.notna(min_date) and pd.notna(max_date):
        fig.update_xaxes(range=[min_date, max_date])
    return fig


def build_assignment_chart(assignment_df: pd.DataFrame, provider_df: pd.DataFrame):
    counts = (
        assignment_df
        .groupby("assigned_provider", as_index=False)
        .size()
        .rename(columns={"size": "patients"})
    )

    counts = (
        provider_df[["provider_id"]]
        .merge(counts, left_on="provider_id", right_on="assigned_provider", how="left")
        .fillna(0)
    )
    counts["patients"] = counts["patients"].astype(int)

    fig = px.bar(
        counts.sort_values("patients"),
        x="patients",
        y="provider_id",
        orientation="h",
        height=400,
        text="patients",
        color="provider_id",
        labels={"patients": "Number of Patients", "provider_id": "Care Provider"},
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis_title="Number of Patients",
        yaxis_title="",
    )
    return fig


# =========================
# Page Layout
# =========================

st.title("🏥 Isala Tactical Allocation Dashboard")
st.markdown(
    "This dashboard allows you to simulate and compare different patient allocation strategies "
    "for rolling-horizon planning under various parameter settings and initial workload profiles."
)

# Verify default data files exist
if not DEFAULT_PATIENTS_FILE.exists():
    st.error(f"patients.csv not found in: {PROJECT_DIR}")
    st.stop()

if not DEFAULT_PROVIDERS_FILE.exists():
    st.error(f"providers.csv not found in: {PROJECT_DIR}")
    st.stop()


# ==========================================
# Sidebar - Parameter Settings & Navigation
# ==========================================

st.sidebar.header("🛠️ Dashboard Settings")

# Navigation
st.sidebar.subheader("Navigation")
page = st.sidebar.radio("Go to Page", ["Metrics Dashboard", "Geographic Map"])

try:
    base_patient_df = load_patient_df(DEFAULT_PATIENTS_FILE)
    base_provider_df = load_provider_df(DEFAULT_PROVIDERS_FILE)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Allocation parameters
st.sidebar.subheader("Allocation Parameters")
lookahead_days = st.sidebar.slider(
    "Planning Horizon (Days)",
    min_value=1,
    max_value=42,
    value=7,
    step=1,
    help="Number of days in advance that patients are known for the allocation decision."
)

penalty_weight = st.sidebar.slider(
    "Overcapacity Penalty Weight",
    min_value=0.0,
    max_value=50.0,
    value=10.0,
    step=1.0,
    help="Penalty weight in the scoring function for exceeding weekly provider capacity."
)

alpha = st.sidebar.slider(
    "Alpha (α) Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.05,
    help="Weight between workload balance (alpha=1.0) and travel time (alpha=0.0) in the scoring functions (Greedy & EDD)."
)

# Initial workload profile settings
st.sidebar.subheader("Initial Workload Scenario")
provider_ids = sorted(base_provider_df["provider_id"].tolist())

# Generate load profiles based on loaded provider IDs
profiles_map = {
    "Default (from CSV)": None,
    "Balanced (60% initial workload)": {pid: 0.60 for pid in provider_ids},
}
for pid in provider_ids:
    p_profile = {p: 0.60 for p in provider_ids}
    p_profile[pid] = 0.90
    profiles_map[f"{pid} Overloaded (90%, others 60%)"] = p_profile

profiles_map["Crisis (95% initial workload)"] = {pid: 0.95 for pid in provider_ids}
profiles_map["Custom initial workload..."] = "custom"

selected_profile_label = st.sidebar.selectbox("Select initial workload profile", list(profiles_map.keys()))

# Determine initial workload percentages per provider
provider_load_pcts = {}
if profiles_map[selected_profile_label] == "custom":
    st.sidebar.markdown("**Manual initial workload (%)**")
    for pid in provider_ids:
        # Get standard value from CSV
        p_row = base_provider_df[base_provider_df["provider_id"] == pid]
        csv_pct = 60.0
        if not p_row.empty:
            cap = p_row.iloc[0]["capacity_hrs_per_week"]
            init_load = p_row.iloc[0]["initial_load_hrs_per_week"]
            csv_pct = (init_load / cap * 100.0) if cap > 0 else 60.0
        
        # Ensure default slider value is strictly within bounds [0.0, 100.0] to avoid ValueError
        csv_pct_clipped = min(max(csv_pct, 0.0), 100.0)
            
        val = st.sidebar.slider(f"Initial Workload {pid}", 0.0, 100.0, float(csv_pct_clipped), 5.0)
        provider_load_pcts[pid] = val / 100.0
elif profiles_map[selected_profile_label] is not None:
    provider_load_pcts = profiles_map[selected_profile_label]
else:
    # Default (from CSV)
    provider_load_pcts = None

# Update provider_df with chosen workloads
provider_df = base_provider_df.copy()
if provider_load_pcts is not None:
    provider_df["initial_load_hrs_per_week"] = provider_df.apply(
        lambda r: r["capacity_hrs_per_week"] * provider_load_pcts.get(r["provider_id"], 0.0),
        axis=1
    )

patient_df = base_patient_df.copy()


# ==========================================
# PAGE 1: Metrics Dashboard
# ==========================================
if page == "Metrics Dashboard":

    tab1, tab2 = st.tabs([
        "Strategy Comparison",
        "Detailed Inspection"
    ])

    # Date range boundary for charts
    min_discharge_date = pd.to_datetime(patient_df["discharge_date"]).min()
    max_discharge_date = pd.to_datetime(patient_df["discharge_date"]).max()

    # ==========================================
    # Tab 1: Strategy Comparison
    # ==========================================
    with tab1:
        st.subheader("Comparison of Dispatching Strategies")
        st.markdown(
            "Below, all 4 strategies are compared side-by-side using the parameters set in the sidebar. "
            "This helps evaluate the trade-offs (travel time, workload distribution, overcapacity) immediately."
        )
        
        # Run all strategies
        with st.spinner("Calculating allocations..."):
            comparison_results = {}
            for label, method_key in STRATEGY_OPTIONS.items():
                ass_df, util_df, act_df, kpis = run_dashboard_allocation(
                    patient_df=patient_df,
                    provider_df=provider_df,
                    alpha=alpha,
                    lookahead_days=lookahead_days,
                    method=method_key,
                    penalty_weight=penalty_weight,
                )
                comparison_results[method_key] = {
                    "label": label,
                    "assignment_df": ass_df,
                    "utilization_df": util_df,
                    "activity_df": act_df,
                    "kpis": kpis
                }
                
        # Construct comparative KPI table
        kpi_table_data = []
        for m_key, m_data in comparison_results.items():
            k = m_data["kpis"]
            
            # Calculate average occupancy per provider
            mean_utils = m_data["utilization_df"].groupby("provider_id")["utilization_pct"].mean().to_dict()
            
            row_data = {
                "Strategy": m_data["label"],
                "Avg. Travel Distance": f"{k['avg_travel_km']:.2f} km",
                "Avg. Travel Time": f"{k['avg_travel_hours'] * 60:.1f} min",
                "Workload Imbalance (Std Dev %)": f"{k['balance_std_pct']:.2f}%",
                "Overcapacity Weeks": k["overcapacity_weeks"],
            }
            
            # Dynamically add occupancy columns for each provider
            for pid in provider_ids:
                row_data[f"{pid} Avg. Occupancy"] = f"{mean_utils.get(pid, 0.0):.1f}%"
                
            kpi_table_data.append(row_data)
            
        kpi_df = pd.DataFrame(kpi_table_data)
        
        st.markdown("### Performance Metrics (KPIs)")
        st.dataframe(kpi_df, width="stretch", hide_index=True)
        
        # Graphs for visual comparison
        st.markdown("### Visual Comparison")
        col1, col2 = st.columns(2)
        
        # Extract data for plotting
        plot_df = pd.DataFrame([
            {
                "Strategy": m_data["label"].split(" - ")[0], # Short name
                "Avg. Travel Distance (km)": m_data["kpis"]["avg_travel_km"],
                "Workload Imbalance (Std Dev %)": m_data["kpis"]["balance_std_pct"],
                "Overcapacity Weeks": m_data["kpis"]["overcapacity_weeks"],
            }
            for m_key, m_data in comparison_results.items()
        ])
        
        with col1:
            fig_travel_comp = px.bar(
                plot_df,
                x="Strategy",
                y="Avg. Travel Distance (km)",
                color="Strategy",
                title="Average Travel Distance per Patient (Shorter is better)",
                text_auto=".2f",
            )
            fig_travel_comp.update_layout(showlegend=False, xaxis_title="", yaxis_title="Travel Distance (km)")
            st.plotly_chart(fig_travel_comp, width="stretch")
            
        with col2:
            fig_imbalance_comp = px.bar(
                plot_df,
                x="Strategy",
                y="Workload Imbalance (Std Dev %)",
                color="Strategy",
                title="Workload Imbalance between Providers (Lower is better)",
                text_auto=".2f",
            )
            fig_imbalance_comp.update_layout(showlegend=False, xaxis_title="", yaxis_title="Standard Deviation (%)")
            st.plotly_chart(fig_imbalance_comp, width="stretch")
            
        col3, col4 = st.columns(2)
        
        with col3:
            fig_overcap_comp = px.bar(
                plot_df,
                x="Strategy",
                y="Overcapacity Weeks",
                color="Strategy",
                title="Total Weeks with Overcapacity (Lower is better)",
                text_auto="d",
            )
            fig_overcap_comp.update_layout(showlegend=False, xaxis_title="", yaxis_title="Weeks with Overcapacity")
            st.plotly_chart(fig_overcap_comp, width="stretch")
            
        with col4:
            # Grouped bar chart showing average utilization per provider per strategy
            util_provider_rows = []
            for m_key, m_data in comparison_results.items():
                mean_util = m_data["utilization_df"].groupby("provider_id")["utilization_pct"].mean().reset_index()
                for row in mean_util.itertuples(index=False):
                    util_provider_rows.append({
                        "Strategy": m_data["label"].split(" - ")[0],
                        "Care Provider": row.provider_id,
                        "Average Occupancy (%)": row.utilization_pct
                    })
            util_provider_df = pd.DataFrame(util_provider_rows)
            
            fig_util_provider = px.bar(
                util_provider_df,
                x="Strategy",
                y="Average Occupancy (%)",
                color="Care Provider",
                barmode="group",
                title="Average Occupancy per Care Provider",
                text_auto=".1f",
            )
            fig_util_provider.update_layout(xaxis_title="", yaxis_title="Average Occupancy (%)")
            st.plotly_chart(fig_util_provider, width="stretch")


    # ==========================================
    # Tab 2: Detailed Inspection
    # ==========================================
    with tab2:
        st.subheader("Detailed Analysis per Strategy")
        
        # Choose strategy to inspect
        selected_strategy_label = st.selectbox(
            "Select a strategy to inspect in detail:",
            list(STRATEGY_OPTIONS.keys()),
            key="detail_strategy_select"
        )
        
        selected_key = STRATEGY_OPTIONS[selected_strategy_label]
        res_data = comparison_results[selected_key]
        
        assignment_df = res_data["assignment_df"]
        utilization_df = res_data["utilization_df"]
        activity_df = res_data["activity_df"]
        kpis = res_data["kpis"]
        
        # Metric cards display
        kpi_cols = st.columns(4)
        
        with kpi_cols[0]:
            st.metric("Total Patients", f"{kpis['total_patients']}")
        
        with kpi_cols[1]:
            st.metric("Peak Occupancy", format_pct(kpis["peak_occupancy_pct"]))
        
        with kpi_cols[2]:
            st.metric("Workload Imbalance (Std Dev)", format_pct(kpis["balance_std_pct"]))
        
        with kpi_cols[3]:
            st.metric("Avg. Travel Distance", f"{format_num(kpis['avg_travel_km'])} km")
            
        st.markdown("---")
        st.subheader("Patient Distribution per Provider (Active Patients per Week)")
        
        # Build per-provider distribution charts with a shared Y-axis max
        dist_plot_df = (
            activity_df
            .groupby(["week_start", "assigned_provider"], as_index=False)["patients"]
            .sum()
            .sort_values("week_start")
        )
        y_max = dist_plot_df["patients"].max() * 1.15 if not dist_plot_df.empty else 10
        
        dist_providers = sorted(activity_df["assigned_provider"].dropna().unique()) if not activity_df.empty else provider_ids
        dist_cols = st.columns(len(dist_providers))
        
        for i, pid in enumerate(dist_providers):
            pid_df = dist_plot_df[dist_plot_df["assigned_provider"] == pid]
            fig_pid = px.bar(
                pid_df,
                x="week_start",
                y="patients",
                height=350,
                labels={"week_start": "Week", "patients": "Patients"},
                title=pid,
            )
            fig_pid.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
                xaxis_title="",
                yaxis_title="Patients" if i == 0 else "",
                yaxis=dict(range=[0, y_max]),
            )
            if pd.notna(min_discharge_date) and pd.notna(max_discharge_date):
                fig_pid.update_xaxes(range=[min_discharge_date, max_discharge_date])
            with dist_cols[i]:
                st.plotly_chart(fig_pid, width="stretch", config={"displayModeBar": False})
        
        st.subheader("Weekly Occupancy Rate per Provider (%)")
        st.plotly_chart(
            build_occupancy_chart(utilization_df, min_discharge_date, max_discharge_date),
            width="stretch",
            config={"displayModeBar": False},
        )
        
        st.subheader("Total Assigned Patients per Provider")
        st.plotly_chart(
            build_assignment_chart(assignment_df, provider_df),
            width="stretch",
            config={"displayModeBar": False},
        )

        
        # Full assignment list table
        with st.expander("Assignment Table (Full Patient List)"):
            st.dataframe(
                assignment_df[[
                    "patient_id",
                    "discharge_date",
                    "length_of_stay",
                    "visit_hours",
                    "assigned_provider",
                    "travel_km",
                    "travel_hours"
                ]].rename(columns={
                    "patient_id": "Patient ID",
                    "discharge_date": "Discharge Date",
                    "length_of_stay": "Length of Stay (days)",
                    "visit_hours": "Care Hours per Week",
                    "assigned_provider": "Assigned Provider",
                    "travel_km": "Travel Distance (km)",
                    "travel_hours": "Travel Time (hours)"
                }),
                width="stretch",
                hide_index=True,
            )

# ==========================================
# PAGE 2: Geographic Map
# ==========================================
elif page == "Geographic Map":
    st.subheader("Geographic Map Visualization")
    st.markdown(
        "This map visualizes the geographical locations of the **Care Providers** "
        "and shows a **Heatmap** representing the density of **Patients** "
        "weighted by their weekly visit hours."
    )

    # Layer toggles
    show_heatmap = st.checkbox("Show Patient Heatmap", value=True)
    show_patient_points = st.checkbox("Show Patient Scatter Points", value=False)
    show_providers = st.checkbox("Show Care Providers", value=True)

    layers = []

    # 1. Patient Heatmap Layer
    if show_heatmap and not patient_df.empty:
        layers.append(pdk.Layer(
            "HeatmapLayer",
            data=patient_df,
            get_position="[longitude, latitude]",
            get_weight="visit_hours",
            radius_pixels=60,
            intensity=1.5,
            threshold=0.03,
        ))

    # 2. Patient Points Layer
    if show_patient_points and not patient_df.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=patient_df,
            get_position="[longitude, latitude]",
            get_color="[255, 140, 0, 160]",  # Orange
            get_radius=150,
            pickable=True,
        ))

    # 3. Provider Points Layer
    if show_providers and not provider_df.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=provider_df,
            get_position="[longitude, latitude]",
            get_color="[31, 119, 180, 240]",  # Blue
            get_radius=1000,
            pickable=True,
        ))

    # Calculate center of the map
    if not patient_df.empty:
        center_lat = patient_df["latitude"].mean()
        center_lon = patient_df["longitude"].mean()
    elif not provider_df.empty:
        center_lat = provider_df["latitude"].mean()
        center_lon = provider_df["longitude"].mean()
    else:
        center_lat = 52.5
        center_lon = 6.1

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=10.5,
        pitch=0,
    )

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Details:</b><br/>"
                    "ID: {provider_id}{patient_id}<br/>"
                    "Capacity/Visit Hours: {capacity_hrs_per_week}{visit_hours} hrs/week"
        }
    )
    
    st.pydeck_chart(r)


# ==========================================
# Sidebar Close Command
# ==========================================
st.sidebar.markdown("---")
if st.sidebar.button("🔴 Stop Dashboard & Terminal"):
    st.sidebar.success("Dashboard is stopping...")
    import signal
    os.kill(os.getpid(), signal.SIGTERM)