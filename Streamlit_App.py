from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Dwing het script om direct via Streamlit te herstarten ALS het met de Play-knop is opgestart
if "streamlit" not in sys.modules and "-m" not in sys.argv:
    if not os.environ.get("STREAMLIT_ALREADY_RUNNING"):
        os.environ["STREAMLIT_ALREADY_RUNNING"] = "1"
        print("Dashboard wordt opgestart in de browser via Streamlit...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", __file__,
            "--browser.gatherUsageStats=False",
            "--server.headless=false",
        ])
        sys.exit()

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
    "Greedy - zwaarste last eerst": "greedy",
    "Nearest - dichtstbijzijnde provider": "nearest",
    "Round-robin": "round_robin",
    "EDD - vroegste ontslagdatum eerst": "edd",
}


# =========================
# Data laden
# =========================

@st.cache_data(show_spinner=False)
def load_patient_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patient_id": str}, parse_dates=["discharge_date"])
    required = {"patient_id", "discharge_date", "length_of_stay", "visit_hours", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"patients.csv mist kolommen: {sorted(missing)}")
    return df


@st.cache_data(show_spinner=False)
def load_provider_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"provider_id": str})
    required = {"provider_id", "latitude", "longitude", "capacity_hrs_per_week"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"providers.csv mist kolommen: {sorted(missing)}")
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
# Dashboard-berekeningen rond bestaande Rolling Horizon
# =========================

def week_floor(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return (ts - pd.Timedelta(days=ts.weekday())).normalize()


def active_chart_weeks(discharge_date, length_of_stay) -> list[pd.Timestamp]:
    start = week_floor(discharge_date)
    care_end = pd.Timestamp(discharge_date) + pd.Timedelta(days=int(length_of_stay))
    last_week = week_floor(care_end - pd.Timedelta(days=1))
    return list(pd.date_range(start, last_week, freq="W-MON"))


@st.cache_data(show_spinner=False)
def run_dashboard_allocation(
    patient_df: pd.DataFrame,
    provider_df: pd.DataFrame,
    alpha: float,
    lookahead_days: int,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
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

    # Bezetting per provider per week reconstrueren uit remaining_capacity
    utilization_rows = []
    provider_capacity = {p.provider_id: p.capacity_hrs_per_week for p in providers}

    for provider_id, weeks_dict in remaining_capacity.items():
        capacity = provider_capacity[provider_id]
        for week, remaining in weeks_dict.items():
            used = capacity - remaining
            utilization_rows.append({
                "provider_id": provider_id,
                "week_start": pd.Timestamp(week),
                "utilization_pct": used / capacity * 100,
            })

    utilization_df = pd.DataFrame(utilization_rows)

    # Toewijzingen toevoegen aan patients dataframe
    assignment_df = patient_df.copy()
    assignment_df["assigned_provider"] = assignment_df["patient_id"].map(assignments)

    # Reisafstand berekenen met dezelfde haversine_km functie uit Rolling_Horizon_Allocation.py
    provider_coord_map = provider_df.set_index("provider_id")[["latitude", "longitude"]].to_dict("index")

    def calc_travel_km(row) -> float:
        provider_id = row["assigned_provider"]
        if pd.isna(provider_id) or provider_id not in provider_coord_map:
            return 0.0
        return haversine_km(
            (float(row["latitude"]), float(row["longitude"])),
            (
                float(provider_coord_map[provider_id]["latitude"]),
                float(provider_coord_map[provider_id]["longitude"]),
            ),
        )

    assignment_df["travel_km"] = assignment_df.apply(calc_travel_km, axis=1)

    # Activiteit per week voor patiëntverdeling
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

    avg_util_per_provider = utilization_df.groupby("provider_id")["utilization_pct"].mean()

    overcapacity_weeks_total = 0
    if isinstance(rh_kpis.get("overcapacity_weeks"), dict):
        overcapacity_weeks_total = int(sum(rh_kpis["overcapacity_weeks"].values()))

    kpis = {
        "total_patients": int(len(assignment_df)),
        "total_assigned": int(rh_kpis.get("total_assigned", assignment_df["assigned_provider"].notna().sum())),
        "peak_occupancy_pct": float(utilization_df["utilization_pct"].max()) if not utilization_df.empty else 0.0,
        "avg_occupancy_pct": float(utilization_df["utilization_pct"].mean()) if not utilization_df.empty else 0.0,
        "avg_travel_km": float(rh_kpis.get("avg_distance_km", assignment_df["travel_km"].mean())),
        "balance_std_pct": float(rh_kpis.get("utilization_std_dev_%", avg_util_per_provider.std(ddof=0))),
        "overcapacity_weeks": overcapacity_weeks_total,
    }

    return assignment_df, utilization_df, activity_df, kpis


# =========================
# Visualisaties
# =========================

def format_pct(value: float) -> str:
    return f"{value:,.1f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def format_num(value: float) -> str:
    return f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


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
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Aantal patiënten",
    )
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
    )

    fig.add_hline(y=100, line_dash="dot")
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Bezettingsgraad (%)",
    )
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
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis_title="Aantal patiënten",
        yaxis_title="",
    )
    return fig


# =========================
# Pagina
# =========================

st.title("Isala Tactical Dashboard")

if not DEFAULT_PATIENTS_FILE.exists():
    st.error(f"patients.csv niet gevonden in: {PROJECT_DIR}")
    st.stop()

if not DEFAULT_PROVIDERS_FILE.exists():
    st.error(f"providers.csv niet gevonden in: {PROJECT_DIR}")
    st.stop()

patient_df = load_patient_df(DEFAULT_PATIENTS_FILE)
provider_df = load_provider_df(DEFAULT_PROVIDERS_FILE)

control_cols = st.columns([1.5, 1, 1])

with control_cols[0]:
    strategy_label = st.selectbox("Strategie", list(STRATEGY_OPTIONS.keys()), index=0)

with control_cols[1]:
    alpha = st.slider("Alpha", 0.0, 1.0, 0.60, 0.05)

with control_cols[2]:
    lookahead_days = st.slider("Zicht-horizon", 1, 42, 7, 1)

assignment_df, utilization_df, activity_df, kpis = run_dashboard_allocation(
    patient_df=patient_df,
    provider_df=provider_df,
    alpha=alpha,
    lookahead_days=lookahead_days,
    method=STRATEGY_OPTIONS[strategy_label],
)

# X-as van de grafieken: start bij eerste discharge date en stop bij laatste discharge date uit patients.csv
min_discharge_date = pd.to_datetime(patient_df["discharge_date"]).min()
max_discharge_date = pd.to_datetime(patient_df["discharge_date"]).max()

kpi_cols = st.columns(5)

with kpi_cols[0]:
    st.metric("Totaal patiënten", f"{kpis['total_patients']}")

with kpi_cols[1]:
    st.metric("Toegewezen", f"{kpis['total_assigned']}")

with kpi_cols[2]:
    st.metric("Piek bezetting", format_pct(kpis["peak_occupancy_pct"]))

with kpi_cols[3]:
    st.metric("Gem. bezetting", format_pct(kpis["avg_occupancy_pct"]))

with kpi_cols[4]:
    st.metric("Gem. reisafstand", f"{format_num(kpis['avg_travel_km'])} km")

mid_cols = st.columns(2)

with mid_cols[0]:
    st.subheader("Patiëntverdeling per homecare")
    st.plotly_chart(
        build_distribution_chart(activity_df, min_discharge_date, max_discharge_date),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with mid_cols[1]:
    st.subheader("Bezettingsgraad per homecare")
    st.plotly_chart(
        build_occupancy_chart(utilization_df, min_discharge_date, max_discharge_date),
        use_container_width=True,
        config={"displayModeBar": False},
    )

st.subheader("Toewijzingen per homecare")
st.plotly_chart(
    build_assignment_chart(assignment_df, provider_df),
    use_container_width=True,
    config={"displayModeBar": False},
)

with st.expander("Toewijzingstabel"):
    st.dataframe(
        assignment_df[[
            "patient_id",
            "discharge_date",
            "length_of_stay",
            "visit_hours",
            "assigned_provider",
            "travel_km",
        ]],
        use_container_width=True,
    )

# ==========================================
# Knop om de applicatie netjes te sluiten
# ==========================================
st.sidebar.markdown("---")
if st.sidebar.button("🔴 Sluit Dashboard & Terminal"):
    st.sidebar.success("Dashboard wordt afgesloten...")
    import signal
    os.kill(os.getpid(), signal.SIGTERM)


#C:/Users/krisl/AppData/Local/Python/pythoncore-3.14-64/python.exe -m streamlit run Streamlit_App.py