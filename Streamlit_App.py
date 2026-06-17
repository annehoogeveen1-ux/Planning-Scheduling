from __future__ import annotations

import os
import sys
import subprocess

# Dwing het script om direct via Streamlit te herstarten ALS het met de Play-knop is opgestart
if "streamlit" not in sys.modules and "-m" not in sys.argv:
    if not os.environ.get("STREAMLIT_ALREADY_RUNNING"):
        os.environ["STREAMLIT_ALREADY_RUNNING"] = "1"
        print("Dashboard wordt opgestart in de browser via Streamlit...")
        
        # We voegen hier argumenten toe die ervoor zorgen dat de server stopt als de browser sluit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", __file__,
            "--browser.gatherUsageStats=False",
            "--server.headless=false"
        ])
        sys.exit()

        
from pathlib import Path
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from Rolling_Horizon_Allocation import (  # gebruikt jullie bestaande code
    Patient,
    Provider,
    generate_weeks,
    haversine_km,
    travel_hours,
    OVERCAPACITY_PENALTY_WEIGHT,
)

try:
    from Patient_Generator import generate_dataset as generate_patient_dataset
except Exception:
    generate_patient_dataset = None


# =========================
# Config
# =========================

st.set_page_config(page_title="Isala Tactical Dashboard", layout="wide")
px.defaults.template = "plotly_white"

DEFAULT_PATIENTS_FILE = PROJECT_DIR / "patients.csv"
DEFAULT_PROVIDERS_FILE = PROJECT_DIR / "providers.csv"

STRATEGY_OPTIONS = {
    "Zwaarste last eerst": "heaviest_load_first",
    "Round-robin": "round_robin",
    "Afstandsbewust": "distance_aware",
}


# =========================
# Data laden
# =========================

@st.cache_data(show_spinner=False)
def load_patient_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["discharge_date"])
    required = {"patient_id", "discharge_date", "length_of_stay", "visit_hours", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"patients.csv mist kolommen: {sorted(missing)}")
    if "type_care" not in df.columns:
        df["type_care"] = "Onbekend"
    return df


@st.cache_data(show_spinner=False)
def load_provider_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"provider_id", "latitude", "longitude", "capacity_hrs_per_week"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"providers.csv mist kolommen: {sorted(missing)}")
    if "initial_load_hrs_per_week" not in df.columns:
        df["initial_load_hrs_per_week"] = 0.0
    return df


@st.cache_data(show_spinner=False)
def build_dummy_patients(n_patients: int) -> pd.DataFrame:
    if generate_patient_dataset is not None:
        df = generate_patient_dataset(n=n_patients).copy()
        df["discharge_date"] = pd.to_datetime(df["discharge_date"])
        if "type_care" not in df.columns:
            df["type_care"] = "Onbekend"
        return df

    rng = np.random.default_rng(42)
    base_date = pd.Timestamp("2026-01-01")
    centre_lat, centre_lon = 52.5137, 6.1237

    df = pd.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(1, n_patients + 1)],
            "discharge_date": base_date + pd.to_timedelta(rng.integers(0, 30, n_patients), unit="D"),
            "length_of_stay": rng.integers(14, 42, n_patients),
            "visit_hours": rng.choice([2, 3, 4, 5, 6], size=n_patients, p=[0.10, 0.20, 0.35, 0.25, 0.10]),
            "latitude": centre_lat + rng.normal(0, 0.18, n_patients),
            "longitude": centre_lon + rng.normal(0, 0.20, n_patients),
            "type_care": rng.choice(["chirurgie", "cardiologie", "urologie", "revalidatie"], size=n_patients),
        }
    )
    return df


@st.cache_data(show_spinner=False)
def build_dummy_providers() -> pd.DataFrame:
    if DEFAULT_PROVIDERS_FILE.exists():
        return load_provider_df(DEFAULT_PROVIDERS_FILE)

    return pd.DataFrame(
        {
            "provider_id": ["HomecareA", "HomecareB", "HomecareC"],
            "latitude": [52.27818, 52.52965, 52.17182],
            "longitude": [5.97271, 5.91715, 5.74478],
            "capacity_hrs_per_week": [80, 80, 80],
            "initial_load_hrs_per_week": [55, 40, 60],
        }
    )


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
# Rolling horizon uitbreiding
# =========================

def week_floor(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return (ts - pd.Timedelta(days=ts.weekday())).normalize()


def active_chart_weeks(discharge_date, length_of_stay) -> list[pd.Timestamp]:
    start = week_floor(discharge_date)
    care_end = pd.Timestamp(discharge_date) + pd.Timedelta(days=int(length_of_stay))
    last_week = week_floor(care_end - pd.Timedelta(days=1))
    return list(pd.date_range(start, last_week, freq="W-MON"))


def order_patients(known_patients: list[Patient], strategy: str, travel_matrix: dict) -> list[Patient]:
    if strategy == "round_robin":
        return sorted(known_patients, key=lambda p: (p.discharge_date, p.patient_id))

    if strategy == "distance_aware":
        return sorted(
            known_patients,
            key=lambda p: (min(travel_matrix[p.patient_id].values()), -p.visit_hours, p.discharge_date),
        )

    return sorted(
        known_patients,
        key=lambda p: (-p.visit_hours, -p.length_of_stay, p.discharge_date, p.patient_id),
    )


def choose_provider(
    patient: Patient,
    providers: list[Provider],
    strategy: str,
    alpha: float,
    remaining_capacity: dict,
    active_periods: dict,
    travel_matrix: dict,
    rr_state: dict,
    overcapacity_penalty_weight: float,
) -> Provider:
    active_weeks = active_periods.get(patient.patient_id, [])

    if strategy == "round_robin":
        ordered = providers[rr_state["next_index"]:] + providers[: rr_state["next_index"]]
        best_provider = None
        best_tuple = (float("inf"), float("inf"))

        for provider in ordered:
            penalty = 0.0
            distance = travel_matrix[patient.patient_id][provider.provider_id]
            for week in active_weeks:
                needed = patient.visit_hours + distance
                deficit = needed - remaining_capacity[provider.provider_id].get(week, provider.capacity_hrs_per_week)
                if deficit > 0:
                    penalty = max(penalty, overcapacity_penalty_weight * deficit)

            candidate = (penalty, distance)
            if candidate < best_tuple:
                best_tuple = candidate
                best_provider = provider

        rr_state["next_index"] = (rr_state["next_index"] + 1) % len(providers)
        return best_provider

    best_provider = None
    best_score = float("inf")

    for provider in providers:
        if active_weeks:
            loads = [
                1 - remaining_capacity[provider.provider_id].get(week, provider.capacity_hrs_per_week)
                / provider.capacity_hrs_per_week
                for week in active_weeks
            ]
            load = max(loads) if loads else 0.0
        else:
            load = 0.0

        distance = travel_matrix[patient.patient_id][provider.provider_id]
        distance_norm = min(distance / 2.0, 1.0)

        penalty = 0.0
        for week in active_weeks:
            needed = patient.visit_hours + distance
            deficit = needed - remaining_capacity[provider.provider_id].get(week, provider.capacity_hrs_per_week)
            if deficit > 0:
                penalty = max(penalty, overcapacity_penalty_weight * deficit)

        if strategy == "distance_aware":
            score = 0.80 * distance_norm + 0.20 * load + penalty
        else:
            score = alpha * load + (1 - alpha) * distance_norm + penalty

        if score < best_score:
            best_score = score
            best_provider = provider

    return best_provider


@st.cache_data(show_spinner=False)
def run_dashboard_allocation(
    patient_df: pd.DataFrame,
    provider_df: pd.DataFrame,
    alpha: float,
    lookahead_days: int,
    booking_horizon_days: int,
    avg_speed_kmh: float,
    strategy: str,
    overcapacity_penalty_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    patients, providers = dataframe_to_objects(patient_df, provider_df)

    all_discharge_dates = sorted(set(p.discharge_date for p in patients))
    global_start = min(p.discharge_date for p in patients)
    global_end = max(
        p.discharge_date + timedelta(days=min(p.length_of_stay, booking_horizon_days))
        for p in patients
    )
    all_weeks = generate_weeks(global_start, global_end)

    remaining_capacity = {
        provider.provider_id: {
            week: provider.capacity_hrs_per_week - provider.initial_load_hrs_per_week
            for week in all_weeks
        }
        for provider in providers
    }

    assignments: dict[str, str] = {}
    travel_hours_map: dict[str, float] = {}
    processed: set[str] = set()
    rr_state = {"next_index": 0}

    for t in all_discharge_dates:
        window_end = t + timedelta(days=lookahead_days)
        known_patients = [
            p for p in patients if t <= p.discharge_date <= window_end and p.patient_id not in processed
        ]
        if not known_patients:
            continue

        active_periods = {}
        travel_matrix = {}

        for patient in known_patients:
            booking_end = patient.discharge_date + timedelta(days=min(patient.length_of_stay, booking_horizon_days))
            effective_end = min(patient.care_end, booking_end)
            active_weeks = [
                week for week in all_weeks if week_floor(patient.discharge_date) <= pd.Timestamp(week) < pd.Timestamp(effective_end)
            ]
            active_periods[patient.patient_id] = [
                week.date() if isinstance(week, pd.Timestamp) else week for week in active_weeks
            ]

            travel_matrix[patient.patient_id] = {
                provider.provider_id: travel_hours(patient.home_coords, provider.coords, avg_speed_kmh)
                for provider in providers
            }

        for patient in order_patients(known_patients, strategy, travel_matrix):
            if patient.patient_id in processed:
                continue

            provider = choose_provider(
                patient=patient,
                providers=providers,
                strategy=strategy,
                alpha=alpha,
                remaining_capacity=remaining_capacity,
                active_periods=active_periods,
                travel_matrix=travel_matrix,
                rr_state=rr_state,
                overcapacity_penalty_weight=overcapacity_penalty_weight,
            )

            distance_hours = travel_matrix[patient.patient_id][provider.provider_id]
            for week in active_periods[patient.patient_id]:
                remaining_capacity[provider.provider_id][week] -= patient.visit_hours + distance_hours

            assignments[patient.patient_id] = provider.provider_id
            travel_hours_map[patient.patient_id] = distance_hours
            processed.add(patient.patient_id)

    utilization_rows = []
    for provider in providers:
        for week in all_weeks:
            used = provider.capacity_hrs_per_week - remaining_capacity[provider.provider_id][week]
            utilization_pct = used / provider.capacity_hrs_per_week * 100
            utilization_rows.append(
                {
                    "provider_id": provider.provider_id,
                    "week_start": pd.Timestamp(week),
                    "utilization_pct": utilization_pct,
                }
            )
    utilization_df = pd.DataFrame(utilization_rows)

    assignment_df = patient_df.copy()
    assignment_df["assigned_provider"] = assignment_df["patient_id"].map(assignments)
    assignment_df["travel_hours"] = assignment_df["patient_id"].map(travel_hours_map)

    provider_coord_map = provider_df.set_index("provider_id")[["latitude", "longitude"]].to_dict("index")
    assignment_df["travel_km"] = assignment_df.apply(
        lambda row: haversine_km(
            (row["latitude"], row["longitude"]),
            (
                provider_coord_map[row["assigned_provider"]]["latitude"],
                provider_coord_map[row["assigned_provider"]]["longitude"],
            ),
        ),
        axis=1,
    )

    activity_rows = []
    for row in assignment_df.itertuples(index=False):
        for week_start in active_chart_weeks(row.discharge_date, row.length_of_stay):
            activity_rows.append(
                {
                    "week_start": pd.Timestamp(week_start),
                    "assigned_provider": row.assigned_provider,
                    "type_care": getattr(row, "type_care", "Onbekend"),
                    "patients": 1,
                }
            )
    activity_df = pd.DataFrame(activity_rows)
    if activity_df.empty:
        activity_df = pd.DataFrame(columns=["week_start", "assigned_provider", "type_care", "patients"])

    avg_util_per_provider = utilization_df.groupby("provider_id")["utilization_pct"].mean()
    kpis = {
        "total_patients": int(len(assignment_df)),
        "peak_occupancy_pct": float(utilization_df["utilization_pct"].max()) if not utilization_df.empty else 0.0,
        "avg_occupancy_pct": float(utilization_df["utilization_pct"].mean()) if not utilization_df.empty else 0.0,
        "avg_travel_hours": float(assignment_df["travel_hours"].mean()) if not assignment_df.empty else 0.0,
        "avg_travel_km": float(assignment_df["travel_km"].mean()) if not assignment_df.empty else 0.0,
        "balance_std_pct": float(avg_util_per_provider.std(ddof=0)) if not avg_util_per_provider.empty else 0.0,
        "overcapacity_weeks": int((utilization_df["utilization_pct"] > 100).sum()) if not utilization_df.empty else 0,
    }

    return assignment_df, utilization_df, activity_df, kpis


# =========================
# Visualisaties
# =========================

def format_pct(value: float) -> str:
    return f"{value:,.1f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def format_num(value: float) -> str:
    return f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def build_distribution_chart(activity_df: pd.DataFrame, split_by: str):
    group_col = "assigned_provider" if split_by == "Homecare" else "type_care"
    plot_df = (
        activity_df.groupby(["week_start", group_col], as_index=False)["patients"].sum().sort_values("week_start")
    )
    fig = px.bar(
        plot_df,
        x="week_start",
        y="patients",
        color=group_col,
        barmode="group",
        height=400,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="",
    )
    return fig


def build_occupancy_chart(utilization_df: pd.DataFrame):
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
        yaxis_title="",
    )
    return fig


def build_assignment_chart(assignment_df: pd.DataFrame, provider_df: pd.DataFrame):
    counts = assignment_df.groupby("assigned_provider", as_index=False).size().rename(columns={"size": "patients"})
    counts = provider_df[["provider_id"]].merge(counts, left_on="provider_id", right_on="assigned_provider", how="left").fillna(0)
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
        xaxis_title="",
        yaxis_title="",
    )
    return fig


# =========================
# Pagina
# =========================

real_data_available = DEFAULT_PATIENTS_FILE.exists() and DEFAULT_PROVIDERS_FILE.exists()

st.title("Isala Tactical Dashboard")

control_cols = st.columns([1.1, 1.4, 0.8, 0.9, 0.9, 0.9, 0.9, 1.0])

with control_cols[0]:
    data_mode = st.selectbox("Data", ["Reële data", "Dummy data"], index=0 if real_data_available else 1)
with control_cols[1]:
    strategy_label = st.selectbox("Strategie", list(STRATEGY_OPTIONS.keys()), index=0)
with control_cols[2]:
    alpha = st.slider("Alpha", 0.0, 1.0, 0.60, 0.05)
with control_cols[3]:
    lookahead_days = st.slider("Zicht-horizon", 1, 42, 7, 1)
with control_cols[4]:
    booking_horizon_days = st.slider("Boek-horizon", 7, 84, 42, 7)
with control_cols[5]:
    avg_speed_kmh = st.slider("Snelheid", 10, 80, 30, 5)
with control_cols[6]:
    penalty_weight = st.slider("Penalty", 1.0, 25.0, float(OVERCAPACITY_PENALTY_WEIGHT), 1.0)
with control_cols[7]:
    split_by = st.selectbox("Verdeling", ["Homecare", "Zorgtype"], index=0)

if data_mode == "Reële data" and real_data_available:
    patient_df = load_patient_df(DEFAULT_PATIENTS_FILE)
    provider_df = load_provider_df(DEFAULT_PROVIDERS_FILE)
else:
    provider_df = build_dummy_providers()
    n_dummy_patients = st.slider("Aantal dummy patiënten", 20, 250, 80, 10)
    patient_df = build_dummy_patients(n_dummy_patients)

assignment_df, utilization_df, activity_df, kpis = run_dashboard_allocation(
    patient_df=patient_df,
    provider_df=provider_df,
    alpha=alpha,
    lookahead_days=lookahead_days,
    booking_horizon_days=booking_horizon_days,
    avg_speed_kmh=avg_speed_kmh,
    strategy=STRATEGY_OPTIONS[strategy_label],
    overcapacity_penalty_weight=penalty_weight,
)

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Totaal patiënten", f"{kpis['total_patients']}")
with kpi_cols[1]:
    st.metric("Piek bezetting", format_pct(kpis["peak_occupancy_pct"]))
with kpi_cols[2]:
    st.metric("Gem. bezetting", format_pct(kpis["avg_occupancy_pct"]))
with kpi_cols[3]:
    st.metric("Gem. reistijd", f"{format_num(kpis['avg_travel_hours'])} uur")

mid_cols = st.columns(2)
with mid_cols[0]:
    st.subheader("Patiëntverdeling")
    st.plotly_chart(build_distribution_chart(activity_df, split_by), use_container_width=True, config={"displayModeBar": False})
with mid_cols[1]:
    st.subheader("Bezettingsgraad")
    st.plotly_chart(build_occupancy_chart(utilization_df), use_container_width=True, config={"displayModeBar": False})

st.subheader("Toewijzingen per homecare")
st.plotly_chart(build_assignment_chart(assignment_df, provider_df), use_container_width=True, config={"displayModeBar": False})


#C:\Users\krisl\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run Streamlit_App.py
#C:/Users/tomto/AppData/Local/Programs/Python/Python313/python.exe -m streamlit run "c:/Users/tomto/OneDrive/Documents/School/MSc/6. MSc Y2Q4/PS Planning and Scheduling/Planning-Scheduling/Streamlit_App.py"