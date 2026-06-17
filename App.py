from __future__ import annotations

"""
README — Streamlit dashboard voor rolling-horizon allocatie
===========================================================

Doel
----
Deze app draait een rolling-horizon allocatie met aanpasbare parameters,
grafieken, kaarten, scenario-opslag en sensitiviteitsanalyse.

Snel starten
------------
1. pip install -r requirements.txt
2. streamlit run app.py

Verwachte CSV's
---------------
providers.csv
    provider_id, latitude, longitude, capacity_hrs_per_week,
    initial_load_hrs_per_week

patients.csv
    patient_id, discharge_date, length_of_stay, visit_hours,
    latitude, longitude
    (optioneel: nurse_skill, type_care)

travel_matrix.csv (optioneel)
    patient_id, provider_id, travel_km of travel_hours

isala.csv (optioneel)
    location_id, name, latitude, longitude, location_type

Integratie eigen allocatiefunctie
---------------------------------
De app ondersteunt twee smaken:

1. Referentie-allocator in deze app.
2. Jullie eigen Python-module.

Voor de eigen module is idealiter beschikbaar:
- rolling_horizon_assignment(patients, providers, alpha, lookahead_days, avg_speed_kmh, ...)
of
- een wrapper met output:
  {
      "assignments": {...} of DataFrame,
      "remaining_capacity": {...},
      "kpis": {...}
  }

Als jullie module geen alternatieve strategieën ondersteunt, gebruikt de app
voor die strategieën automatisch de referentie-allocator.
"""

import html
import importlib.util
import inspect
import io
import json
import math
import tempfile
import textwrap
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st


st.set_page_config(
    page_title="Isala tactisch dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Isala tactisch dashboard voor rolling-horizon allocatie"
APP_SUBTITLE = (
    "Realtime scenariovergelijking voor patiënttoewijzing, bezetting, reisdruk en kaartvisualisatie."
)

DEFAULT_CENTER = {"lat": 52.5136992436934, "lon": 6.123670119684271, "naam": "Isala Zwolle"}

DEFAULT_HOMECARES = [
    {"provider_id": "HomecareA", "latitude": 52.27818005978114, "longitude": 5.972706090677909},
    {"provider_id": "HomecareB", "latitude": 52.52965432815976, "longitude": 5.917148356828077},
    {"provider_id": "HomecareC", "latitude": 52.171816247239505, "longitude": 5.744777726409898},
]

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

STRATEGIES = {
    "heaviest_first": "Zwaarste last eerst",
    "round_robin": "Round-robin",
    "distance_based": "Kortste afstand eerst",
    "capacity_proportional": "Capaciteit-proportioneel",
    "custom_pipeline": "Custom stappenplan",
}

ISOLATION_RULES = {
    "vast_bij_ziekenhuis": "Vast bij Isala-locatie",
    "provider_centroids": "Bij homecare-centra",
    "patient_clusters": "Cluster-centra op basis van patiënten",
    "handmatig": "Handmatig uit CSV",
}


@dataclass
class Config:
    alpha: float = 0.60
    strategy: str = "heaviest_first"
    lookahead_days: int = 7
    horizon_days: int = 30
    timestep_days: int = 1
    avg_speed_kmh: float = 30.0
    distance_norm_hours: float = 2.0
    overcapacity_penalty_weight: float = 10.0
    allow_overcapacity: bool = True
    capacity_buffer_pct: float = 0.0
    random_seed: int = 42

    patient_count: int = 500
    provider_count: int = 3
    base_capacity_hrs_per_week: float = 80.0
    base_initial_load_hrs_per_week: float = 50.0
    synthetic_radius_km: float = 50.0
    los_min_days: int = 14
    los_max_days: int = 42
    visit_hours_min: float = 2.0
    visit_hours_max: float = 7.0

    travel_matrix_mode: str = "bereken"
    isolation_rule: str = "vast_bij_ziekenhuis"

    engine: str = "referentie"
    module_path: str = ""
    module_function_name: str = "rolling_horizon_assignment"
    custom_strategy_code: str = ""

    sensitivity_alpha_values: str = "0.0,0.2,0.4,0.6,0.8,1.0"
    sensitivity_horizon_values: str = "7,14,21,30"


@dataclass
class JobState:
    job_id: str
    progress: float = 0.0
    status: str = "Niet gestart"
    logs: List[str] = field(default_factory=list)
    done: bool = False
    error: Optional[str] = None
    result: Optional[dict] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def log(self, text: str) -> None:
        with self.lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self.logs.append(f"[{ts}] {text}")
            self.logs = self.logs[-200:]

    def set_progress(self, value: float, label: Optional[str] = None) -> None:
        with self.lock:
            self.progress = max(0.0, min(1.0, float(value)))
            if label is not None:
                self.status = label

    def finalize(self, result: Optional[dict] = None, error: Optional[str] = None) -> None:
        with self.lock:
            self.done = True
            self.result = result
            self.error = error
            self.finished_at = time.time()
            if error is None:
                self.progress = 1.0
                self.status = "Klaar"
            else:
                self.status = "Mislukt"


def init_state() -> None:
    if "config_dict" not in st.session_state:
        st.session_state.config_dict = asdict(Config())
    if "job_state" not in st.session_state:
        st.session_state.job_state = None
    if "job_future" not in st.session_state:
        st.session_state.job_future = None
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "scenario_history" not in st.session_state:
        st.session_state.scenario_history = []
    if "sensitivity_df" not in st.session_state:
        st.session_state.sensitivity_df = None


init_state()


@st.cache_resource(show_spinner=False)
def get_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="alloc-dashboard")


def now_date() -> date:
    return datetime.now().date()


def week_start(d: date | pd.Timestamp) -> date:
    if isinstance(d, pd.Timestamp):
        d = d.date()
    return d - timedelta(days=d.weekday())


def generate_week_starts(start: date, end: date) -> List[date]:
    cur = week_start(start)
    weeks = []
    while cur <= end:
        weeks.append(cur)
        cur += timedelta(days=7)
    return weeks


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def parse_numeric_list(text: str, kind=float) -> List[Any]:
    out = []
    for part in [p.strip() for p in str(text).split(",") if p.strip()]:
        try:
            out.append(kind(part))
        except Exception:
            pass
    return out


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return r * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def travel_hours_from_km(distance_km: float, avg_speed_kmh: float) -> float:
    if avg_speed_kmh <= 0:
        return 0.0
    return (distance_km / avg_speed_kmh) * 2.0


def hex_to_rgb(value: str) -> List[int]:
    value = value.lstrip("#")
    if len(value) != 6:
        return [31, 119, 180]
    return [int(value[i:i+2], 16) for i in (0, 2, 4)]


def provider_color(provider_id: str) -> str:
    return COLORS[abs(hash(provider_id)) % len(COLORS)]


def color_map_for_providers(provider_ids: List[str]) -> Dict[str, str]:
    return {pid: provider_color(pid) for pid in provider_ids}


def fmt_num(x: Any, digits: int = 2) -> str:
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and math.isnan(x):
            return "-"
    except Exception:
        pass
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def normalize_providers(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    rename = {}
    aliases = {
        "provider_id": ["provider_id", "provider", "homecare", "organisatie", "org_id"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon", "lng"],
        "capacity_hrs_per_week": ["capacity_hrs_per_week", "capacity", "weekly_capacity"],
        "initial_load_hrs_per_week": ["initial_load_hrs_per_week", "initial_load", "current_load", "baseline_load"],
    }
    for target, opts in aliases.items():
        for opt in opts:
            if opt in cols:
                rename[cols[opt]] = target
                break
    df = df.rename(columns=rename).copy()
    required = ["provider_id", "latitude", "longitude", "capacity_hrs_per_week"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"providers.csv mist kolommen: {missing}")
    if "initial_load_hrs_per_week" not in df.columns:
        df["initial_load_hrs_per_week"] = 0.0
    for c in ["latitude", "longitude", "capacity_hrs_per_week", "initial_load_hrs_per_week"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["provider_id"] = df["provider_id"].astype(str)
    return df.dropna(subset=required).reset_index(drop=True)


def normalize_patients(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    rename = {}
    aliases = {
        "patient_id": ["patient_id", "patient", "id", "pat_id"],
        "discharge_date": ["discharge_date", "start_date", "arrival_date", "date"],
        "length_of_stay": ["length_of_stay", "los", "days", "care_days"],
        "visit_hours": ["visit_hours", "demand_hours", "hours", "care_hours"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon", "lng"],
    }
    for target, opts in aliases.items():
        for opt in opts:
            if opt in cols:
                rename[cols[opt]] = target
                break
    df = df.rename(columns=rename).copy()
    required = ["patient_id", "discharge_date", "length_of_stay", "visit_hours", "latitude", "longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"patients.csv mist kolommen: {missing}")
    df["patient_id"] = df["patient_id"].astype(str)
    df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce").dt.date
    for c in ["length_of_stay", "visit_hours", "latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "type_care" not in df.columns:
        df["type_care"] = "onbekend"
    if "nurse_skill" in df.columns:
        df["nurse_skill"] = pd.to_numeric(df["nurse_skill"], errors="coerce")
    return df.dropna(subset=required).reset_index(drop=True)


def normalize_travel(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    rename = {}
    aliases = {
        "patient_id": ["patient_id", "patient"],
        "provider_id": ["provider_id", "provider"],
        "travel_km": ["travel_km", "distance_km", "distance"],
        "travel_hours": ["travel_hours", "hours", "time_hours"],
    }
    for target, opts in aliases.items():
        for opt in opts:
            if opt in cols:
                rename[cols[opt]] = target
                break
    df = df.rename(columns=rename).copy()
    if "patient_id" not in df.columns or "provider_id" not in df.columns:
        raise ValueError("travel_matrix.csv moet patient_id en provider_id bevatten")
    if "travel_km" not in df.columns and "travel_hours" not in df.columns:
        raise ValueError("travel_matrix.csv moet travel_km of travel_hours bevatten")
    df["patient_id"] = df["patient_id"].astype(str)
    df["provider_id"] = df["provider_id"].astype(str)
    if "travel_km" in df.columns:
        df["travel_km"] = pd.to_numeric(df["travel_km"], errors="coerce")
    if "travel_hours" in df.columns:
        df["travel_hours"] = pd.to_numeric(df["travel_hours"], errors="coerce")
    return df


def normalize_isala(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    rename = {}
    aliases = {
        "location_id": ["location_id", "id", "site_id"],
        "name": ["name", "naam", "location_name"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon", "lng"],
        "location_type": ["location_type", "type", "site_type"],
    }
    for target, opts in aliases.items():
        for opt in opts:
            if opt in cols:
                rename[cols[opt]] = target
                break
    df = df.rename(columns=rename).copy()
    if "location_id" not in df.columns:
        df["location_id"] = [f"L{i+1}" for i in range(len(df))]
    if "name" not in df.columns:
        df["name"] = df["location_id"]
    if "location_type" not in df.columns:
        df["location_type"] = "ziekenhuis"
    required = ["location_id", "name", "latitude", "longitude", "location_type"]
    return df[required].copy()


@st.cache_data(show_spinner=False)
def uploaded_csv_to_df(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


def default_isala_df() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "location_id": "ISALA1",
            "name": DEFAULT_CENTER["naam"],
            "latitude": DEFAULT_CENTER["lat"],
            "longitude": DEFAULT_CENTER["lon"],
            "location_type": "ziekenhuis",
        }]
    )


def generate_demo_providers(config: Config) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    rows = []
    for i in range(config.provider_count):
        if i < len(DEFAULT_HOMECARES):
            base = DEFAULT_HOMECARES[i]
            lat = base["latitude"]
            lon = base["longitude"]
            pid = base["provider_id"]
        else:
            angle = 2 * math.pi * i / max(1, config.provider_count)
            radius_km = 10 + (i % 3) * 7
            lat = DEFAULT_CENTER["lat"] + (radius_km / 111.0) * math.cos(angle)
            lon = DEFAULT_CENTER["lon"] + (
                radius_km / (111.0 * math.cos(math.radians(DEFAULT_CENTER["lat"])))
            ) * math.sin(angle)
            pid = f"Homecare{i+1}"
        rows.append({
            "provider_id": pid,
            "latitude": lat,
            "longitude": lon,
            "capacity_hrs_per_week": round(max(10.0, config.base_capacity_hrs_per_week + rng.normal(0, 8)), 1),
            "initial_load_hrs_per_week": round(max(0.0, config.base_initial_load_hrs_per_week + rng.normal(0, 6)), 1),
        })
    return pd.DataFrame(rows)


def generate_demo_patients(config: Config) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    diagnoses = ["chirurgie", "urologie", "cardiologie", "orthopedie", "revalidatie", "inwendige geneeskunde"]
    start = now_date()
    rows = []
    for i in range(config.patient_count):
        distance_km = rng.uniform(0, max(1.0, config.synthetic_radius_km))
        bearing = rng.uniform(0, 2 * math.pi)
        lat = DEFAULT_CENTER["lat"] + (distance_km / 111.0) * math.cos(bearing)
        lon = DEFAULT_CENTER["lon"] + (
            distance_km / (111.0 * math.cos(math.radians(DEFAULT_CENTER["lat"])))
        ) * math.sin(bearing)
        rows.append({
            "patient_id": f"P{i+1:04d}",
            "discharge_date": start + timedelta(days=int(rng.integers(0, max(1, config.horizon_days)))),
            "length_of_stay": int(rng.integers(config.los_min_days, config.los_max_days + 1)),
            "visit_hours": float(rng.integers(int(config.visit_hours_min), int(config.visit_hours_max) + 1)),
            "latitude": lat,
            "longitude": lon,
            "nurse_skill": int(rng.integers(2, 7)),
            "type_care": rng.choice(diagnoses),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["discharge_date", "visit_hours"], ascending=[True, False]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def computed_distance_matrix(
    patients_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    avg_speed_kmh: float,
) -> pd.DataFrame:
    rows = []
    for _, p in patients_df.iterrows():
        for _, o in providers_df.iterrows():
            dist = haversine_km(float(p["latitude"]), float(p["longitude"]), float(o["latitude"]), float(o["longitude"]))
            rows.append({
                "patient_id": str(p["patient_id"]),
                "provider_id": str(o["provider_id"]),
                "travel_km": dist,
                "travel_hours": travel_hours_from_km(dist, avg_speed_kmh),
            })
    return pd.DataFrame(rows)


def build_distance_lookup(
    patients_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    config: Config,
    travel_df: Optional[pd.DataFrame] = None,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    if travel_df is not None and not travel_df.empty and config.travel_matrix_mode == "upload":
        tdf = normalize_travel(travel_df)
        if "travel_km" not in tdf.columns:
            tdf["travel_km"] = tdf["travel_hours"] * config.avg_speed_kmh / 2.0
        if "travel_hours" not in tdf.columns:
            tdf["travel_hours"] = tdf["travel_km"] / config.avg_speed_kmh * 2.0
    else:
        tdf = computed_distance_matrix(patients_df, providers_df, config.avg_speed_kmh)
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for _, row in tdf.iterrows():
        out[(str(row["patient_id"]), str(row["provider_id"]))] = {
            "travel_km": float(row["travel_km"]),
            "travel_hours": float(row["travel_hours"]),
        }
    return out


def infer_isolation_locations(
    patients_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    isala_df: Optional[pd.DataFrame],
    config: Config,
) -> pd.DataFrame:
    if config.isolation_rule == "handmatig" and isala_df is not None and not isala_df.empty:
        return normalize_isala(isala_df)

    if config.isolation_rule == "vast_bij_ziekenhuis":
        return default_isala_df() if isala_df is None or isala_df.empty else normalize_isala(isala_df)

    if config.isolation_rule == "provider_centroids":
        temp = providers_df[["provider_id", "latitude", "longitude"]].copy()
        temp["location_id"] = temp["provider_id"]
        temp["name"] = "Isolatiepunt " + temp["provider_id"].astype(str)
        temp["location_type"] = "isolatie"
        return temp[["location_id", "name", "latitude", "longitude", "location_type"]]

    if patients_df.empty:
        return default_isala_df()

    temp = patients_df[["latitude", "longitude"]].sort_values(["latitude", "longitude"]).reset_index(drop=True)
    groups = np.array_split(temp, min(max(1, len(providers_df)), 5))
    rows = []
    for i, grp in enumerate(groups, start=1):
        rows.append({
            "location_id": f"ISO{i}",
            "name": f"Cluster-isolatie {i}",
            "latitude": float(grp["latitude"].mean()),
            "longitude": float(grp["longitude"].mean()),
            "location_type": "isolatie",
        })
    return pd.DataFrame(rows)


# ---- custom strategie-registry ----

CustomStrategyFn = Callable[[pd.Series, pd.DataFrame, Dict[str, Any]], pd.DataFrame]
CUSTOM_REGISTRY: Dict[str, CustomStrategyFn] = {}


def register_builtin_custom_strategies() -> None:
    def korte_afstand_dan_load(patient_row: pd.Series, candidate_df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        return candidate_df.sort_values(["distance_normalized", "load", "provider_id"]).reset_index(drop=True)

    CUSTOM_REGISTRY["korte_afstand_dan_load"] = korte_afstand_dan_load


register_builtin_custom_strategies()


def load_user_custom_strategy_code(code_text: str) -> None:
    if not code_text.strip():
        return
    namespace: Dict[str, Any] = {}
    exec(code_text, {}, namespace)
    for name, obj in namespace.items():
        if callable(obj):
            CUSTOM_REGISTRY[name] = obj


def patient_active_weeks(patient_row: pd.Series, planning_end: date) -> List[date]:
    start = pd.to_datetime(patient_row["discharge_date"]).date()
    end = min(start + timedelta(days=int(patient_row["length_of_stay"])), planning_end)
    if end < start:
        return []
    return [w for w in generate_week_starts(start, end) if w < end]


def order_patients(window_df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    df = window_df.copy()
    if strategy == "heaviest_first":
        return df.sort_values(["visit_hours", "discharge_date", "patient_id"], ascending=[False, True, True]).reset_index(drop=True)
    if strategy == "round_robin":
        return df.sort_values(["discharge_date", "patient_id"]).reset_index(drop=True)
    if strategy == "distance_based":
        return df.sort_values(["discharge_date", "visit_hours"], ascending=[True, False]).reset_index(drop=True)
    if strategy == "capacity_proportional":
        return df.sort_values(["visit_hours", "discharge_date"], ascending=[False, True]).reset_index(drop=True)
    return df.sort_values(["visit_hours", "discharge_date", "patient_id"], ascending=[False, True, True]).reset_index(drop=True)


def build_candidate_scores(
    patient_row: pd.Series,
    providers_df: pd.DataFrame,
    remaining_capacity: Dict[Tuple[str, date], float],
    active_weeks: List[date],
    distance_lookup: Dict[Tuple[str, str], Dict[str, float]],
    config: Config,
    rr_pointer: Dict[str, int],
) -> pd.DataFrame:
    rows = []
    patient_id = str(patient_row["patient_id"])
    provider_ids = list(providers_df["provider_id"].astype(str))
    rr_idx = rr_pointer["value"]

    for idx, provider_row in providers_df.reset_index(drop=True).iterrows():
        provider_id = str(provider_row["provider_id"])
        cap = safe_float(provider_row["capacity_hrs_per_week"]) * (1 + config.capacity_buffer_pct / 100.0)
        lookup = distance_lookup[(patient_id, provider_id)]
        travel_km = lookup["travel_km"]
        travel_hours = lookup["travel_hours"]
        visit_hours = safe_float(patient_row["visit_hours"])
        needed = visit_hours + travel_hours

        loads = []
        deficits = []
        for w in active_weeks:
            rem = remaining_capacity.get((provider_id, w), cap)
            loads.append(1 - rem / cap if cap > 0 else 0.0)
            deficits.append(max(0.0, needed - rem))

        load = max(loads) if loads else 0.0
        deficit = max(deficits) if deficits else 0.0
        distance_normalized = min(travel_hours / max(0.1, config.distance_norm_hours), 1.0)
        infeasible = (not config.allow_overcapacity) and deficit > 0.0
        penalty = (1e9 if infeasible else 0.0) if not config.allow_overcapacity else config.overcapacity_penalty_weight * deficit
        score_base = config.alpha * load + (1 - config.alpha) * distance_normalized + penalty

        if config.strategy == "distance_based":
            strategy_bias = distance_normalized
        elif config.strategy == "round_robin":
            desired_idx = rr_idx % max(1, len(provider_ids))
            circular_distance = min(abs(idx - desired_idx), len(provider_ids) - abs(idx - desired_idx)) if len(provider_ids) > 1 else 0
            strategy_bias = circular_distance * 0.01
        elif config.strategy == "capacity_proportional":
            free_ratios = []
            for w in active_weeks:
                rem = remaining_capacity.get((provider_id, w), cap)
                free_ratios.append(rem / cap if cap > 0 else 0.0)
            strategy_bias = -(min(free_ratios) if free_ratios else 0.0)
        else:
            strategy_bias = 0.0

        rows.append({
            "provider_id": provider_id,
            "travel_km": travel_km,
            "travel_hours": travel_hours,
            "distance_normalized": distance_normalized,
            "load": load,
            "deficit": deficit,
            "penalty": penalty,
            "score": score_base + strategy_bias,
            "infeasible": infeasible,
        })

    return pd.DataFrame(rows).sort_values(["score", "provider_id"]).reset_index(drop=True)


def analyse_from_assignments(
    assignments_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    config: Config,
    isala_df: Optional[pd.DataFrame],
    runtime_sec: float,
    objective_source: str = "surrogaat in dashboard",
) -> dict:
    providers_df = normalize_providers(providers_df)
    assignments_df = assignments_df.copy()
    assignments_df["discharge_date"] = pd.to_datetime(assignments_df["discharge_date"], errors="coerce").dt.date
    if "assigned_at_timestep" not in assignments_df.columns:
        assignments_df["assigned_at_timestep"] = assignments_df["discharge_date"]

    planning_start = pd.to_datetime(assignments_df["discharge_date"]).min().date()
    planning_end = planning_start + timedelta(days=int(config.horizon_days))
    all_weeks = generate_week_starts(planning_start, planning_end + timedelta(days=int(config.los_max_days)))

    # Activiteit per week
    act_rows = []
    for _, row in assignments_df.dropna(subset=["assigned_provider"]).iterrows():
        start = pd.to_datetime(row["discharge_date"]).date()
        care_end = start + timedelta(days=int(row["length_of_stay"]))
        for w in generate_week_starts(start, min(care_end, planning_end + timedelta(days=int(config.los_max_days)))):
            if w >= care_end:
                continue
            act_rows.append({
                "patient_id": str(row["patient_id"]),
                "provider_id": str(row["assigned_provider"]),
                "week_start": w,
                "visit_hours": safe_float(row["visit_hours"]),
                "travel_hours": safe_float(row.get("travel_hours")),
                "care_load_hrs": safe_float(row["visit_hours"]) + safe_float(row.get("travel_hours")),
            })
    activity_df = pd.DataFrame(act_rows)

    # Bezetting
    occ_rows = []
    for _, provider_row in providers_df.iterrows():
        provider_id = str(provider_row["provider_id"])
        cap = safe_float(provider_row["capacity_hrs_per_week"]) * (1 + config.capacity_buffer_pct / 100.0)
        initial_load = safe_float(provider_row.get("initial_load_hrs_per_week"))
        this_activity = activity_df[activity_df["provider_id"] == provider_id].copy() if not activity_df.empty else pd.DataFrame()
        this_assign = assignments_df[assignments_df["assigned_provider"] == provider_id].copy()
        for w in all_weeks:
            extra_used = this_activity.loc[this_activity["week_start"] == w, "care_load_hrs"].sum() if not this_activity.empty else 0.0
            used = initial_load + extra_used
            remaining = cap - used
            util_pct = (used / cap * 100) if cap > 0 else 0.0
            occ_rows.append({
                "provider_id": provider_id,
                "week_start": w,
                "capacity_hrs_per_week": cap,
                "initial_load_hrs_per_week": initial_load,
                "used_hrs": used,
                "remaining_hrs": remaining,
                "utilization_pct": util_pct,
                "active_patients": this_activity.loc[this_activity["week_start"] == w, "patient_id"].nunique() if not this_activity.empty else 0,
                "new_assignments": this_assign[this_assign["assigned_at_timestep"].apply(lambda x: week_start(pd.to_datetime(x).date()) if pd.notna(x) else None) == w]["patient_id"].nunique() if not this_assign.empty else 0,
                "overcapacity": util_pct > 100,
            })
    occupancy_df = pd.DataFrame(occ_rows)

    # Tijdsreeks toewijzingen
    ts_df = (
        assignments_df.assign(assigned_at_timestep=pd.to_datetime(assignments_df["assigned_at_timestep"], errors="coerce").dt.date)
        .groupby(["assigned_at_timestep", "assigned_provider"], dropna=False)["patient_id"]
        .nunique()
        .reset_index(name="assigned_patients")
    )

    # Routes
    p_lookup = providers_df.set_index("provider_id")[["latitude", "longitude"]].to_dict("index")
    route_rows = []
    for _, row in assignments_df.dropna(subset=["assigned_provider"]).iterrows():
        pid = str(row["assigned_provider"])
        if pid not in p_lookup:
            continue
        base = p_lookup[pid]
        route_rows.append({
            "patient_id": str(row["patient_id"]),
            "provider_id": pid,
            "from_lat": float(base["latitude"]),
            "from_lon": float(base["longitude"]),
            "to_lat": float(row["latitude"]),
            "to_lon": float(row["longitude"]),
            "travel_km": safe_float(row.get("travel_km")),
            "travel_hours": safe_float(row.get("travel_hours")),
        })
    routes_df = pd.DataFrame(route_rows)

    # KPI's
    total_assigned = int(assignments_df["assigned_provider"].notna().sum())
    unassigned_count = int(assignments_df["assigned_provider"].isna().sum())
    avg_travel_km = float(assignments_df["travel_km"].dropna().mean()) if assignments_df["travel_km"].notna().any() else 0.0
    avg_travel_hrs = float(assignments_df["travel_hours"].dropna().mean()) if assignments_df["travel_hours"].notna().any() else 0.0
    avg_util_by_provider = occupancy_df.groupby("provider_id")["utilization_pct"].mean().to_dict() if not occupancy_df.empty else {}
    util_values = list(avg_util_by_provider.values())
    util_std = float(np.std(util_values)) if util_values else 0.0
    total_overcapacity_weeks = int(occupancy_df["overcapacity"].sum()) if not occupancy_df.empty else 0
    total_overcapacity_hours = float(
        occupancy_df.assign(over_hrs=lambda d: np.maximum(d["used_hrs"] - d["capacity_hrs_per_week"], 0))["over_hrs"].sum()
    ) if not occupancy_df.empty else 0.0
    objective_value = config.alpha * util_std + (1 - config.alpha) * avg_travel_hrs + 100.0 * unassigned_count

    if isala_df is None or isala_df.empty:
        isala_df = default_isala_df()
    else:
        isala_df = normalize_isala(isala_df)

    isolation_df = infer_isolation_locations(
        assignments_df.drop(columns=[c for c in ["assigned_provider", "travel_km", "travel_hours", "assigned_at_timestep"] if c in assignments_df.columns], errors="ignore"),
        providers_df,
        isala_df,
        config,
    )

    return {
        "summary": {
            "total_assigned": total_assigned,
            "unassigned_count": unassigned_count,
            "avg_travel_km": avg_travel_km,
            "avg_travel_hrs": avg_travel_hrs,
            "utilization_std_dev_pct": util_std,
            "total_overcapacity_weeks": total_overcapacity_weeks,
            "total_overcapacity_hours": total_overcapacity_hours,
            "objective_value": objective_value,
            "runtime_sec": runtime_sec,
            "providers": int(providers_df["provider_id"].nunique()),
            "patients": int(assignments_df["patient_id"].nunique()),
            "strategy": config.strategy,
            "objective_source": objective_source,
            "avg_utilization_by_provider": avg_util_by_provider,
        },
        "assignments_df": assignments_df,
        "occupancy_df": occupancy_df,
        "activity_df": activity_df,
        "time_series_assignments_df": ts_df,
        "routes_df": routes_df,
        "providers_df": providers_df,
        "isala_df": isala_df,
        "isolation_df": isolation_df,
    }


def run_reference_allocator(
    patients_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    config: Config,
    travel_df: Optional[pd.DataFrame] = None,
    isala_df: Optional[pd.DataFrame] = None,
    job_state: Optional[JobState] = None,
) -> dict:
    t0 = time.perf_counter()

    patients_df = normalize_patients(patients_df)
    providers_df = normalize_providers(providers_df)

    if config.strategy == "custom_pipeline":
        load_user_custom_strategy_code(config.custom_strategy_code)

    if job_state:
        job_state.log("Referentie-allocator gestart")
        job_state.set_progress(0.05, "Data normaliseren en afstandsmatrix opbouwen")

    distance_lookup = build_distance_lookup(patients_df, providers_df, config, travel_df)

    planning_start = pd.to_datetime(patients_df["discharge_date"]).min().date()
    planning_end = planning_start + timedelta(days=int(config.horizon_days))
    planning_points = []
    cur = planning_start
    while cur <= planning_end:
        planning_points.append(cur)
        cur += timedelta(days=max(1, config.timestep_days))

    all_weeks = generate_week_starts(planning_start, planning_end + timedelta(days=int(config.los_max_days)))

    remaining_capacity: Dict[Tuple[str, date], float] = {}
    for _, row in providers_df.iterrows():
      provider_id = str(row["provider_id"])
      cap = safe_float(row["capacity_hrs_per_week"]) * (1 + config.capacity_buffer_pct / 100.0)
      initial_load = safe_float(row.get("initial_load_hrs_per_week"))
      for w in all_weeks:
          remaining_capacity[(provider_id, w)] = cap - initial_load

    assigned: set[str] = set()
    rr_pointer = {"value": 0}
    assignment_rows = []
    process_log_rows = []

    total_steps = max(1, len(planning_points))

    for idx, t in enumerate(planning_points, start=1):
        window_end = min(t + timedelta(days=int(config.lookahead_days)), planning_end)
        window_df = patients_df[
            (patients_df["discharge_date"] >= t)
            & (patients_df["discharge_date"] <= window_end)
            & (~patients_df["patient_id"].isin(assigned))
        ].copy()

        if job_state:
            job_state.set_progress(0.05 + 0.80 * (idx / total_steps), f"Rolling horizon stap {idx}/{total_steps}")
            job_state.log(f"Venster {t} t/m {window_end}: {len(window_df)} bekende patiënten")

        if window_df.empty:
            continue

        ordered = order_patients(window_df, config.strategy)

        for _, p in ordered.iterrows():
            patient_id = str(p["patient_id"])
            if patient_id in assigned:
                continue

            active_weeks = patient_active_weeks(p, planning_end)
            cand = build_candidate_scores(p, providers_df, remaining_capacity, active_weeks, distance_lookup, config, rr_pointer)

            if config.strategy == "custom_pipeline":
                context = {
                    "config": config,
                    "remaining_capacity": remaining_capacity,
                    "active_weeks": active_weeks,
                }
                for name, fn in CUSTOM_REGISTRY.items():
                    try:
                        cand = fn(p, cand.copy(), context)
                    except Exception:
                        pass
                cand = cand.sort_values(["score", "provider_id"]).reset_index(drop=True)

            feasible = cand if config.allow_overcapacity else cand[~cand["infeasible"]].copy()

            if feasible.empty:
                assigned.add(patient_id)
                assignment_rows.append({
                    **p.to_dict(),
                    "assigned_provider": None,
                    "travel_km": np.nan,
                    "travel_hours": np.nan,
                    "assigned_at_timestep": t,
                    "unmet_demand": True,
                    "score": np.nan,
                })
                process_log_rows.append({
                    "patient_id": patient_id,
                    "decision_time": t,
                    "decision": "unassigned",
                    "reason": "geen haalbare provider binnen harde capaciteit",
                })
                continue

            best = feasible.iloc[0]
            provider_id = str(best["provider_id"])
            travel_km = float(best["travel_km"])
            travel_hours = float(best["travel_hours"])
            total_load = safe_float(p["visit_hours"]) + travel_hours

            for w in active_weeks:
                remaining_capacity[(provider_id, w)] = remaining_capacity.get((provider_id, w), 0.0) - total_load

            assignment_rows.append({
                **p.to_dict(),
                "assigned_provider": provider_id,
                "travel_km": travel_km,
                "travel_hours": travel_hours,
                "assigned_at_timestep": t,
                "unmet_demand": False,
                "score": float(best["score"]),
                "load_component": float(best["load"]),
                "penalty_component": float(best["penalty"]),
            })
            process_log_rows.append({
                "patient_id": patient_id,
                "decision_time": t,
                "decision": "assigned",
                "assigned_provider": provider_id,
                "travel_km": travel_km,
                "score": float(best["score"]),
            })
            assigned.add(patient_id)

            if config.strategy == "round_robin":
                rr_pointer["value"] += 1

    assignments_df = pd.DataFrame(assignment_rows)
    if assignments_df.empty:
        assignments_df = patients_df.copy()
        assignments_df["assigned_provider"] = None
        assignments_df["travel_km"] = np.nan
        assignments_df["travel_hours"] = np.nan
        assignments_df["assigned_at_timestep"] = assignments_df["discharge_date"]
        assignments_df["unmet_demand"] = True
        assignments_df["score"] = np.nan

    result = analyse_from_assignments(assignments_df, providers_df, config, isala_df, runtime_sec=time.perf_counter() - t0)
    result["process_log_df"] = pd.DataFrame(process_log_rows)
    result["config"] = asdict(config)

    if job_state:
        job_state.log("Referentie-allocator afgerond")
        job_state.set_progress(0.98, "KPI's en kaarten opbouwen")

    return result


class UserModuleAdapter:
    def __init__(self, path: str, function_name: str) -> None:
        self.path = Path(path)
        self.function_name = function_name
        spec = importlib.util.spec_from_file_location(f"user_alloc_{uuid.uuid4().hex}", str(self.path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Kan module niet laden vanaf: {path}")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)  # type: ignore[attr-defined]
        if not hasattr(self.module, function_name):
            raise AttributeError(f"Functie '{function_name}' niet gevonden in {path}")
        self.fn = getattr(self.module, function_name)

    def _build_dataclass_objects(self, patients_df: pd.DataFrame, providers_df: pd.DataFrame):
        PatientCls = getattr(self.module, "Patient", None)
        ProviderCls = getattr(self.module, "Provider", None)
        if PatientCls is None or ProviderCls is None:
            return None, None

        patients = []
        providers = []
        for _, row in patients_df.iterrows():
            patients.append(
                PatientCls(
                    patient_id=str(row["patient_id"]),
                    discharge_date=pd.to_datetime(row["discharge_date"]).date(),
                    length_of_stay=int(row["length_of_stay"]),
                    visit_hours=float(row["visit_hours"]),
                    home_coords=(float(row["latitude"]), float(row["longitude"])),
                )
            )
        for _, row in providers_df.iterrows():
            providers.append(
                ProviderCls(
                    provider_id=str(row["provider_id"]),
                    coords=(float(row["latitude"]), float(row["longitude"])),
                    capacity_hrs_per_week=float(row["capacity_hrs_per_week"]),
                    initial_load_hrs_per_week=float(row.get("initial_load_hrs_per_week", 0.0)),
                )
            )
        return patients, providers

    def run(
        self,
        patients_df: pd.DataFrame,
        providers_df: pd.DataFrame,
        config: Config,
        travel_df: Optional[pd.DataFrame],
        isala_df: Optional[pd.DataFrame],
        job_state: Optional[JobState],
    ) -> dict:
        t0 = time.perf_counter()

        # Voor alternatieve strategieën valt de app terug op de referentie-allocator.
        if config.strategy != "heaviest_first":
            if job_state:
                job_state.log("Eigen module ondersteunt mogelijk geen alternatieve strategieën; referentie-allocator wordt gebruikt.")
            return run_reference_allocator(patients_df, providers_df, config, travel_df, isala_df, job_state)

        if hasattr(self.module, "OVERCAPACITY_PENALTY_WEIGHT"):
            setattr(self.module, "OVERCAPACITY_PENALTY_WEIGHT", config.overcapacity_penalty_weight)

        if job_state:
            job_state.log(f"Eigen module geladen: {self.path.name}")
            job_state.set_progress(0.15, "Eigen allocatiefunctie uitvoeren")

        sig = inspect.signature(self.fn)
        kwargs = {}
        patients_obj, providers_obj = self._build_dataclass_objects(patients_df, providers_df)

        if "patients" in sig.parameters and "providers" in sig.parameters:
            kwargs["patients"] = patients_obj if patients_obj is not None else patients_df
            kwargs["providers"] = providers_obj if providers_obj is not None else providers_df
        elif "patients_df" in sig.parameters and "providers_df" in sig.parameters:
            kwargs["patients_df"] = patients_df
            kwargs["providers_df"] = providers_df
        else:
            kwargs["patients"] = patients_df
            kwargs["providers"] = providers_df

        passthrough = {
            "alpha": config.alpha,
            "lookahead_days": config.lookahead_days,
            "avg_speed_kmh": config.avg_speed_kmh,
            "horizon_days": config.horizon_days,
            "timestep_days": config.timestep_days,
            "strategy": config.strategy,
        }
        for k, v in passthrough.items():
            if k in sig.parameters:
                kwargs[k] = v

        raw = self.fn(**kwargs)

        if not isinstance(raw, dict):
            raise TypeError("Eigen allocatiefunctie moet een dictionary retourneren")

        assignments = raw.get("assignments")
        kpis = raw.get("kpis", {})
        remaining_capacity = raw.get("remaining_capacity", {})

        assignments_df = patients_df.copy()
        assignments_df["assigned_provider"] = None
        assignments_df["travel_km"] = np.nan
        assignments_df["travel_hours"] = np.nan
        assignments_df["assigned_at_timestep"] = assignments_df["discharge_date"]
        assignments_df["unmet_demand"] = True

        if isinstance(assignments, dict):
            assignments_df["assigned_provider"] = assignments_df["patient_id"].map(assignments)
        elif isinstance(assignments, pd.DataFrame) and "patient_id" in assignments.columns:
            cand = assignments.copy()
            if "provider_id" in cand.columns and "assigned_provider" not in cand.columns:
                cand = cand.rename(columns={"provider_id": "assigned_provider"})
            assignments_df = assignments_df.merge(cand[["patient_id", "assigned_provider"]], on="patient_id", how="left", suffixes=("", "_m"))
            if "assigned_provider_m" in assignments_df.columns:
                assignments_df["assigned_provider"] = assignments_df["assigned_provider_m"]
                assignments_df = assignments_df.drop(columns=["assigned_provider_m"])
        else:
            raise TypeError("assignments moet een dict of DataFrame zijn")

        distance_lookup = build_distance_lookup(patients_df, providers_df, config, travel_df)

        def _km(row):
            if pd.isna(row["assigned_provider"]):
                return np.nan
            return distance_lookup[(str(row["patient_id"]), str(row["assigned_provider"]))]["travel_km"]

        def _hrs(row):
            if pd.isna(row["assigned_provider"]):
                return np.nan
            return distance_lookup[(str(row["patient_id"]), str(row["assigned_provider"]))]["travel_hours"]

        mask = assignments_df["assigned_provider"].notna()
        assignments_df.loc[mask, "travel_km"] = assignments_df.loc[mask].apply(_km, axis=1)
        assignments_df.loc[mask, "travel_hours"] = assignments_df.loc[mask].apply(_hrs, axis=1)
        assignments_df["unmet_demand"] = assignments_df["assigned_provider"].isna()

        result = analyse_from_assignments(
            assignments_df=assignments_df,
            providers_df=providers_df,
            config=config,
            isala_df=isala_df,
            runtime_sec=time.perf_counter() - t0,
            objective_source="eigen module + dashboardafleiding",
        )
        result["summary"]["module_kpis_raw"] = kpis
        result["summary"]["remaining_capacity_raw"] = remaining_capacity
        result["process_log_df"] = pd.DataFrame()
        result["config"] = asdict(config)
        return result


def run_job(
    config_dict: dict,
    patients_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    travel_df: Optional[pd.DataFrame],
    isala_df: Optional[pd.DataFrame],
    module_path: Optional[str],
    job_state: JobState,
) -> None:
    try:
        config = Config(**config_dict)
        job_state.log("Job geaccepteerd")
        if config.engine == "eigen_module" and module_path:
            result = UserModuleAdapter(module_path, config.module_function_name).run(
                patients_df, providers_df, config, travel_df, isala_df, job_state
            )
        else:
            result = run_reference_allocator(patients_df, providers_df, config, travel_df, isala_df, job_state)
        job_state.finalize(result=result)
    except Exception as exc:
        job_state.log(f"Fout: {exc}")
        job_state.finalize(error=f"{exc}\n\n{traceback.format_exc()}")


def submit_job(
    config_dict: dict,
    patients_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    travel_df: Optional[pd.DataFrame],
    isala_df: Optional[pd.DataFrame],
    module_path: Optional[str],
) -> Tuple[JobState, Future]:
    state = JobState(job_id=uuid.uuid4().hex)
    future = get_executor().submit(
        run_job,
        config_dict,
        patients_df.copy(),
        providers_df.copy(),
        None if travel_df is None else travel_df.copy(),
        None if isala_df is None else isala_df.copy(),
        module_path,
        state,
    )
    return state, future


# ---- visualisaties ----

def occupancy_chart(occupancy_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        occupancy_df,
        x="week_start",
        y="utilization_pct",
        color="provider_id",
        markers=True,
        title="Bezettingsgraad per homecare over de tijd",
    )
    fig.add_hline(y=100.0, line_dash="dash", annotation_text="100% capaciteit")
    fig.update_layout(xaxis_title="Week", yaxis_title="Bezettingsgraad (%)", legend_title_text="Homecare")
    return fig


def active_patients_chart(occupancy_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    cmap = color_map_for_providers(list(occupancy_df["provider_id"].astype(str).unique()))
    for provider_id, grp in occupancy_df.groupby("provider_id"):
        fig.add_trace(
            go.Scatter(
                x=grp["week_start"],
                y=grp["active_patients"],
                mode="lines",
                stackgroup="one",
                name=str(provider_id),
                line=dict(color=cmap.get(str(provider_id))),
            )
        )
    fig.update_layout(title="Actieve patiënten per homecare over de tijd", xaxis_title="Week", yaxis_title="Aantal actieve patiënten")
    return fig


def assignment_ts_chart(ts_df: pd.DataFrame) -> go.Figure:
    df = ts_df.dropna(subset=["assigned_at_timestep", "assigned_provider"]).copy()
    if df.empty:
        return go.Figure()
    fig = px.line(
        df,
        x="assigned_at_timestep",
        y="assigned_patients",
        color="assigned_provider",
        markers=True,
        title="Nieuwe toewijzingen per tijdstap",
    )
    fig.update_layout(xaxis_title="Tijdstap", yaxis_title="Aantal toegewezen patiënten")
    return fig


def travel_heatmap(assignments_df: pd.DataFrame) -> go.Figure:
    df = assignments_df.dropna(subset=["assigned_provider"]).copy()
    if df.empty:
        return go.Figure()
    piv = (
        df.groupby(["assigned_provider", "type_care"])["travel_km"]
        .mean()
        .reset_index()
        .pivot(index="assigned_provider", columns="type_care", values="travel_km")
        .fillna(0)
    )
    fig = go.Figure(data=go.Heatmap(z=piv.values, x=list(piv.columns), y=list(piv.index), colorbar_title="Gem. km"))
    fig.update_layout(title="Gemiddelde reisafstand per homecare en zorgtype")
    return fig


def map_deck(
    providers_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    isala_df: pd.DataFrame,
    isolation_df: pd.DataFrame,
    show_routes: bool = True,
) -> pdk.Deck:
    provider_ids = list(providers_df["provider_id"].astype(str).unique())
    cmap = color_map_for_providers(provider_ids)

    prov_map = providers_df.copy()
    prov_map["fill_color"] = prov_map["provider_id"].astype(str).map(
        lambda x: hex_to_rgb(cmap.get(x, "#1f77b4")) + [220]
    )

    pat_map = assignments_df.copy()
    pat_map["fill_color"] = pat_map["assigned_provider"].astype(str).map(
        lambda x: hex_to_rgb(cmap.get(x, "#999999")) + [120]
    )
    pat_map.loc[pat_map["assigned_provider"].isna(), "fill_color"] = [[120, 120, 120, 120]] * int(
        pat_map["assigned_provider"].isna().sum()
    )

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=pat_map,
            get_position="[longitude, latitude]",
            get_fill_color="fill_color",
            get_radius=350,
            pickable=True,
            auto_highlight=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=prov_map,
            get_position="[longitude, latitude]",
            get_fill_color="fill_color",
            get_radius=1800,
            pickable=True,
            auto_highlight=True,
        ),
    ]

    if isala_df is not None and not isala_df.empty:
        temp = isala_df.copy()
        temp["fill_color"] = [[0, 0, 0, 255] for _ in range(len(temp))]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=temp,
                get_position="[longitude, latitude]",
                get_fill_color="fill_color",
                get_radius=2500,
                pickable=True,
            )
        )

    if isolation_df is not None and not isolation_df.empty:
        temp = isolation_df.copy()
        temp["fill_color"] = [[255, 0, 0, 180] for _ in range(len(temp))]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=temp,
                get_position="[longitude, latitude]",
                get_fill_color="fill_color",
                get_radius=2200,
                pickable=True,
            )
        )

    if show_routes and routes_df is not None and not routes_df.empty:
        temp = routes_df.copy()
        temp["source_color"] = temp["provider_id"].astype(str).map(
            lambda x: hex_to_rgb(cmap.get(x, "#1f77b4")) + [170]
        )
        temp["target_color"] = temp["provider_id"].astype(str).map(
            lambda x: hex_to_rgb(cmap.get(x, "#1f77b4")) + [50]
        )
        layers.append(
            pdk.Layer(
                "ArcLayer",
                data=temp,
                get_source_position="[from_lon, from_lat]",
                get_target_position="[to_lon, to_lat]",
                get_source_color="source_color",
                get_target_color="target_color",
                get_width=2,
                pickable=True,
            )
        )

    lats = list(providers_df["latitude"].astype(float)) + list(assignments_df["latitude"].astype(float))
    lons = list(providers_df["longitude"].astype(float)) + list(assignments_df["longitude"].astype(float))
    center_lat = float(np.mean(lats)) if lats else DEFAULT_CENTER["lat"]
    center_lon = float(np.mean(lons)) if lons else DEFAULT_CENTER["lon"]

    return pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=8.2, pitch=35),
        layers=layers,
        tooltip={"html": "<b>{provider_id}</b><br/>{patient_id}<br/>km: {travel_km}", "style": {"color": "white"}},
    )


def render_mermaid(code: str, title: str) -> None:
    escaped = html.escape(code)
    unique_id = f"mermaid_{uuid.uuid4().hex}"
    block = f"""
    <div>
      <h4>{html.escape(title)}</h4>
      <pre class="mermaid" id="{unique_id}">{escaped}</pre>
      <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
      </script>
    </div>
    """
    st.html(block, unsafe_allow_javascript=True)
    with st.expander("Toon Mermaid-code"):
        st.code(code, language="mermaid")


def mermaid_flowchart(config: Config) -> str:
    return textwrap.dedent(
        f"""
        flowchart TD
            A[Data inladen of demo genereren] --> B[Parameters ophalen]
            B --> C[Rolling horizon vensters maken]
            C --> D{{Strategie kiezen}}
            D -->|Heaviest first| E[Prioriteer hoge zorgvraag]
            D -->|Round robin| F[Verdeel cyclisch]
            D -->|Distance based| G[Kies primair korte afstand]
            D -->|Capacity proportional| H[Stuur op vrije capaciteit]
            D -->|Custom pipeline| I[Voer eigen stappenplan uit]
            E --> J[Score provider per patiënt]
            F --> J
            G --> J
            H --> J
            I --> J
            J --> K[Update resterende capaciteit]
            K --> L[KPI's, grafieken en kaarten]
            L --> M[Scenario opslaan of exporteren]
        """
    ).strip()


def mermaid_timeline(config: Config) -> str:
    return textwrap.dedent(
        f"""
        timeline
            title Rolling-horizon proces
            T0 : Data laden of demo genereren
            T1 : Parameters instellen : alpha = {config.alpha}
            T2 : Lookahead-venster : {config.lookahead_days} dagen
            T3 : Strategie : {STRATEGIES.get(config.strategy, config.strategy)}
            T4 : Toewijzen aan homecare
            T5 : KPI's berekenen
            T6 : Kaarten en scenariovergelijking tonen
        """
    ).strip()


def strategy_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Strategie": "Zwaarste last eerst",
            "Logica": "Sorteert patiënten op hoogste zorgvraag; score combineert load, afstand en overcapaciteitsstraf.",
            "Sterk punt": "Sterk voor balancing onder hoge vraag.",
            "Risico": "Kan afstandsoptimaliteit missen.",
        },
        {
            "Strategie": "Round-robin",
            "Logica": "Verdeelt toewijzingen cyclisch over providers.",
            "Sterk punt": "Eenvoudig en uitlegbaar.",
            "Risico": "Kan meer reisafstand geven.",
        },
        {
            "Strategie": "Kortste afstand eerst",
            "Logica": "Minimaliseert primair reistijd/afstand.",
            "Sterk punt": "Compact werkgebied en minder reistijd.",
            "Risico": "Kan balans verslechteren.",
        },
        {
            "Strategie": "Capaciteit-proportioneel",
            "Logica": "Stuurt op vrije capaciteit per provider.",
            "Sterk punt": "Past goed bij tactische spreiding.",
            "Risico": "Patiënten kunnen verder weg terechtkomen.",
        },
        {
            "Strategie": "Custom stappenplan",
            "Logica": "Laat eigen candidate-reordering of heuristische stappen toe.",
            "Sterk punt": "Flexibel voor experimenten.",
            "Risico": "Meer validatie nodig.",
        },
    ])


def parameter_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"Parameter": "alpha", "Status": "Gevonden in huidige code", "Default": 0.60, "Effect": "Balans tussen load balancing en afstand"},
        {"Parameter": "lookahead_days", "Status": "Gevonden in huidige code", "Default": 7, "Effect": "Aantal dagen vooruitkijken per rolling-horizon stap"},
        {"Parameter": "avg_speed_kmh", "Status": "Gevonden in huidige code", "Default": 30.0, "Effect": "Converteert afstand naar reistijd"},
        {"Parameter": "overcapacity_penalty_weight", "Status": "Gevonden in huidige code", "Default": 10.0, "Effect": "Straft capaciteitstekort in score"},
        {"Parameter": "capacity_hrs_per_week", "Status": "Gevonden in providers.csv", "Default": 80.0, "Effect": "Capaciteitslimiet per provider"},
        {"Parameter": "initial_load_hrs_per_week", "Status": "Gevonden in providers.csv", "Default": 50.0, "Effect": "Startbelasting van lopende caseload"},
        {"Parameter": "distance_norm_hours", "Status": "Impliciet in huidige code", "Default": 2.0, "Effect": "Normalisatie van afstandscomponent"},
        {"Parameter": "timestep_days", "Status": "Niet expliciet; default toegevoegd", "Default": 1, "Effect": "Herplanfrequentie"},
        {"Parameter": "horizon_days", "Status": "Niet expliciet als losse UI-parameter; default toegevoegd", "Default": 30, "Effect": "Analysehorizon"},
        {"Parameter": "random_seed", "Status": "Niet expliciet; default toegevoegd", "Default": 42, "Effect": "Reproduceerbare demo's en sweeps"},
        {"Parameter": "travel_matrix_mode", "Status": "Niet expliciet; default toegevoegd", "Default": "bereken", "Effect": "Uploadmatrix of berekende matrix"},
        {"Parameter": "isolation_rule", "Status": "Niet gespecificeerd; default toegevoegd", "Default": "vast_bij_ziekenhuis", "Effect": "Plaatsingslogica voor Isala-/isolatiepunten"},
    ])


def output_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"Output": "Toewijzingen per tijdstap", "Beschrijving": "Welke patiënt in welke rolling-horizon stap aan welke homecare is toegewezen"},
        {"Output": "Bezettingsgraad per week", "Beschrijving": "Gebruikte uren, resterende uren en overschrijding per provider"},
        {"Output": "Actieve patiënten per provider", "Beschrijving": "Aantal patiënten per week in zorg"},
        {"Output": "Onvervulde vraag", "Beschrijving": "Patiënten zonder provider; in de zachte-capaciteitsvariant vaak 0"},
        {"Output": "Doelwaarde", "Beschrijving": "Surrogaat-objectief voor scenariovergelijking"},
        {"Output": "Runtime", "Beschrijving": "Uitvoeringstijd per run of sweep"},
        {"Output": "Reisdruk", "Beschrijving": "Gemiddelde km/uren en geografische spreiding van routes"},
    ])


def result_download_json(result: dict) -> bytes:
    payload = {
        "summary": result.get("summary", {}),
        "config": result.get("config", {}),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def df_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def scenario_record(name: str, result: dict) -> dict:
    return {
        "scenario_name": name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "summary": result.get("summary", {}),
        "config": result.get("config", {}),
    }


def run_sensitivity(
    base_config: Config,
    patients_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    travel_df: Optional[pd.DataFrame],
    isala_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    alphas = parse_numeric_list(base_config.sensitivity_alpha_values, float)
    horizons = parse_numeric_list(base_config.sensitivity_horizon_values, int)
    rows = []
    total = max(1, len(alphas) * len(horizons))
    counter = 0
    bar = st.progress(0.0, text="Sensitiviteitsanalyse draait")
    for alpha in alphas:
        for horizon in horizons:
            counter += 1
            cfg = Config(**asdict(base_config))
            cfg.alpha = float(alpha)
            cfg.horizon_days = int(horizon)
            cfg.engine = "referentie"
            res = run_reference_allocator(patients_df, providers_df, cfg, travel_df, isala_df, None)
            rows.append({
                "alpha": alpha,
                "horizon_days": horizon,
                "objective_value": res["summary"]["objective_value"],
                "avg_travel_hrs": res["summary"]["avg_travel_hrs"],
                "utilization_std_dev_pct": res["summary"]["utilization_std_dev_pct"],
                "unassigned_count": res["summary"]["unassigned_count"],
                "runtime_sec": res["summary"]["runtime_sec"],
            })
            bar.progress(counter / total, text=f"Sensitiviteitsanalyse {counter}/{total}")
    return pd.DataFrame(rows)


@st.fragment(run_every="2s")
def job_panel() -> None:
    job_state: Optional[JobState] = st.session_state.get("job_state")
    future: Optional[Future] = st.session_state.get("job_future")
    if job_state is None:
        return

    st.subheader("Uitvoering")
    with job_state.lock:
        progress = job_state.progress
        status = job_state.status
        logs = list(job_state.logs)
        done = job_state.done
        error = job_state.error
        result = job_state.result

    box = st.status(status, expanded=True)
    st.progress(progress, text=status)
    if logs:
        box.write("\n".join(logs[-20:]))

    if future is not None and future.done() and not done:
        try:
            future.result()
        except Exception as exc:
            job_state.finalize(error=str(exc))

    if job_state.done:
        if error:
            box.update(label="Mislukt", state="error")
            st.error(error)
        else:
            box.update(label="Klaar", state="complete")
            st.success("Allocatie-run succesvol afgerond.")
            st.session_state.latest_result = result

        if st.button("Uitvoering resetten", key="reset_run_btn"):
            st.session_state.job_state = None
            st.session_state.job_future = None
            st.rerun()


def sidebar_config() -> Tuple[Config, str]:
    st.sidebar.title("Instellingen")

    # Scenario load
    scenario_file = st.sidebar.file_uploader("Scenario laden (.json)", type=["json"])
    if scenario_file is not None:
        try:
            loaded = json.loads(scenario_file.getvalue().decode("utf-8"))
            loaded_cfg = loaded.get("config", loaded)
            st.session_state.config_dict.update(loaded_cfg)
            st.sidebar.success("Scenario geladen. Widgets tonen de nieuwe defaults bij volgende rerun.")
        except Exception as exc:
            st.sidebar.error(f"Scenario kon niet worden geladen: {exc}")

    cfg_defaults = st.session_state.config_dict

    source_mode = st.sidebar.radio("Databron", ["Demo-data", "Upload CSV's"], index=0)
    engine_mode = st.sidebar.radio("Allocatie-engine", ["Referentie-implementatie", "Eigen module"], index=0)

    with st.sidebar.expander("Kernelparameters", expanded=True):
        alpha = st.slider("alpha", 0.0, 1.0, float(cfg_defaults.get("alpha", 0.60)), 0.05)
        strategy = st.selectbox(
            "Toewijzingsstrategie",
            list(STRATEGIES.keys()),
            index=list(STRATEGIES.keys()).index(cfg_defaults.get("strategy", "heaviest_first")),
            format_func=lambda x: STRATEGIES[x],
        )
        lookahead_days = st.number_input("Lookahead in dagen", 1, 365, int(cfg_defaults.get("lookahead_days", 7)))
        horizon_days = st.number_input("Analysehorizon in dagen", 1, 365, int(cfg_defaults.get("horizon_days", 30)))
        timestep_days = st.number_input("Lengte van tijdstap in dagen", 1, 30, int(cfg_defaults.get("timestep_days", 1)))
        avg_speed_kmh = st.number_input("Gemiddelde snelheid km/u", 1.0, 130.0, float(cfg_defaults.get("avg_speed_kmh", 30.0)), 1.0)
        distance_norm_hours = st.number_input("Afstand-normalisatie in uren", 0.1, 24.0, float(cfg_defaults.get("distance_norm_hours", 2.0)), 0.1)
        allow_overcapacity = st.checkbox("Zachte capaciteitsgrens toestaan", value=bool(cfg_defaults.get("allow_overcapacity", True)))
        overcapacity_penalty_weight = st.number_input("Penalty weight voor overcapaciteit", 0.0, 1000.0, float(cfg_defaults.get("overcapacity_penalty_weight", 10.0)), 0.5)
        capacity_buffer_pct = st.number_input("Capaciteitsbuffer (%)", -50.0, 100.0, float(cfg_defaults.get("capacity_buffer_pct", 0.0)), 1.0)

    with st.sidebar.expander("Demo- en kaartparameters", expanded=False):
        random_seed = st.number_input("Random seed", 0, 999999, int(cfg_defaults.get("random_seed", 42)))
        patient_count = st.number_input("Aantal demo-patiënten", 10, 10000, int(cfg_defaults.get("patient_count", 500)))
        provider_count = st.number_input("Aantal demo-homecares", 1, 50, int(cfg_defaults.get("provider_count", 3)))
        base_capacity_hrs_per_week = st.number_input("Capaciteit per week (basis)", 1.0, 1000.0, float(cfg_defaults.get("base_capacity_hrs_per_week", 80.0)), 1.0)
        base_initial_load_hrs_per_week = st.number_input("Beginbelasting per week (basis)", 0.0, 1000.0, float(cfg_defaults.get("base_initial_load_hrs_per_week", 50.0)), 1.0)
        synthetic_radius_km = st.number_input("Straal rond Isala voor demo-patiënten (km)", 1.0, 300.0, float(cfg_defaults.get("synthetic_radius_km", 50.0)), 1.0)
        los_min_days = st.number_input("Min. duur thuiszorgtraject (dagen)", 1, 365, int(cfg_defaults.get("los_min_days", 14)))
        los_max_days = st.number_input("Max. duur thuiszorgtraject (dagen)", 1, 365, int(cfg_defaults.get("los_max_days", 42)))
        visit_hours_min = st.number_input("Min. zorgvraag (uur/week)", 0.0, 168.0, float(cfg_defaults.get("visit_hours_min", 2.0)), 0.5)
        visit_hours_max = st.number_input("Max. zorgvraag (uur/week)", 0.0, 168.0, float(cfg_defaults.get("visit_hours_max", 7.0)), 0.5)
        travel_matrix_mode = st.selectbox("Reismatrix", ["bereken", "upload"], index=["bereken", "upload"].index(cfg_defaults.get("travel_matrix_mode", "bereken")))
        isolation_rule = st.selectbox(
            "Regel voor Isala-/isolatieplaatsing",
            list(ISOLATION_RULES.keys()),
            index=list(ISOLATION_RULES.keys()).index(cfg_defaults.get("isolation_rule", "vast_bij_ziekenhuis")),
            format_func=lambda x: ISOLATION_RULES[x],
        )

    with st.sidebar.expander("Eigen module", expanded=(engine_mode == "Eigen module")):
        module_path = st.text_input("Pad naar Python-bestand", value=str(cfg_defaults.get("module_path", "")))
        module_function_name = st.text_input("Functienaam in jullie module", value=str(cfg_defaults.get("module_function_name", "rolling_horizon_assignment")))
        custom_strategy_code = st.text_area(
            "Custom strategiecode",
            value=str(cfg_defaults.get("custom_strategy_code", "")),
            height=180,
            help="Definieer functies met signatuur: fn(patient_row, candidate_df, context) -> candidate_df",
        )

    with st.sidebar.expander("Sensitiviteitsanalyse", expanded=False):
        sensitivity_alpha_values = st.text_input("Alpha-waarden", value=str(cfg_defaults.get("sensitivity_alpha_values", "0.0,0.2,0.4,0.6,0.8,1.0")))
        sensitivity_horizon_values = st.text_input("Horizon-waarden (dagen)", value=str(cfg_defaults.get("sensitivity_horizon_values", "7,14,21,30")))

    config = Config(
        alpha=float(alpha),
        strategy=strategy,
        lookahead_days=int(lookahead_days),
        horizon_days=int(horizon_days),
        timestep_days=int(timestep_days),
        avg_speed_kmh=float(avg_speed_kmh),
        distance_norm_hours=float(distance_norm_hours),
        overcapacity_penalty_weight=float(overcapacity_penalty_weight),
        allow_overcapacity=bool(allow_overcapacity),
        capacity_buffer_pct=float(capacity_buffer_pct),
        random_seed=int(random_seed),
        patient_count=int(patient_count),
        provider_count=int(provider_count),
        base_capacity_hrs_per_week=float(base_capacity_hrs_per_week),
        base_initial_load_hrs_per_week=float(base_initial_load_hrs_per_week),
        synthetic_radius_km=float(synthetic_radius_km),
        los_min_days=int(los_min_days),
        los_max_days=int(los_max_days),
        visit_hours_min=float(visit_hours_min),
        visit_hours_max=float(visit_hours_max),
        travel_matrix_mode=travel_matrix_mode,
        isolation_rule=isolation_rule,
        engine="eigen_module" if engine_mode == "Eigen module" else "referentie",
        module_path=module_path,
        module_function_name=module_function_name,
        custom_strategy_code=custom_strategy_code,
        sensitivity_alpha_values=sensitivity_alpha_values,
        sensitivity_horizon_values=sensitivity_horizon_values,
    )
    st.session_state.config_dict = asdict(config)
    return config, source_mode


def load_data_ui(config: Config, source_mode: str):
    patients_df = None
    providers_df = None
    travel_df = None
    isala_df = None
    uploaded_module_path = None

    if source_mode == "Upload CSV's":
        with st.expander("Upload bronbestanden", expanded=True):
            c1, c2 = st.columns(2)
            providers_file = c1.file_uploader("Upload providers.csv", type=["csv"])
            patients_file = c2.file_uploader("Upload patients.csv", type=["csv"])
            c3, c4, c5 = st.columns(3)
            travel_file = c3.file_uploader("Optioneel: travel_matrix.csv", type=["csv"])
            isala_file = c4.file_uploader("Optioneel: isala.csv", type=["csv"])
            module_file = c5.file_uploader("Optioneel: eigen code (.py)", type=["py"])

        if providers_file is not None:
            providers_df = normalize_providers(uploaded_csv_to_df(providers_file.getvalue()))
        if patients_file is not None:
            patients_df = normalize_patients(uploaded_csv_to_df(patients_file.getvalue()))
        if travel_file is not None:
            travel_df = normalize_travel(uploaded_csv_to_df(travel_file.getvalue()))
        if isala_file is not None:
            isala_df = normalize_isala(uploaded_csv_to_df(isala_file.getvalue()))
        if module_file is not None:
            tmp = Path(tempfile.mkdtemp(prefix="alloc_module_")) / module_file.name
            tmp.write_bytes(module_file.getvalue())
            uploaded_module_path = str(tmp)
    else:
        providers_df = generate_demo_providers(config)
        patients_df = generate_demo_patients(config)
        isala_df = default_isala_df()

    return patients_df, providers_df, travel_df, isala_df, uploaded_module_path


def metric_cards(summary: dict) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Toegewezen", f"{summary.get('total_assigned', 0)}")
    c2.metric("Onvervulde vraag", f"{summary.get('unassigned_count', 0)}")
    c3.metric("Gem. reistijd", f"{fmt_num(summary.get('avg_travel_hrs', 0.0), 2)} uur")
    c4.metric("Spreiding bezetting", f"{fmt_num(summary.get('utilization_std_dev_pct', 0.0), 2)}%")
    c5.metric("Doelwaarde", fmt_num(summary.get("objective_value", 0.0), 3))
    c6.metric("Runtime", f"{fmt_num(summary.get('runtime_sec', 0.0), 2)} s")


def main() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    config, source_mode = sidebar_config()
    patients_df, providers_df, travel_df, isala_df, uploaded_module_path = load_data_ui(config, source_mode)

    st.markdown("### Kern-tabellen")
    t1, t2, t3 = st.tabs(["Strategieën", "Instelbare parameters", "Verwachte outputs"])
    with t1:
        st.dataframe(strategy_table(), use_container_width=True)
    with t2:
        st.dataframe(parameter_table(), use_container_width=True)
    with t3:
        st.dataframe(output_table(), use_container_width=True)

    if patients_df is None or providers_df is None:
        st.info("Upload `providers.csv` en `patients.csv`, of gebruik demo-data.")
        st.stop()

    st.markdown("### Data-overzicht")
    d1, d2 = st.columns(2)
    with d1:
        st.dataframe(providers_df, use_container_width=True, height=280)
    with d2:
        st.dataframe(patients_df, use_container_width=True, height=280)

    st.markdown("### Procesdiagrammen")
    m1, m2 = st.columns(2)
    with m1:
        render_mermaid(mermaid_flowchart(config), "Flowchart van de allocatie")
    with m2:
        render_mermaid(mermaid_timeline(config), "Timeline van de run")

    st.markdown("### Scenario-acties")
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2.4])
    run_clicked = c1.button("▶️ Run allocatie", type="primary", use_container_width=True)
    sens_clicked = c2.button("🧪 Draai sensitiviteit", use_container_width=True)
    scenario_name = c4.text_input("Naam voor scenario-opslag", value=f"Scenario {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c3.download_button(
        "💾 Download scenario",
        data=json.dumps({"config": asdict(config)}, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="scenario_config.json",
        mime="application/json",
        use_container_width=True,
    )

    if run_clicked:
        module_path = uploaded_module_path or config.module_path or None
        job_state, future = submit_job(asdict(config), patients_df, providers_df, travel_df, isala_df, module_path)
        st.session_state.job_state = job_state
        st.session_state.job_future = future
        st.session_state.latest_result = None
        st.rerun()

    if sens_clicked:
        with st.spinner("Sensitiviteitsanalyse draait..."):
            st.session_state.sensitivity_df = run_sensitivity(config, patients_df, providers_df, travel_df, isala_df)

    job_panel()

    result = st.session_state.get("latest_result")
    sens_df = st.session_state.get("sensitivity_df")

    if result is not None:
        st.markdown("### Overzicht van huidige run")
        metric_cards(result["summary"])

        if st.button("✅ Scenario opslaan in sessie"):
            st.session_state.scenario_history.append(scenario_record(scenario_name, result))
            st.success("Scenario opgeslagen in sessiehistorie.")

        tabs = st.tabs(["Overzicht", "Grafieken", "Kaarten", "Tabellen", "Export", "Scenariohistorie"])

        assignments_df = result["assignments_df"]
        occupancy_df = result["occupancy_df"]
        routes_df = result["routes_df"]
        providers_res_df = result["providers_df"]
        isala_res_df = result["isala_df"]
        isolation_df = result["isolation_df"]
        process_log_df = result.get("process_log_df", pd.DataFrame())
        ts_df = result["time_series_assignments_df"]

        with tabs[0]:
            a1, a2 = st.columns(2)
            with a1:
                st.dataframe(
                    assignments_df[[
                        "patient_id", "discharge_date", "visit_hours", "length_of_stay",
                        "assigned_provider", "assigned_at_timestep", "travel_km", "unmet_demand"
                    ]].sort_values(["assigned_at_timestep", "patient_id"]),
                    use_container_width=True,
                    height=420,
                )
            with a2:
                st.dataframe(
                    occupancy_df[[
                        "provider_id", "week_start", "capacity_hrs_per_week", "used_hrs",
                        "remaining_hrs", "utilization_pct", "active_patients", "new_assignments", "overcapacity"
                    ]],
                    use_container_width=True,
                    height=420,
                )
            with st.expander("Proceslogboek"):
                st.dataframe(process_log_df, use_container_width=True, height=260)

        with tabs[1]:
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(occupancy_chart(occupancy_df), use_container_width=True)
                st.plotly_chart(assignment_ts_chart(ts_df), use_container_width=True)
            with g2:
                st.plotly_chart(active_patients_chart(occupancy_df), use_container_width=True)
                st.plotly_chart(travel_heatmap(assignments_df), use_container_width=True)

        with tabs[2]:
            show_routes = st.toggle("Toon routes tussen homecare en patiënt", value=True)
            st.pydeck_chart(
                map_deck(providers_res_df, assignments_df, routes_df, isala_res_df, isolation_df, show_routes),
                use_container_width=True,
            )
            st.dataframe(
                pd.concat(
                    [
                        providers_res_df.assign(object_type="homecare"),
                        isala_res_df.rename(columns={"location_id": "provider_id"}).assign(object_type="isala"),
                        isolation_df.rename(columns={"location_id": "provider_id"}).assign(object_type="isolatie"),
                    ],
                    ignore_index=True,
                    sort=False,
                ),
                use_container_width=True,
            )

        with tabs[3]:
            provider_summary = occupancy_df.groupby("provider_id").agg(
                avg_utilization_pct=("utilization_pct", "mean"),
                max_utilization_pct=("utilization_pct", "max"),
                overcapacity_weeks=("overcapacity", "sum"),
                avg_active_patients=("active_patients", "mean"),
            ).reset_index()
            st.dataframe(provider_summary, use_container_width=True)
            st.dataframe(assignments_df, use_container_width=True, height=350)

        with tabs[4]:
            e1, e2, e3 = st.columns(3)
            e1.download_button(
                "Assignments CSV",
                data=df_csv_bytes(assignments_df),
                file_name="assignments.csv",
                mime="text/csv",
                use_container_width=True,
            )
            e2.download_button(
                "Occupancy CSV",
                data=df_csv_bytes(occupancy_df),
                file_name="occupancy.csv",
                mime="text/csv",
                use_container_width=True,
            )
            e3.download_button(
                "Run JSON",
                data=result_download_json(result),
                file_name="run_summary.json",
                mime="application/json",
                use_container_width=True,
            )

        with tabs[5]:
            history = st.session_state.get("scenario_history", [])
            if history:
                hist_df = pd.DataFrame([{
                    "scenario_name": h["scenario_name"],
                    "saved_at": h["saved_at"],
                    "strategy": h.get("summary", {}).get("strategy"),
                    "objective_value": h.get("summary", {}).get("objective_value"),
                    "avg_travel_hrs": h.get("summary", {}).get("avg_travel_hrs"),
                    "utilization_std_dev_pct": h.get("summary", {}).get("utilization_std_dev_pct"),
                    "runtime_sec": h.get("summary", {}).get("runtime_sec"),
                } for h in history])
                st.dataframe(hist_df, use_container_width=True)
            else:
                st.info("Nog geen scenario's opgeslagen in deze sessie.")

    if sens_df is not None and not sens_df.empty:
        st.markdown("### Resultaten sensitiviteitsanalyse")
        st.dataframe(sens_df, use_container_width=True)
        piv = sens_df.pivot(index="alpha", columns="horizon_days", values="objective_value")
        fig = go.Figure(data=go.Heatmap(z=piv.values, x=list(piv.columns), y=list(piv.index), colorbar_title="Doelwaarde"))
        fig.update_layout(title="Doelwaarde per alpha en horizon")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

#C:\Users\krisl\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run App.py