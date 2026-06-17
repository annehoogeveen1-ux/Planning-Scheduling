from __future__ import annotations

import ast
import copy
import importlib.util
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class AppPaths:
    base_dir: Path
    rolling_path: Optional[Path]
    patient_generator_path: Optional[Path]
    config_path: Optional[Path]
    patients_csv: Optional[Path]
    providers_csv: Optional[Path]
    nurse_skills_csv: Optional[Path]
    type_care_csv: Optional[Path]
    visit_frequency_csv: Optional[Path]


def find_project_paths(start_dirs: Optional[list[Path]] = None) -> AppPaths:
    candidates = start_dirs or [Path.cwd(), Path("/mnt/data"), Path("/mnt/data/ps_main/Planning-Scheduling-main")]
    found_root = None

    for candidate in candidates:
        if not candidate.exists():
            continue
        for path in [candidate] + list(candidate.rglob("Rolling_Horizon_Allocation.py")):
            p = path if path.is_dir() else path.parent
            if (p / "Rolling_Horizon_Allocation.py").exists():
                found_root = p
                break
        if found_root:
            break

    if found_root is None:
        found_root = Path.cwd()

    def maybe(name: str) -> Optional[Path]:
        p = found_root / name
        return p if p.exists() else None

    return AppPaths(
        base_dir=found_root,
        rolling_path=maybe("Rolling_Horizon_Allocation.py"),
        patient_generator_path=maybe("Patient_Generator.py"),
        config_path=maybe("Config.py"),
        patients_csv=maybe("patients.csv"),
        providers_csv=maybe("providers.csv"),
        nurse_skills_csv=maybe("P_Nurse_Skills.csv"),
        type_care_csv=maybe("P_Type_Care.csv"),
        visit_frequency_csv=maybe("P_Visit_Frequency.csv"),
    )


def _import_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Kan module niet laden: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DefaultVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            value = None
        for target in node.targets:
            if isinstance(target, ast.Name) and value is not None:
                self.values[target.id] = value
        self.generic_visit(node)


def _extract_function_defaults(module_path: Path, function_name: str) -> dict[str, Any]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            args = node.args.args
            defaults = node.args.defaults
            mapped: dict[str, Any] = {}
            offset = len(args) - len(defaults)
            for arg, default in zip(args[offset:], defaults):
                try:
                    mapped[arg.arg] = ast.literal_eval(default)
                except Exception:
                    pass
            return mapped
    return {}


def discover_parameters(paths: AppPaths) -> dict[str, Any]:
    defaults = {
        "alpha": 0.5,
        "lookahead_days": 7,
        "avg_speed_kmh": 30.0,
        "overcapacity_penalty_weight": 10.0,
        "synthetic_n_patients": 500,
        "synthetic_n_providers": 6,
        "synthetic_provider_capacity": 80.0,
        "synthetic_initial_load_pct": 0.55,
        "synthetic_horizon_days": 30,
        "synthetic_length_of_stay_min": 14,
        "synthetic_length_of_stay_max": 42,
        "synthetic_radius_km": 50,
        "synthetic_seed": 42,
        "strategy": "heaviest_load_first",
        "use_existing_heaviest": True,
    }

    if paths.rolling_path and paths.rolling_path.exists():
        visitor = _DefaultVisitor()
        visitor.visit(ast.parse(paths.rolling_path.read_text(encoding="utf-8")))
        defaults["overcapacity_penalty_weight"] = visitor.values.get(
            "OVERCAPACITY_PENALTY_WEIGHT", defaults["overcapacity_penalty_weight"]
        )
        fn_defaults = _extract_function_defaults(paths.rolling_path, "rolling_horizon_assignment")
        defaults["alpha"] = fn_defaults.get("alpha", defaults["alpha"])
        defaults["lookahead_days"] = fn_defaults.get("lookahead_days", defaults["lookahead_days"])
        defaults["avg_speed_kmh"] = fn_defaults.get("avg_speed_kmh", defaults["avg_speed_kmh"])

    if paths.patient_generator_path and paths.patient_generator_path.exists():
        visitor = _DefaultVisitor()
        visitor.visit(ast.parse(paths.patient_generator_path.read_text(encoding="utf-8")))
        defaults["synthetic_horizon_days"] = visitor.values.get(
            "HORIZON_LENGTH_DAYS", defaults["synthetic_horizon_days"]
        )
        defaults["synthetic_length_of_stay_min"] = visitor.values.get(
            "LENGTH_OF_STAY_MIN_DAYS", defaults["synthetic_length_of_stay_min"]
        )
        defaults["synthetic_length_of_stay_max"] = visitor.values.get(
            "LENGTH_OF_STAY_MAX_DAYS", defaults["synthetic_length_of_stay_max"]
        )
        fn_defaults = _extract_function_defaults(paths.patient_generator_path, "generate_dataset")
        defaults["synthetic_n_patients"] = fn_defaults.get("n", defaults["synthetic_n_patients"])

    if paths.providers_csv and paths.providers_csv.exists():
        providers_df = pd.read_csv(paths.providers_csv)
        if "capacity_hrs_per_week" in providers_df.columns and not providers_df.empty:
            defaults["synthetic_provider_capacity"] = float(providers_df["capacity_hrs_per_week"].mean())
        if {
            "capacity_hrs_per_week",
            "initial_load_hrs_per_week",
        }.issubset(providers_df.columns) and not providers_df.empty:
            ratio = providers_df["initial_load_hrs_per_week"] / providers_df["capacity_hrs_per_week"].replace(0, np.nan)
            defaults["synthetic_initial_load_pct"] = float(ratio.fillna(0).mean())
            defaults["synthetic_n_providers"] = int(len(providers_df))

    return defaults


def load_existing_module(paths: AppPaths) -> Optional[Any]:
    if not paths.rolling_path:
        return None
    return _import_module(paths.rolling_path, "rolling_horizon_user_module")


def load_existing_patients(paths: AppPaths, filepath: Optional[Path] = None) -> pd.DataFrame:
    patient_path = filepath or paths.patients_csv
    if patient_path is None or not patient_path.exists():
        raise FileNotFoundError("patients.csv niet gevonden")
    df = pd.read_csv(patient_path, parse_dates=["discharge_date"])
    required = {"patient_id", "discharge_date", "length_of_stay", "visit_hours", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"patients.csv mist kolommen: {sorted(missing)}")
    return df


def load_existing_providers(paths: AppPaths, filepath: Optional[Path] = None) -> pd.DataFrame:
    provider_path = filepath or paths.providers_csv
    if provider_path is None or not provider_path.exists():
        raise FileNotFoundError("providers.csv niet gevonden")
    df = pd.read_csv(provider_path)
    required = {"provider_id", "latitude", "longitude", "capacity_hrs_per_week"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"providers.csv mist kolommen: {sorted(missing)}")
    if "initial_load_hrs_per_week" not in df.columns:
        df["initial_load_hrs_per_week"] = 0.0
    return df


def _load_distribution_csv(path: Optional[Path], sep: str = ";") -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path, sep=sep, decimal=",")


def _sample_with_probs(values: np.ndarray, probs: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    total = probs.sum()
    if total <= 0:
        probs = np.repeat(1 / len(values), len(values))
    else:
        probs = probs / total
    return rng.choice(values, size=size, p=probs)


def _load_location_centre(paths: AppPaths) -> tuple[float, float]:
    centre_lat, centre_lon = 52.5136992436934, 6.123670119684271
    if paths.config_path and paths.config_path.exists():
        try:
            cfg = _import_module(paths.config_path, "rolling_horizon_config_module")
            if hasattr(cfg, "location_centre"):
                centre_lat = float(cfg.location_centre.get("centre_lat", centre_lat))
                centre_lon = float(cfg.location_centre.get("centre_lon", centre_lon))
        except Exception:
            pass
    return centre_lat, centre_lon


def _point_from_center(lat: float, lon: float, distance_km: float, bearing_deg: float) -> tuple[float, float]:
    radius_earth = 6371.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing = math.radians(bearing_deg)
    angular_distance = distance_km / radius_earth

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


def generate_synthetic_providers(paths: AppPaths, n_providers: int, capacity_hrs_per_week: float,
                                 initial_load_pct: float, radius_km: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centre_lat, centre_lon = _load_location_centre(paths)
    records = []
    for i in range(n_providers):
        distance = float(rng.uniform(0, radius_km * 0.9))
        bearing = float(rng.uniform(0, 360))
        lat, lon = _point_from_center(centre_lat, centre_lon, distance, bearing)
        cap = float(max(20.0, rng.normal(capacity_hrs_per_week, capacity_hrs_per_week * 0.08)))
        initial_load = float(max(0.0, min(cap * 0.95, rng.normal(cap * initial_load_pct, cap * 0.08))))
        records.append(
            {
                "provider_id": f"Homecare{chr(65 + i)}",
                "latitude": lat,
                "longitude": lon,
                "capacity_hrs_per_week": round(cap, 1),
                "initial_load_hrs_per_week": round(initial_load, 1),
            }
        )
    return pd.DataFrame(records)


def generate_synthetic_patients(paths: AppPaths, n_patients: int, horizon_days: int,
                                length_of_stay_min: int, length_of_stay_max: int,
                                radius_km: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    centre_lat, centre_lon = _load_location_centre(paths)

    skill_df = _load_distribution_csv(paths.nurse_skills_csv)
    care_df = _load_distribution_csv(paths.type_care_csv)
    visit_df = _load_distribution_csv(paths.visit_frequency_csv)

    if skill_df is not None and {"Niveau", "Dist"}.issubset(skill_df.columns):
        nurse_skills = _sample_with_probs(skill_df["Niveau"].to_numpy(), skill_df["Dist"].to_numpy(), n_patients, rng)
    else:
        nurse_skills = rng.integers(1, 8, size=n_patients)

    if care_df is not None and {"Diagnose", "Distribution"}.issubset(care_df.columns):
        type_care = _sample_with_probs(care_df["Diagnose"].to_numpy(dtype=object), care_df["Distribution"].to_numpy(), n_patients, rng)
    else:
        fallback_care = np.array(["cardiologie", "chirurgie", "orthopedie", "neurologie"], dtype=object)
        type_care = rng.choice(fallback_care, size=n_patients)

    if visit_df is not None and {"uur/week", "CDF Bucket"}.issubset(visit_df.columns):
        visit_hours = _sample_with_probs(visit_df["uur/week"].to_numpy(), visit_df["CDF Bucket"].to_numpy(), n_patients, rng)
        visit_hours = np.maximum(1, visit_hours.astype(int))
    else:
        visit_hours = rng.integers(2, 7, size=n_patients)

    start_date = date.today()
    records = []
    for i in range(n_patients):
        distance = float(rng.uniform(0, radius_km))
        bearing = float(rng.uniform(0, 360))
        lat, lon = _point_from_center(centre_lat, centre_lon, distance, bearing)
        discharge_offset = int(rng.integers(0, max(1, horizon_days)))
        discharge_date = start_date + timedelta(days=discharge_offset)
        length_of_stay = int(rng.integers(length_of_stay_min, length_of_stay_max + 1))
        records.append(
            {
                "patient_id": f"P{i + 1:04d}",
                "discharge_date": pd.Timestamp(discharge_date),
                "length_of_stay": length_of_stay,
                "visit_hours": int(visit_hours[i]),
                "latitude": lat,
                "longitude": lon,
                "nurse_skill": int(nurse_skills[i]),
                "type_care": str(type_care[i]),
            }
        )

    return pd.DataFrame(records).sort_values(["discharge_date", "patient_id"]).reset_index(drop=True)


def haversine_km(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    radius_earth = 6371.0
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return radius_earth * c


def travel_hours(coord1: tuple[float, float], coord2: tuple[float, float], avg_speed_kmh: float) -> float:
    distance = haversine_km(coord1, coord2)
    return (distance / avg_speed_kmh) * 2.0


def monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def generate_weeks(start: date, end: date) -> list[date]:
    current = monday_of(start)
    weeks = []
    while current <= end:
        weeks.append(current)
        current += timedelta(days=7)
    return weeks


def active_weeks_full(discharge_date: date, length_of_stay: int, all_weeks: list[date]) -> list[date]:
    care_end = discharge_date + timedelta(days=int(length_of_stay))
    return [w for w in all_weeks if discharge_date <= w < care_end]


def _row_to_patient_obj(row: pd.Series, patient_cls: Any) -> Any:
    return patient_cls(
        patient_id=str(row["patient_id"]),
        discharge_date=pd.Timestamp(row["discharge_date"]).date(),
        length_of_stay=int(row["length_of_stay"]),
        visit_hours=float(row["visit_hours"]),
        home_coords=(float(row["latitude"]), float(row["longitude"])),
    )


def _row_to_provider_obj(row: pd.Series, provider_cls: Any) -> Any:
    return provider_cls(
        provider_id=str(row["provider_id"]),
        coords=(float(row["latitude"]), float(row["longitude"])),
        capacity_hrs_per_week=float(row["capacity_hrs_per_week"]),
        initial_load_hrs_per_week=float(row.get("initial_load_hrs_per_week", 0.0)),
    )


def _fallback_patient_cls():
    @dataclass
    class Patient:
        patient_id: str
        discharge_date: date
        length_of_stay: int
        visit_hours: float
        home_coords: tuple[float, float]
        assigned_provider: Optional[str] = None

        @property
        def care_end(self) -> date:
            return self.discharge_date + timedelta(days=self.length_of_stay)

    return Patient


def _fallback_provider_cls():
    @dataclass
    class Provider:
        provider_id: str
        coords: tuple[float, float]
        capacity_hrs_per_week: float
        initial_load_hrs_per_week: float = 0.0

    return Provider


def _prepare_runtime_objects(existing_module: Optional[Any], patients_df: pd.DataFrame, providers_df: pd.DataFrame) -> tuple[list[Any], list[Any]]:
    patient_cls = getattr(existing_module, "Patient", _fallback_patient_cls())
    provider_cls = getattr(existing_module, "Provider", _fallback_provider_cls())

    patients = [_row_to_patient_obj(row, patient_cls) for _, row in patients_df.iterrows()]
    providers = [_row_to_provider_obj(row, provider_cls) for _, row in providers_df.iterrows()]
    return patients, providers


def _calculate_load(provider_id: str, active_weeks: list[date], remaining_capacity: dict[str, dict[date, float]],
                    provider_capacity: dict[str, float]) -> float:
    if not active_weeks:
        return 0.0
    return max(
        1 - (remaining_capacity[provider_id][w] / provider_capacity[provider_id])
        for w in active_weeks
        if w in remaining_capacity[provider_id]
    )


def _calculate_penalty(provider_id: str, active_weeks: list[date], patient_visit_hours: float, travel_hrs: float,
                       remaining_capacity: dict[str, dict[date, float]], penalty_weight: float) -> float:
    penalty = 0.0
    need = patient_visit_hours + travel_hrs
    for w in active_weeks:
        deficit = need - remaining_capacity[provider_id].get(w, 0.0)
        if deficit > 0:
            penalty = max(penalty, penalty_weight * deficit)
    return penalty


def _choose_provider_heaviest(provider_ids: list[str], provider_capacity: dict[str, float],
                              active_weeks: list[date], patient_visit_hours: float,
                              travel_hours_map: dict[str, float], remaining_capacity: dict[str, dict[date, float]],
                              alpha: float, penalty_weight: float) -> tuple[str, float]:
    best_provider = provider_ids[0]
    best_score = float("inf")
    for provider_id in provider_ids:
        load = _calculate_load(provider_id, active_weeks, remaining_capacity, provider_capacity)
        distance_raw = travel_hours_map[provider_id]
        distance_normalized = min(distance_raw / 2.0, 1.0)
        penalty = _calculate_penalty(provider_id, active_weeks, patient_visit_hours, distance_raw, remaining_capacity, penalty_weight)
        score = alpha * load + (1 - alpha) * distance_normalized + penalty
        if score < best_score:
            best_score = score
            best_provider = provider_id
    return best_provider, best_score


def _choose_provider_distance(provider_ids: list[str], provider_capacity: dict[str, float],
                              active_weeks: list[date], patient_visit_hours: float,
                              travel_hours_map: dict[str, float], remaining_capacity: dict[str, dict[date, float]],
                              alpha: float, penalty_weight: float) -> tuple[str, float]:
    ranked = []
    for provider_id in provider_ids:
        distance_raw = travel_hours_map[provider_id]
        penalty = _calculate_penalty(provider_id, active_weeks, patient_visit_hours, distance_raw, remaining_capacity, penalty_weight)
        load = _calculate_load(provider_id, active_weeks, remaining_capacity, provider_capacity)
        score = penalty + distance_raw + alpha * load
        ranked.append((score, provider_id))
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1], ranked[0][0]


def _choose_provider_balanced(provider_ids: list[str], provider_capacity: dict[str, float],
                              active_weeks: list[date], patient_visit_hours: float,
                              travel_hours_map: dict[str, float], remaining_capacity: dict[str, dict[date, float]],
                              alpha: float, penalty_weight: float) -> tuple[str, float]:
    ranked = []
    for provider_id in provider_ids:
        load = _calculate_load(provider_id, active_weeks, remaining_capacity, provider_capacity)
        distance_raw = travel_hours_map[provider_id]
        distance_normalized = min(distance_raw / 2.0, 1.0)
        penalty = _calculate_penalty(provider_id, active_weeks, patient_visit_hours, distance_raw, remaining_capacity, penalty_weight)
        score = penalty + load + (1 - alpha) * distance_normalized
        ranked.append((score, provider_id))
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1], ranked[0][0]


def _choose_provider_round_robin(provider_ids: list[str], provider_capacity: dict[str, float],
                                 active_weeks: list[date], patient_visit_hours: float,
                                 travel_hours_map: dict[str, float], remaining_capacity: dict[str, dict[date, float]],
                                 penalty_weight: float, rr_index: int) -> tuple[str, float, int]:
    rotated = provider_ids[rr_index:] + provider_ids[:rr_index]
    ranked = []
    for order, provider_id in enumerate(rotated):
        distance_raw = travel_hours_map[provider_id]
        penalty = _calculate_penalty(provider_id, active_weeks, patient_visit_hours, distance_raw, remaining_capacity, penalty_weight)
        ranked.append((penalty, order, distance_raw, provider_id))
    ranked.sort(key=lambda x: (x[0], x[1], x[2]))
    chosen = ranked[0][3]
    next_rr = (provider_ids.index(chosen) + 1) % len(provider_ids)
    return chosen, ranked[0][0], next_rr


def run_allocation_extended(
    patients_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    alpha: float,
    lookahead_days: int,
    avg_speed_kmh: float,
    strategy: str,
    penalty_weight: float,
) -> dict[str, Any]:
    patients_df = patients_df.copy()
    providers_df = providers_df.copy()
    patients_df["discharge_date"] = pd.to_datetime(patients_df["discharge_date"])

    provider_ids = providers_df["provider_id"].astype(str).tolist()
    provider_capacity = providers_df.set_index("provider_id")["capacity_hrs_per_week"].astype(float).to_dict()
    provider_initial = (
        providers_df.set_index("provider_id")["initial_load_hrs_per_week"].astype(float).to_dict()
        if "initial_load_hrs_per_week" in providers_df.columns
        else {pid: 0.0 for pid in provider_ids}
    )
    provider_coords = providers_df.set_index("provider_id")[["latitude", "longitude"]].to_dict("index")

    all_discharge_dates = sorted({pd.Timestamp(d).date() for d in patients_df["discharge_date"]})
    global_start = pd.Timestamp(patients_df["discharge_date"].min()).date()
    global_end = pd.Timestamp(patients_df["discharge_date"].max()).date()
    planning_weeks = generate_weeks(global_start, global_end)

    remaining_capacity = {
        pid: {w: provider_capacity[pid] - provider_initial.get(pid, 0.0) for w in planning_weeks}
        for pid in provider_ids
    }

    assignments: dict[str, str] = {}
    travel_log_hrs: dict[str, float] = {}
    travel_log_km: dict[str, float] = {}
    score_log: dict[str, float] = {}
    processed: set[str] = set()
    rr_index = 0

    patient_provider_travel_hrs: dict[str, dict[str, float]] = {}
    patient_provider_travel_km: dict[str, dict[str, float]] = {}
    for _, row in patients_df.iterrows():
        pid = str(row["patient_id"])
        patient_coord = (float(row["latitude"]), float(row["longitude"]))
        patient_provider_travel_hrs[pid] = {}
        patient_provider_travel_km[pid] = {}
        for provider_id in provider_ids:
            coord = (float(provider_coords[provider_id]["latitude"]), float(provider_coords[provider_id]["longitude"]))
            km = haversine_km(patient_coord, coord)
            patient_provider_travel_km[pid][provider_id] = km
            patient_provider_travel_hrs[pid][provider_id] = (km / avg_speed_kmh) * 2.0

    patient_lookup = {
        str(row["patient_id"]): {
            "patient_id": str(row["patient_id"]),
            "discharge_date": pd.Timestamp(row["discharge_date"]).date(),
            "length_of_stay": int(row["length_of_stay"]),
            "visit_hours": float(row["visit_hours"]),
        }
        for _, row in patients_df.iterrows()
    }

    for t in all_discharge_dates:
        window_end = t + timedelta(days=int(lookahead_days))
        known_patients = [
            patient_lookup[pid]
            for pid, p in patient_lookup.items()
            if t <= p["discharge_date"] <= window_end and pid not in processed
        ]
        if not known_patients:
            continue

        local_horizon_start = min(p["discharge_date"] for p in known_patients)
        local_horizon_end = max(p["discharge_date"] for p in known_patients)
        local_weeks = generate_weeks(local_horizon_start, local_horizon_end)

        active_weeks_map = {
            p["patient_id"]: [
                w
                for w in local_weeks
                if p["discharge_date"] <= w < min(p["discharge_date"] + timedelta(days=p["length_of_stay"]), local_horizon_end)
            ]
            for p in known_patients
        }

        if strategy == "heaviest_load_first":
            ordered = sorted(known_patients, key=lambda x: (-x["visit_hours"], x["discharge_date"], x["patient_id"]))
        elif strategy == "distance_based":
            ordered = sorted(
                known_patients,
                key=lambda x: (
                    min(patient_provider_travel_km[x["patient_id"]].values()),
                    x["discharge_date"],
                    x["patient_id"],
                ),
            )
        elif strategy == "round_robin":
            ordered = sorted(known_patients, key=lambda x: (x["discharge_date"], x["patient_id"]))
        else:
            ordered = sorted(known_patients, key=lambda x: (x["discharge_date"], -x["visit_hours"], x["patient_id"]))

        for patient in ordered:
            pid = patient["patient_id"]
            active_weeks = active_weeks_map[pid]
            travel_map = patient_provider_travel_hrs[pid]

            if strategy == "heaviest_load_first":
                best_provider, best_score = _choose_provider_heaviest(
                    provider_ids, provider_capacity, active_weeks, patient["visit_hours"], travel_map,
                    remaining_capacity, alpha, penalty_weight
                )
            elif strategy == "distance_based":
                best_provider, best_score = _choose_provider_distance(
                    provider_ids, provider_capacity, active_weeks, patient["visit_hours"], travel_map,
                    remaining_capacity, alpha, penalty_weight
                )
            elif strategy == "round_robin":
                best_provider, best_score, rr_index = _choose_provider_round_robin(
                    provider_ids, provider_capacity, active_weeks, patient["visit_hours"], travel_map,
                    remaining_capacity, penalty_weight, rr_index
                )
            else:
                best_provider, best_score = _choose_provider_balanced(
                    provider_ids, provider_capacity, active_weeks, patient["visit_hours"], travel_map,
                    remaining_capacity, alpha, penalty_weight
                )

            for week in active_weeks:
                remaining_capacity[best_provider][week] -= patient["visit_hours"] + travel_map[best_provider]

            assignments[pid] = best_provider
            travel_log_hrs[pid] = travel_map[best_provider]
            travel_log_km[pid] = patient_provider_travel_km[pid][best_provider]
            score_log[pid] = float(best_score)
            processed.add(pid)

    return build_dashboard_outputs(
        patients_df=patients_df,
        providers_df=providers_df,
        assignments=assignments,
        travel_log_km=travel_log_km,
        travel_log_hrs=travel_log_hrs,
        score_log=score_log,
        alpha=alpha,
        avg_speed_kmh=avg_speed_kmh,
        strategy=strategy,
    )


def build_dashboard_outputs(
    patients_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    assignments: dict[str, str],
    travel_log_km: dict[str, float],
    travel_log_hrs: dict[str, float],
    score_log: dict[str, float],
    alpha: float,
    avg_speed_kmh: float,
    strategy: str,
) -> dict[str, Any]:
    patients = patients_df.copy()
    providers = providers_df.copy()
    patients["discharge_date"] = pd.to_datetime(patients["discharge_date"])
    patients["provider_id"] = patients["patient_id"].astype(str).map(assignments)
    patients["travel_km"] = patients["patient_id"].astype(str).map(travel_log_km).fillna(0.0)
    patients["travel_hrs"] = patients["patient_id"].astype(str).map(travel_log_hrs).fillna(0.0)
    patients["score"] = patients["patient_id"].astype(str).map(score_log).fillna(0.0)

    providers["provider_id"] = providers["provider_id"].astype(str)
    provider_capacity = providers.set_index("provider_id")["capacity_hrs_per_week"].astype(float).to_dict()
    provider_initial = providers.set_index("provider_id")["initial_load_hrs_per_week"].astype(float).to_dict()

    start = patients["discharge_date"].min().date()
    end = (patients["discharge_date"] + pd.to_timedelta(patients["length_of_stay"], unit="D")).max().date()
    all_weeks = generate_weeks(start, end)

    weekly_rows: list[dict[str, Any]] = []
    for provider_id in providers["provider_id"]:
        provider_patients = patients[patients["provider_id"] == provider_id].copy()
        for week in all_weeks:
            used_hours = float(provider_initial.get(provider_id, 0.0))
            active_patients = 0
            new_assignments = 0
            for _, p in provider_patients.iterrows():
                discharge = pd.Timestamp(p["discharge_date"]).date()
                care_end = discharge + timedelta(days=int(p["length_of_stay"]))
                if discharge <= week < care_end:
                    used_hours += float(p["visit_hours"]) + float(p["travel_hrs"])
                    active_patients += 1
                if monday_of(discharge) == week:
                    new_assignments += 1
            cap = float(provider_capacity[provider_id])
            weekly_rows.append(
                {
                    "week": pd.Timestamp(week),
                    "provider_id": provider_id,
                    "used_hours": round(used_hours, 2),
                    "capacity_hours": round(cap, 2),
                    "utilization_pct": round((used_hours / cap) * 100 if cap else 0.0, 2),
                    "active_patients": int(active_patients),
                    "new_assignments": int(new_assignments),
                }
            )

    weekly_df = pd.DataFrame(weekly_rows).sort_values(["week", "provider_id"])

    summary = (
        patients.groupby("provider_id", dropna=False)
        .agg(
            toewijzingen=("patient_id", "count"),
            gem_reis_km=("travel_km", "mean"),
            gem_reis_uur=("travel_hrs", "mean"),
            gem_vraag_uur=("visit_hours", "mean"),
        )
        .reset_index()
    )

    all_provider_ids = providers[["provider_id"]].copy()
    summary = all_provider_ids.merge(summary, on="provider_id", how="left").fillna(0)
    util_summary = (
        weekly_df.groupby("provider_id")
        .agg(
            gem_bezetting_pct=("utilization_pct", "mean"),
            max_bezetting_pct=("utilization_pct", "max"),
            overschrijdingsweken=("utilization_pct", lambda s: int((s > 100).sum())),
            piek_actieve_patienten=("active_patients", "max"),
        )
        .reset_index()
    )
    summary = summary.merge(util_summary, on="provider_id", how="left")
    summary["gem_reis_km"] = summary["gem_reis_km"].round(1)
    summary["gem_reis_uur"] = summary["gem_reis_uur"].round(2)
    summary["gem_vraag_uur"] = summary["gem_vraag_uur"].round(2)
    summary["gem_bezetting_pct"] = summary["gem_bezetting_pct"].round(1)
    summary["max_bezetting_pct"] = summary["max_bezetting_pct"].round(1)

    weekly_total_util = weekly_df.groupby("week")["utilization_pct"].mean().reset_index()
    weekly_total_active = weekly_df.groupby("week")["active_patients"].sum().reset_index()

    kpis = {
        "totaal_toegewezen": int(len(assignments)),
        "gem_bezetting_pct": round(float(weekly_df["utilization_pct"].mean()) if not weekly_df.empty else 0.0, 1),
        "gem_reis_km": round(float(patients.loc[patients["provider_id"].notna(), "travel_km"].mean()) if not patients.empty else 0.0, 1),
        "gem_reis_uur": round(float(patients.loc[patients["provider_id"].notna(), "travel_hrs"].mean()) if not patients.empty else 0.0, 2),
        "overschrijdingsweken": int((weekly_df["utilization_pct"] > 100).sum()) if not weekly_df.empty else 0,
        "organisaties_gebruikt": int((summary["toewijzingen"] > 0).sum()) if not summary.empty else 0,
        "doelwaarde": round(float(sum(score_log.values())), 2),
    }

    discovered_parameters = [
        {"sleutel": "alpha", "categorie": "model", "standaard": alpha},
        {"sleutel": "lookahead_days", "categorie": "model", "standaard": None},
        {"sleutel": "avg_speed_kmh", "categorie": "model", "standaard": avg_speed_kmh},
        {"sleutel": "overcapacity_penalty_weight", "categorie": "model", "standaard": None},
        {"sleutel": "capacity_hrs_per_week", "categorie": "provider-data", "standaard": None},
        {"sleutel": "initial_load_hrs_per_week", "categorie": "provider-data", "standaard": None},
        {"sleutel": "HORIZON_LENGTH_DAYS", "categorie": "synthetische data", "standaard": None},
        {"sleutel": "LENGTH_OF_STAY_MIN_DAYS", "categorie": "synthetische data", "standaard": None},
        {"sleutel": "LENGTH_OF_STAY_MAX_DAYS", "categorie": "synthetische data", "standaard": None},
        {"sleutel": "n_patients", "categorie": "synthetische data", "standaard": None},
        {"sleutel": "radius_km", "categorie": "synthetische data", "standaard": None},
        {"sleutel": "strategy", "categorie": "heuristiek", "standaard": strategy},
    ]

    return {
        "patients_df": patients,
        "providers_df": providers,
        "weekly_df": weekly_df,
        "summary_df": summary,
        "kpis": kpis,
        "weekly_total_util": weekly_total_util,
        "weekly_total_active": weekly_total_active,
        "parameters_found": pd.DataFrame(discovered_parameters),
    }


def run_existing_or_extended(
    paths: AppPaths,
    existing_module: Optional[Any],
    patients_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    alpha: float,
    lookahead_days: int,
    avg_speed_kmh: float,
    strategy: str,
    penalty_weight: float,
    use_existing_heaviest: bool,
) -> dict[str, Any]:
    if existing_module is not None and strategy == "heaviest_load_first" and use_existing_heaviest:
        try:
            patients_runtime, providers_runtime = _prepare_runtime_objects(existing_module, patients_df, providers_df)
            if hasattr(existing_module, "OVERCAPACITY_PENALTY_WEIGHT"):
                setattr(existing_module, "OVERCAPACITY_PENALTY_WEIGHT", penalty_weight)
            result = existing_module.rolling_horizon_assignment(
                patients=copy.deepcopy(patients_runtime),
                providers=copy.deepcopy(providers_runtime),
                alpha=float(alpha),
                lookahead_days=int(lookahead_days),
                avg_speed_kmh=float(avg_speed_kmh),
            )
            assignments = result.get("assignments", {})
            patients_mapped = patients_df.copy()
            providers_mapped = providers_df.copy()

            provider_coords = providers_mapped.set_index("provider_id")[["latitude", "longitude"]].to_dict("index")
            travel_km = {}
            travel_hrs_map = {}
            for _, row in patients_mapped.iterrows():
                pid = str(row["patient_id"])
                assigned_provider = assignments.get(pid)
                if assigned_provider is None:
                    continue
                km = haversine_km(
                    (float(row["latitude"]), float(row["longitude"])),
                    (
                        float(provider_coords[assigned_provider]["latitude"]),
                        float(provider_coords[assigned_provider]["longitude"]),
                    ),
                )
                travel_km[pid] = km
                travel_hrs_map[pid] = (km / avg_speed_kmh) * 2.0
            score_log = {pid: 0.0 for pid in assignments}
            return build_dashboard_outputs(
                patients_df=patients_df,
                providers_df=providers_df,
                assignments=assignments,
                travel_log_km=travel_km,
                travel_log_hrs=travel_hrs_map,
                score_log=score_log,
                alpha=alpha,
                avg_speed_kmh=avg_speed_kmh,
                strategy=strategy,
            )
        except Exception:
            pass

    return run_allocation_extended(
        patients_df=patients_df,
        providers_df=providers_df,
        alpha=alpha,
        lookahead_days=lookahead_days,
        avg_speed_kmh=avg_speed_kmh,
        strategy=strategy,
        penalty_weight=penalty_weight,
    )
