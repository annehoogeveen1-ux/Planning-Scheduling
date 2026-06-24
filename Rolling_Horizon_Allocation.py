import math
import pandas as pd
from datetime import date, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional




# =============================================================================
# CONSTANTS
# =============================================================================

# Weight for the overcapacity penalty in the score function.
# Chosen high enough so that an assignment without violation
# ALWAYS takes precedence over an assignment with violation,
# regardless of load/distance differences (which lie in the range [0,1]).
OVERCAPACITY_PENALTY_WEIGHT = 10.0
TRAVEL_SPEED_KMH = 30.0
MAX_TRAVEL_HOURS = 1.0


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Patient:
    patient_id: str
    discharge_date: date
    length_of_stay: int          # in days
    visit_hours: float           # care hours per week
    home_coords: tuple           # (latitude, longitude)
    assigned_provider: Optional[str] = None

    @property
    def care_end(self) -> date:
        return self.discharge_date + timedelta(days=self.length_of_stay)


@dataclass
class Provider:
    provider_id: str
    coords: tuple                # (latitude, longitude)
    capacity_hrs_per_week: float
    initial_load_hrs_per_week: float = 0.0  # current caseload in hours


# =============================================================================
# CSV LOADING
# =============================================================================

def load_providers_from_csv(filepath: str) -> list:
    """
    Loads home care organizations from a CSV file.

    Expected format:
        provider_id,latitude,longitude,capacity_hrs_per_week,initial_load_hrs_per_week
        ThuiszorgA,52.21,6.89,80.0,55.0
    """
    df = pd.read_csv(filepath, dtype={'provider_id': str})

    required_cols = {'provider_id', 'latitude', 'longitude', 'capacity_hrs_per_week'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing mandatory columns: {missing}")

    has_initial_load = 'initial_load_hrs_per_week' in df.columns

    providers = []
    for _, row in df.iterrows():
        providers.append(Provider(
            provider_id=row['provider_id'],
            coords=(float(row['latitude']), float(row['longitude'])),
            capacity_hrs_per_week=float(row['capacity_hrs_per_week']),
            initial_load_hrs_per_week=float(row['initial_load_hrs_per_week']) if has_initial_load else 0.0
        ))

    return providers


def load_patients_from_csv(filepath: str) -> list:
    """
    Loads patients from a CSV file.

    Expected format:
        patient_id,discharge_date,length_of_stay,visit_hours,latitude,longitude
        P0001,2024-01-03,28,4,52.30,5.76
    """
    df = pd.read_csv(filepath, dtype={'patient_id': str}, parse_dates=['discharge_date'])

    required_cols = {'patient_id', 'discharge_date', 'length_of_stay', 'visit_hours', 'latitude', 'longitude'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing mandatory columns: {missing}")

    patients = []
    for _, row in df.iterrows():
        patients.append(Patient(
            patient_id=row['patient_id'],
            discharge_date=row['discharge_date'].date(),
            length_of_stay=int(row['length_of_stay']),
            visit_hours=float(row['visit_hours']),
            home_coords=(float(row['latitude']), float(row['longitude']))
        ))

    return patients

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def haversine_km(coord1: tuple, coord2: tuple) -> float:
    """ Calculates the great-circle distance in kilometers. """
    R = 6371 
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def generate_weeks(start: date, end: date) -> list:
    current = start - timedelta(days=start.weekday())
    weeks = []
    while current <= end:
        weeks.append(current)
        current += timedelta(weeks=1)
    return weeks


def active_weeks_for_patient(patient: Patient, horizon_end: date, all_weeks: list) -> list:
    effective_end = min(patient.care_end, horizon_end)
    return [
        w for w in all_weeks
        if w < effective_end and (w + timedelta(weeks=1)) > patient.discharge_date
    ]

# =============================================================================
# TRAVEL TIME CALCULATION
# =============================================================================

def travel_hours(distance_km: float, speed_kmh: float = TRAVEL_SPEED_KMH) -> float:
    DETOUR_FACTOR = 1.3  
    road_distance_km = distance_km * DETOUR_FACTOR
    return road_distance_km / speed_kmh


# =============================================================================
# ASSIGNMENT METHODS
# =============================================================================

def _book_capacity(p, best_provider, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed):
    for w in patient_active_weeks[p.patient_id]:
        if w in remaining_capacity[best_provider.provider_id]:
            remaining_capacity[best_provider.provider_id][w] -= p.visit_hours

    assignment_map[p.patient_id] = best_provider.provider_id
    distance_log[p.patient_id] = travel_hours(distance_km[p.patient_id][best_provider.provider_id])
    processed.add(p.patient_id)


def _overcapacity_penalty(p, o, patient_active_weeks, remaining_capacity):
    penalty = 0.0
    for w in patient_active_weeks[p.patient_id]:
        if w in remaining_capacity[o.provider_id]:
            needed = p.visit_hours
            deficit = needed - remaining_capacity[o.provider_id][w]
            if deficit > 0:
                penalty = max(penalty, deficit)
    return penalty


def method_greedy_heaviest_first(patients, providers, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed, alpha, **kwargs):
    patients_sorted = sorted(patients, key=lambda p: p.visit_hours, reverse=True)

    for p in patients_sorted:
        if p.patient_id in processed:
            continue

        best_provider = None
        best_score = float('inf')
        active_w = patient_active_weeks[p.patient_id]

        for o in providers:
            load = max(
                (1 - remaining_capacity[o.provider_id][w] / o.capacity_hrs_per_week)
                for w in active_w if w in remaining_capacity[o.provider_id]
            ) if active_w else 0.0

            t_hours = travel_hours(distance_km[p.patient_id][o.provider_id])
            distance_normalized = min(t_hours / MAX_TRAVEL_HOURS, 1.0)
            penalty = _overcapacity_penalty(p, o, patient_active_weeks, remaining_capacity)
            
            score = (alpha * load) + ((1 - alpha) * distance_normalized) + (OVERCAPACITY_PENALTY_WEIGHT * penalty)

            if score < best_score:
                best_score = score
                best_provider = o

        _book_capacity(p, best_provider, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed)


def method_nearest_provider(patients, providers, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed, **kwargs):
    patients_sorted = sorted(patients, key=lambda p: p.discharge_date)

    for p in patients_sorted:
        if p.patient_id in processed:
            continue

        best_provider = None
        best_score = float('inf')

        for o in providers:
            distance = distance_km[p.patient_id][o.provider_id]
            penalty = _overcapacity_penalty(p, o, patient_active_weeks, remaining_capacity)
            score = distance + penalty

            if score < best_score:
                best_score = score
                best_provider = o

        _book_capacity(p, best_provider, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed)


def method_round_robin(patients, providers, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed, round_robin_index, **kwargs):
    patients_sorted = sorted(patients, key=lambda p: p.discharge_date)

    for p in patients_sorted:
        if p.patient_id in processed:
            continue

        chosen = providers[round_robin_index[0] % len(providers)]
        round_robin_index[0] += 1

        _book_capacity(p, chosen, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed)


def method_edd(patients, providers, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed, alpha=0.5, **kwargs):
    patients_sorted = sorted(patients, key=lambda p: p.discharge_date)

    for p in patients_sorted:
        if p.patient_id in processed:
            continue

        best_provider = None
        best_score = float('inf')

        for o in providers:
            active_w = patient_active_weeks[p.patient_id]
            load = max(
                (1 - remaining_capacity[o.provider_id][w] / o.capacity_hrs_per_week)
                for w in active_w if w in remaining_capacity[o.provider_id]
            ) if active_w else 0.0

            t_hours = travel_hours(distance_km[p.patient_id][o.provider_id])
            distance_normalized = min(t_hours / MAX_TRAVEL_HOURS, 1.0)
            penalty = _overcapacity_penalty(p, o, patient_active_weeks, remaining_capacity)

            score = (alpha * load) + ((1 - alpha) * distance_normalized) + (OVERCAPACITY_PENALTY_WEIGHT * penalty)

            if score < best_score:
                best_score = score
                best_provider = o

        _book_capacity(p, best_provider, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed)


def assign_patients(method, patients, providers, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed, alpha, round_robin_index):
    method_lower = method.lower()
    if method_lower == 'greedy':
        method_greedy_heaviest_first(patients, providers, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed, alpha)
    elif method_lower == 'nearest':
        method_nearest_provider(patients, providers, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed)
    elif method_lower == 'round_robin':
        method_round_robin(patients, providers, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed, round_robin_index)
    elif method_lower == 'edd':
        method_edd(patients, providers, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed, alpha)
    else:
        raise ValueError(f"Unknown method '{method}'.")
    

# =============================================================================
# ROLLING HORIZON ALGORITHM
# =============================================================================

def rolling_horizon_assignment(patients: list, providers: list, alpha: float = 0.5, lookahead_days: int = 7, method: str = 'greedy') -> dict:
    all_discharge_dates = sorted(set(p.discharge_date for p in patients))
    global_horizon_start = min(p.discharge_date for p in patients)
    global_horizon_end = max(p.discharge_date for p in patients)
    all_weeks = generate_weeks(global_horizon_start, global_horizon_end)

    remaining_capacity = {
        o.provider_id: {w: o.capacity_hrs_per_week - o.initial_load_hrs_per_week for w in all_weeks}
        for o in providers
    }

    assignment_map = {}
    distance_log = {}
    round_robin_index = [0]
    processed = set()

    for t in all_discharge_dates:
        window_end = t + timedelta(days=lookahead_days)
        known_patients = [p for p in patients if t <= p.discharge_date <= window_end and p.patient_id not in processed]

        if not known_patients:
            continue

        horizon_end = max(p.discharge_date for p in known_patients)
        patient_active_weeks = {p.patient_id: active_weeks_for_patient(p, horizon_end, all_weeks) for p in known_patients}

        distance_km = {}
        for p in known_patients:
            distance_km[p.patient_id] = {o.provider_id: haversine_km(p.home_coords, o.coords) for o in providers}

        assign_patients(
            method=method, patients=known_patients, providers=providers,
            patient_active_weeks=patient_active_weeks, distance_km=distance_km,
            remaining_capacity=remaining_capacity, assignment_map=assignment_map,
            distance_log=distance_log, processed=processed, alpha=alpha,
            round_robin_index=round_robin_index
        )

    kpis = compute_kpis(assignment_map, remaining_capacity, providers, all_weeks, distance_log)
    return {'assignments': assignment_map, 'remaining_capacity': remaining_capacity, 'kpis': kpis}


# =============================================================================
# KPI CALCULATION & DISPLAY
# =============================================================================

def compute_kpis(assignment_map, remaining_capacity, providers, all_weeks, distance_log) -> dict:
    utilization = {}
    for o in providers:
        utilization[o.provider_id] = {}
        for w in all_weeks:
            if w in remaining_capacity[o.provider_id]:
                used = o.capacity_hrs_per_week - remaining_capacity[o.provider_id][w]
                utilization[o.provider_id][w] = round(used / o.capacity_hrs_per_week * 100, 1)

    avg_utilization = {}
    for o in providers:
        vals = list(utilization[o.provider_id].values())
        avg_utilization[o.provider_id] = round(sum(vals) / len(vals), 1) if vals else 0.0

    avg_distance = round(sum(distance_log.values()) / len(distance_log), 2) if distance_log else 0.0

    util_values = list(avg_utilization.values())
    mean_util = sum(util_values) / len(util_values) if util_values else 0
    std_util = round(math.sqrt(sum((v - mean_util) ** 2 for v in util_values) / len(util_values)), 2) if util_values else 0.0

    overcapacity_weeks = {
        o.provider_id: sum(1 for v in utilization[o.provider_id].values() if v > 100) for o in providers
    }

    return {
        'total_assigned': len(assignment_map),
        'avg_travel_hours': avg_distance,
        'avg_utilization_%': avg_utilization,
        'utilization_std_dev_%': std_util,
        'overcapacity_weeks': overcapacity_weeks
    }


def print_results(method_name: str, result: dict, providers: list):
    kpis = result['kpis']
    
    print("\n" + "=" * 70)
    print(f"  STRATEGY: {method_name.upper()}")
    print("=" * 70)
    
    print(f"  Total Assigned          : {kpis['total_assigned']}")
    print(f"  Avg. Travel Time        : {kpis['avg_travel_hours']:.2f} hrs")
    print(f"  Utilization Std. Dev.   : {kpis['utilization_std_dev_%']:.2f}%")
    print("-" * 70)
    
    print(f"  {'Organization':<20} | {'Avg. Utilization':<18} | {'Overcap. Weeks':<15}")
    print("  " + "-" * 66)
    
    for o in providers:
        oid = o.provider_id
        avg_util = kpis['avg_utilization_%'].get(oid, 0.0)
        overcap_w = kpis['overcapacity_weeks'].get(oid, 0)
        
        print(f"  {oid:<20} | {f'{avg_util:.1f}%':>18} | {f'{overcap_w} week(s)':>15}")
        
    print("=" * 70 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import copy

    try:
        base_patients = load_patients_from_csv("patients.csv")
        base_providers = load_providers_from_csv("providers.csv")

        methods_to_run = ['greedy', 'nearest', 'round_robin', 'edd']

        for method in methods_to_run:
            patients_copy = copy.deepcopy(base_patients)
            providers_copy = copy.deepcopy(base_providers)
            
            result = rolling_horizon_assignment(
                patients=patients_copy,
                providers=providers_copy,
                alpha=0.6,
                lookahead_days=7,
                method=method
            )
            
            print_results(method, result, base_providers)
            
    except FileNotFoundError:
        print("Notice: Please place 'patients.csv' and 'providers.csv' in the same directory to run.")