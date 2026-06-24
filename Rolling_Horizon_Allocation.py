"""
TACTISCH THUISZORG TOEWIJZINGSALGORITME — ROLLING HORIZON (AFSTANDSGEBASEERD)
=============================================================================

PSEUDOCODE (volledig):
----------------------

INPUT:
  Patients  = [{id, discharge_date, length_of_stay,
                visit_hours, home_coords}]
  Providers = [{id, coords, capacity_hrs_per_week}]
  α         = gewicht voor load balancing vs reisafstand (0–1)
  lookahead = aantal dagen vooruitkijken per planningsronde

────────────────────────────────────────────────────────
OUTER LOOP — Rolling Horizon
────────────────────────────────────────────────────────
  planning_moments = alle unieke discharge_dates gesorteerd

  initialiseer remaining_capacity[o][w] voor alle providers en weken
  initialiseer assignment_map = {}

  FOR each planning_moment t in planning_moments:

      known_patients = patiënten met discharge_date
                       in [t, t + lookahead_days]

      sla over als known_patients leeg is

      ────────────────────────────────────────────────
      STEP 1 — Bepaal lokale horizon voor deze ronde
      ────────────────────────────────────────────────
      horizon_start = MIN discharge_date in known_patients
      horizon_end   = MAX discharge_date in known_patients
      weeks         = weekstartdatums van horizon_start t/m horizon_end

      ────────────────────────────────────────────────
      STEP 2 — Bepaal actieve weken per patiënt (afkappen)
      ────────────────────────────────────────────────
      FOR each patient p in known_patients:
          care_end       = discharge_date + length_of_stay
          active_weeks   = {w | discharge_date <= w
                               < MIN(care_end, horizon_end)}

      ────────────────────────────────────────────────
      STEP 3 — Bereken afstand per combinatie
      ────────────────────────────────────────────────
      FOR each patient p, FOR each provider o:
          distance_km[p][o] = haversine(p.coords, o.coords)

      ────────────────────────────────────────────────
      STEP 4 — Sorteer patiënten (zwaarste eerst)
      ────────────────────────────────────────────────
      patients_sorted = SORT known_patients BY visit_hours DESCENDING

      ────────────────────────────────────────────────
      STEP 5 — Greedy toewijzing
      ────────────────────────────────────────────────
      FOR each patient p in patients_sorted:

          sla over als p al toegewezen is

          best_provider = NULL
          best_score    = +∞

          FOR each provider o:

              // Bezettingsgraad: hoe vol is de provider al? (mag > 1 worden)
              load     = MAX over active_weeks of
                         (1 - remaining_capacity[o][w]
                              / o.capacity_hrs_per_week)
              distance = distance_km[p][o]

              // OVERCAPACITY PENALTY (zachte grens, geen afwijzing)
              overcap_penalty = 0
              FOR each week w in p.active_weeks:
                  needed  = p.visit_hours
                  deficit = needed - remaining_capacity[o][w]
                  IF deficit > 0:
                      overcap_penalty = MAX(overcap_penalty,
                                            PENALTY_WEIGHT * deficit)

              score = α * load + (1 - α) * distance_normalized + overcap_penalty

              IF score < best_score:
                  best_score    = score
                  best_provider = o

          // Patiënt toewijzen en capaciteit afboeken
          FOR each week w in p.active_weeks:
              remaining_capacity[best_provider][w] -= p.visit_hours
              
          assignment_map[p] = best_provider

  ────────────────────────────────────────────────────
  STEP 6 — Output & KPIs
  ────────────────────────────────────────────────────
  RETURN assignment_map
  RETURN remaining_capacity

  KPIs:
    - Bezettingsgraad per organisatie per week (%, kan > 100%)
    - Gemiddelde reisafstand per toewijzing (km)
    - Aantal weken met overschrijding per organisatie
    - Standaarddeviatie in belasting tussen organisaties

=============================================================================
PYTHON IMPLEMENTATIE
=============================================================================
"""

import math
import pandas as pd
from datetime import date, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional


# =============================================================================
# CONSTANTEN
# =============================================================================

# Gewicht voor de overcapaciteit-penalty in de scorefunctie.
# Hoog genoeg gekozen zodat een toewijzing zonder overschrijding
# ALTIJD de voorkeur krijgt boven een toewijzing met overschrijding,
# ongeacht load/afstand verschillen (die liggen in range [0,1]).
OVERCAPACITY_PENALTY_WEIGHT = 10.0


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Patient:
    patient_id: str
    discharge_date: date
    length_of_stay: int          # in dagen
    visit_hours: float           # uren zorg per week
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
    initial_load_hrs_per_week: float = 0.0  # lopende caseload in uren


# =============================================================================
# CSV INLEZEN
# =============================================================================

def load_providers_from_csv(filepath: str) -> list:
    """
    Laadt thuiszorgorganisaties in vanuit een CSV-bestand.

    Verwacht formaat (kolomkoppen exact zo benoemd):

        provider_id,latitude,longitude,capacity_hrs_per_week,initial_load_hrs_per_week
        ThuiszorgA,52.21,6.89,80.0,55.0
        ThuiszorgB,52.24,6.93,80.0,40.0
        ThuiszorgC,52.19,6.86,75.0,60.0

    'initial_load_hrs_per_week' is optioneel — ontbreekt deze kolom,
    dan wordt 0.0 aangenomen (lege start).

    Returns:
    --------
    list van Provider objecten
    """
    df = pd.read_csv(filepath, dtype={'provider_id': str})

    required_cols = {'provider_id', 'latitude', 'longitude',
                      'capacity_hrs_per_week'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV mist verplichte kolommen: {missing}")

    has_initial_load = 'initial_load_hrs_per_week' in df.columns

    providers = []
    for _, row in df.iterrows():
        providers.append(Provider(
            provider_id=row['provider_id'],
            coords=(float(row['latitude']), float(row['longitude'])),
            capacity_hrs_per_week=float(row['capacity_hrs_per_week']),
            initial_load_hrs_per_week=float(row['initial_load_hrs_per_week'])
                if has_initial_load else 0.0
        ))

    return providers


def load_patients_from_csv(filepath: str) -> list:
    """
    Laadt patiënten in vanuit een CSV-bestand.

    Verwacht formaat (kolomkoppen exact zo benoemd, eventuele extra
    kolommen zoals nurse_skill/type_care worden genegeerd):

        patient_id,discharge_date,length_of_stay,visit_hours,latitude,longitude
        P0001,2024-01-03,28,4,52.30,5.76
        P0002,2024-01-05,21,3,52.78,6.52

    discharge_date moet leesbaar zijn als datum (bijv. YYYY-MM-DD).

    Returns:
    --------
    list van Patient objecten
    """
    df = pd.read_csv(filepath, dtype={'patient_id': str},
                     parse_dates=['discharge_date'])

    required_cols = {'patient_id', 'discharge_date', 'length_of_stay',
                      'visit_hours', 'latitude', 'longitude'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV mist verplichte kolommen: {missing}")

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
# HULPFUNCTIES
# =============================================================================

def haversine_km(coord1: tuple, coord2: tuple) -> float:
    """ Berekent de hemelsbrede afstand in kilometers. """
    R = 6371 #radius of the earth in km
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
    return [w for w in all_weeks if patient.discharge_date <= w < effective_end]


# =============================================================================
# TRAVEL TIME CALCULATION
# =============================================================================

# Average driving speed for home care in urban/rural areas (km/h).
# Literature values range between 25–40 km/h; 30 is a common choice.
TRAVEL_SPEED_KMH = 30.0

# Maximum travel time (hours) used for normalisation to [0, 1].
# 1 hour = a realistic hard ceiling for home care trips in the Netherlands.
MAX_TRAVEL_HOURS = 1.0


def travel_hours(distance_km: float, speed_kmh: float = TRAVEL_SPEED_KMH) -> float:
    """
    Converts straight-line distance to estimated travel time in hours.

    Uses a fixed average speed as a proxy for driving time.
    Straight-line distance underestimates road distance — a detour
    factor of ~1.3 (standard for the Netherlands) is applied internally.

    Parameters:
    -----------
    distance_km : float — straight-line distance in km
    speed_kmh   : float — average driving speed in km/h (default 30)

    Returns:
    --------
    float — estimated travel time in hours
    """
    DETOUR_FACTOR = 1.3  #straight-line → actual road distance
    road_distance_km = distance_km * DETOUR_FACTOR
    return road_distance_km / speed_kmh


# =============================================================================
# TOEWIJZINGSMETHODES
# =============================================================================

def _book_capacity(p, best_provider, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed):
    """ Boekt patiënturen op de capaciteit en logt de afstand. """
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
                penalty = max(penalty, OVERCAPACITY_PENALTY_WEIGHT * deficit)
    return penalty


def method_greedy_heaviest_first(patients, providers, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed, alpha, **kwargs):
    """
    METHOD 1 — Most severe patient first + scored greedy assignment.

    Sorts patients from high to low based on visit_hours so that the
    most severe (hardest to place) patients get a spot first.
    Selects the provider with the lowest combined
    score of occupancy rate (load) and travel distance for each patient, weighted via alpha.
    """
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

            # Normaliseer afstand fictief tot max 15km voor de scorebalans
            t_hours = travel_hours(distance_km[p.patient_id][o.provider_id])
            distance_normalized = min(t_hours / MAX_TRAVEL_HOURS, 1.0)
            penalty = _overcapacity_penalty(p, o, patient_active_weeks, remaining_capacity)
            
            score = (alpha * load) + ((1 - alpha) * distance_normalized) + penalty

            if score < best_score:
                best_score = score
                best_provider = o

        _book_capacity(p, best_provider, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed)


def method_nearest_provider(patients, providers, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed, **kwargs):
    """ 
    METHOD 2 — Nearest provider (purely geographical).

    Assigns each patient to the provider with the shortest distance,
    regardless of current occupancy. Overcapacity is only taken into account as a
    tiebreaker via the penalty (not as the primary score). 
    """
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
    """
    METHOD 3 — Round-robin (strictly rotating).

    Assigns patients to providers in turn in a fixed order,
    regardless of load or distance. Serves as a neutral benchmark: shows
    how much the smarter methods actually contribute.
    """
    patients_sorted = sorted(patients, key=lambda p: p.discharge_date)

    for p in patients_sorted:
        if p.patient_id in processed:
            continue

        chosen = providers[round_robin_index[0] % len(providers)]
        round_robin_index[0] += 1

        _book_capacity(p, chosen, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed)


def method_edd(patients, providers, patient_active_weeks, distance_km, remaining_capacity, assignment_map, distance_log, processed, alpha=0.5, **kwargs):
    """
    METHOD 4 — Earliest Due Date (EDD).

    Sorts patients strictly by the earliest discharge date (discharge_date).
    Patients who need care NOW are assigned a provider first.
    Within the assignment, load and distance are balanced via alpha.
    """
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

            score = (alpha * load) + ((1 - alpha) * distance_normalized) + penalty

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
        raise ValueError(f"Onbekende methode '{method}'.")
    

# =============================================================================
# ROLLING HORIZON ALGORITME
# =============================================================================

def rolling_horizon_assignment(patients: list, providers: list, alpha: float = 0.5, lookahead_days: int = 7, method: str = 'greedy') -> dict:
    """
    Assigns patients to home care organizations via a rolling horizon algorithm.

    Parameters:
    -----------
    patients : list of Patient objects
    providers : list of Provider objects
    alpha : weight load balancing (0=distance only, 1=load only)
    only relevant for method='greedy' and method='edd'
    lookahead_days: how many days ahead per planning round
    method : assignment method, choose from:

    'greedy' — heaviest patient first + load/distance score
    'nearest' — nearest provider
    'round_robin' — strictly rotating across providers
    'edd' — earliest due date
    Returns:
    --------
    dict with:

    'assignments' : {patient_id: provider_id}
    'remaining_capacity' : {provider_id: {week: remaining_hours}}
    'kpis' : dict with evaluation metrics
    """
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
        weeks = generate_weeks(t, horizon_end)

        patient_active_weeks = {p.patient_id: active_weeks_for_patient(p, horizon_end, weeks) for p in known_patients}

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
# KPI BEREKENING & WEERGAVE
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
    print(f"  STRATEGIE: {method_name.upper()}")
    print("=" * 70)
    
    print(f"  Totaal toegewezen       : {kpis['total_assigned']}")
    print(f"  Avg. travel time        : {kpis['avg_travel_hours']:.2f} hrs")
    print(f"  Spreiding bezetting     : {kpis['utilization_std_dev_%']:.2f}%")
    print("-" * 70)
    
    print(f"  {'Organisatie':<20} | {'Gem. Bezetting':<18} | {'Overcap. Weken':<15}")
    print("  " + "-" * 66)
    
    for o in providers:
        oid = o.provider_id
        avg_util = kpis['avg_utilization_%'].get(oid, 0.0)
        overcap_w = kpis['overcapacity_weeks'].get(oid, 0)
        
        print(f"  {oid:<20} | {f'{avg_util:.1f}%':>18} | {f'{overcap_w} week(en)':>15}")
        
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
        print("Let op: Plaats 'patients.csv' en 'providers.csv' in dezelfde map om te runnen.")