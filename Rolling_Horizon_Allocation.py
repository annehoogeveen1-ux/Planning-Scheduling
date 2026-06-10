"""
=============================================================================
TACTISCH THUISZORG TOEWIJZINGSALGORITME — ROLLING HORIZON
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
  initialiseer unassignable = []

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
      STEP 3 — Bereken reistijd per combinatie
      ────────────────────────────────────────────────
      FOR each patient p, FOR each provider o:
          travel_hrs[p][o] = haversine(p.coords, o.coords)
                             / avg_speed_kmh * 2   // heen + terug

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

              // Check feasibility: genoeg capaciteit in alle actieve weken?
              feasible = TRUE
              FOR each week w in p.active_weeks:
                  needed = p.visit_hours + travel_hrs[p][o]
                  IF remaining_capacity[o][w] < needed:
                      feasible = FALSE; BREAK

              IF NOT feasible: SKIP

              // Scorefunctie: combineer bezetting en reisafstand
              load     = MAX over active_weeks of
                         (1 - remaining_capacity[o][w]
                              / o.capacity_hrs_per_week)
              distance = travel_hrs[p][o]
              score    = α * load + (1 - α) * distance

              IF score < best_score:
                  best_score    = score
                  best_provider = o

          // Wijs toe of markeer als niet plaatsbaar
          IF best_provider != NULL:
              FOR each week w in p.active_weeks:
                  remaining_capacity[best_provider][w] -=
                      (p.visit_hours + travel_hrs[p][best_provider])
              assignment_map[p] = best_provider
          ELSE:
              unassignable.append(p)

  ────────────────────────────────────────────────────
  STEP 6 — Output & KPIs
  ────────────────────────────────────────────────────
  RETURN assignment_map
  RETURN remaining_capacity
  RETURN unassignable

  KPIs:
    - Bezettingsgraad per organisatie per week (%)
    - Gemiddelde reisafstand per toewijzing (km)
    - Aantal niet-geplaatste patiënten
    - Standaarddeviatie in belasting tussen organisaties

=============================================================================
PYTHON IMPLEMENTATIE
=============================================================================
"""

import math
from datetime import date, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional


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


# =============================================================================
# HULPFUNCTIES
# =============================================================================

def haversine_km(coord1: tuple, coord2: tuple) -> float:
    """
    Berekent de afstand in km tussen twee GPS-coördinaten
    via de Haversine-formule (crow-vlucht).
    """
    R = 6371  # straal aarde in km
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + \
        math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def travel_hours(coord1: tuple, coord2: tuple,
                 avg_speed_kmh: float = 30.0) -> float:
    """
    Schat reistijd in uren (heen + terug) op basis van afstand.
    Standaard gemiddelde snelheid: 30 km/u (stedelijk rijden).
    """
    distance = haversine_km(coord1, coord2)
    return (distance / avg_speed_kmh) * 2  # heen + terug


def generate_weeks(start: date, end: date) -> list:
    """
    Genereert een lijst van weekstartdatums (maandag)
    van start t/m end.
    """
    # Ga terug naar de dichtstbijzijnde maandag
    current = start - timedelta(days=start.weekday())
    weeks = []
    while current <= end:
        weeks.append(current)
        current += timedelta(weeks=1)
    return weeks


def active_weeks_for_patient(patient: Patient,
                              horizon_end: date,
                              all_weeks: list) -> list:
    """
    Geeft de weken terug waarin een patiënt actief is,
    afgekapt op horizon_end.
    """
    effective_end = min(patient.care_end, horizon_end)
    return [
        w for w in all_weeks
        if patient.discharge_date <= w < effective_end
    ]


# =============================================================================
# ROLLING HORIZON ALGORITME
# =============================================================================

def rolling_horizon_assignment(
    patients: list,
    providers: list,
    alpha: float = 0.5,
    lookahead_days: int = 7,
    avg_speed_kmh: float = 30.0
) -> dict:
    """
    Wijst patiënten toe aan thuiszorgorganisaties via een
    rolling horizon greedy algoritme.

    Parameters:
    -----------
    patients      : lijst van Patient objecten
    providers     : lijst van Provider objecten
    alpha         : gewicht load balancing (0=alleen afstand, 1=alleen load)
    lookahead_days: hoeveel dagen vooruit per planningsronde
    avg_speed_kmh : gemiddelde rijsnelheid voor reistijdschatting

    Returns:
    --------
    dict met:
      'assignments'        : {patient_id: provider_id}
      'unassignable'       : [patient_ids]
      'remaining_capacity' : {provider_id: {week: resterende_uren}}
      'kpis'               : dict met evaluatiemetrieken
    """

    # -------------------------------------------------------------------------
    # Initialisatie
    # -------------------------------------------------------------------------

    # Bepaal de globale horizon over alle patiënten
    all_discharge_dates = sorted(set(p.discharge_date for p in patients))
    global_horizon_start = min(p.discharge_date for p in patients)
    global_horizon_end   = max(p.discharge_date for p in patients)
    all_weeks = generate_weeks(global_horizon_start, global_horizon_end)

    # Capaciteitsboekhouding: remaining_capacity[provider_id][week] = uren
    remaining_capacity = {
        o.provider_id: {w: o.capacity_hrs_per_week for w in all_weeks}
        for o in providers
    }

    # Resultaten
    assignment_map  = {}   # patient_id → provider_id
    unassignable    = []   # patient_ids die niet geplaatst konden worden
    travel_log      = {}   # patient_id → reisuren naar toegewezen provider

    # Bijhoud welke patiënten al verwerkt zijn
    processed = set()

    # -------------------------------------------------------------------------
    # OUTER LOOP: Rolling Horizon
    # Elke unieke ontslagdatum is een planningsmoment
    # -------------------------------------------------------------------------
    for t in all_discharge_dates:

        # Selecteer patiënten die ontslagen worden binnen het lookahead venster
        window_end = t + timedelta(days=lookahead_days)
        known_patients = [
            p for p in patients
            if t <= p.discharge_date <= window_end
            and p.patient_id not in processed
        ]

        if not known_patients:
            continue

        # ---------------------------------------------------------------------
        # STEP 1: Bepaal lokale horizon voor deze planningsronde
        # ---------------------------------------------------------------------
        horizon_start = min(p.discharge_date for p in known_patients)
        horizon_end   = max(p.discharge_date for p in known_patients)
        weeks         = generate_weeks(horizon_start, horizon_end)

        # ---------------------------------------------------------------------
        # STEP 2: Bepaal actieve weken per patiënt (met afkappen)
        # ---------------------------------------------------------------------
        patient_active_weeks = {}
        for p in known_patients:
            patient_active_weeks[p.patient_id] = active_weeks_for_patient(
                p, horizon_end, weeks
            )

        # ---------------------------------------------------------------------
        # STEP 3: Bereken reistijd per patiënt-provider combinatie
        # ---------------------------------------------------------------------
        travel_hrs = {}
        for p in known_patients:
            travel_hrs[p.patient_id] = {}
            for o in providers:
                travel_hrs[p.patient_id][o.provider_id] = travel_hours(
                    p.home_coords, o.coords, avg_speed_kmh
                )

        # ---------------------------------------------------------------------
        # STEP 4: Sorteer patiënten op zorgvraag (zwaarste eerst)
        # ---------------------------------------------------------------------
        patients_sorted = sorted(
            known_patients,
            key=lambda p: p.visit_hours,
            reverse=True
        )

        # ---------------------------------------------------------------------
        # STEP 5: Greedy toewijzing
        # ---------------------------------------------------------------------
        for p in patients_sorted:

            if p.patient_id in processed:
                continue

            best_provider = None
            best_score    = float('inf')

            for o in providers:

                # Check feasibility: genoeg capaciteit in alle actieve weken?
                feasible = True
                for w in patient_active_weeks[p.patient_id]:
                    needed = p.visit_hours + travel_hrs[p.patient_id][o.provider_id]
                    if w not in remaining_capacity[o.provider_id]:
                        feasible = False
                        break
                    if remaining_capacity[o.provider_id][w] < needed:
                        feasible = False
                        break

                if not feasible:
                    continue

                # Bereken score
                active_w = patient_active_weeks[p.patient_id]

                if active_w:
                    # Bezettingsgraad: hoe vol is de provider al?
                    load = max(
                        1 - remaining_capacity[o.provider_id][w]
                            / o.capacity_hrs_per_week
                        for w in active_w
                        if w in remaining_capacity[o.provider_id]
                    )
                else:
                    load = 0.0

                distance = travel_hrs[p.patient_id][o.provider_id]

                # Normaliseer afstand (max ~2 uur reistijd als referentie)
                distance_normalized = min(distance / 2.0, 1.0)

                score = alpha * load + (1 - alpha) * distance_normalized

                if score < best_score:
                    best_score    = score
                    best_provider = o

            # Wijs toe of markeer als niet plaatsbaar
            if best_provider is not None:
                # Boek capaciteit
                for w in patient_active_weeks[p.patient_id]:
                    if w in remaining_capacity[best_provider.provider_id]:
                        remaining_capacity[best_provider.provider_id][w] -= (
                            p.visit_hours +
                            travel_hrs[p.patient_id][best_provider.provider_id]
                        )

                assignment_map[p.patient_id] = best_provider.provider_id
                travel_log[p.patient_id] = (
                    travel_hrs[p.patient_id][best_provider.provider_id]
                )
            else:
                unassignable.append(p.patient_id)

            processed.add(p.patient_id)

    # -------------------------------------------------------------------------
    # STEP 6: Bereken KPIs
    # -------------------------------------------------------------------------
    kpis = compute_kpis(
        assignment_map, unassignable, remaining_capacity,
        providers, all_weeks, travel_log, patients
    )

    return {
        'assignments':        assignment_map,
        'unassignable':       unassignable,
        'remaining_capacity': remaining_capacity,
        'kpis':               kpis
    }


# =============================================================================
# KPI BEREKENING
# =============================================================================

def compute_kpis(assignment_map, unassignable, remaining_capacity,
                 providers, all_weeks, travel_log, patients) -> dict:
    """
    Berekent evaluatiemetrieken over de volledige planning.
    """
    provider_map = {o.provider_id: o for o in providers}

    # Bezettingsgraad per provider per week (%)
    utilization = {}
    for o in providers:
        utilization[o.provider_id] = {}
        for w in all_weeks:
            if w in remaining_capacity[o.provider_id]:
                used = (o.capacity_hrs_per_week
                        - remaining_capacity[o.provider_id][w])
                utilization[o.provider_id][w] = round(
                    used / o.capacity_hrs_per_week * 100, 1
                )

    # Gemiddelde bezettingsgraad per provider
    avg_utilization = {}
    for o in providers:
        vals = list(utilization[o.provider_id].values())
        avg_utilization[o.provider_id] = round(
            sum(vals) / len(vals), 1
        ) if vals else 0.0

    # Gemiddelde reistijd per toewijzing (uren)
    avg_travel = (
        round(sum(travel_log.values()) / len(travel_log), 2)
        if travel_log else 0.0
    )

    # Spreiding in gemiddelde belasting tussen providers (std dev)
    util_values = list(avg_utilization.values())
    mean_util   = sum(util_values) / len(util_values) if util_values else 0
    std_util    = round(
        math.sqrt(sum((v - mean_util) ** 2 for v in util_values)
                  / len(util_values)), 2
    ) if util_values else 0.0

    return {
        'total_assigned':        len(assignment_map),
        'total_unassignable':    len(unassignable),
        'avg_travel_hrs':        avg_travel,
        'avg_utilization_%':     avg_utilization,
        'utilization_std_dev_%': std_util,
        'utilization_per_week':  utilization
    }


# =============================================================================
# RESULTATEN WEERGAVE
# =============================================================================

def print_results(result: dict, patients: list, providers: list):
    """
    Print een overzichtelijk rapport van de planningsresultaten.
    """
    print("\n" + "=" * 60)
    print("  THUISZORG TOEWIJZING — RESULTATEN")
    print("=" * 60)

    patient_map  = {p.patient_id: p for p in patients}
    provider_map = {o.provider_id: o for o in providers}

    # Toewijzingen per provider
    print("\n📋 TOEWIJZINGEN PER ORGANISATIE:")
    assignments_by_provider = defaultdict(list)
    for pid, oid in result['assignments'].items():
        assignments_by_provider[oid].append(pid)

    for oid, pids in sorted(assignments_by_provider.items()):
        print(f"\n  {oid} ({len(pids)} patiënten):")
        for pid in pids:
            p = patient_map[pid]
            print(f"    - {pid} | ontslag: {p.discharge_date} | "
                  f"{p.visit_hours} uur/week | "
                  f"{p.length_of_stay} dagen zorg")

    # Niet plaatsbare patiënten
    if result['unassignable']:
        print(f"\n⚠️  NIET PLAATSBAAR ({len(result['unassignable'])}):")
        for pid in result['unassignable']:
            p = patient_map[pid]
            print(f"    - {pid} | {p.visit_hours} uur/week | "
                  f"{p.length_of_stay} dagen")

    # KPIs
    kpis = result['kpis']
    print("\n📊 KPIs:")
    print(f"  Totaal toegewezen       : {kpis['total_assigned']}")
    print(f"  Totaal niet plaatsbaar  : {kpis['total_unassignable']}")
    print(f"  Gem. reistijd           : {kpis['avg_travel_hrs']} uur")
    print(f"  Spreiding bezetting     : {kpis['utilization_std_dev_%']}%")

    print("\n  Gemiddelde bezettingsgraad per organisatie:")
    for oid, util in kpis['avg_utilization_%'].items():
        bar = "█" * int(util / 5)
        print(f"    {oid}: {util:5.1f}%  {bar}")

    print("\n" + "=" * 60)


# =============================================================================
# VOORBEELD / TEST
# =============================================================================

if __name__ == "__main__":

    # Voorbeelddata: patiënten
    patients = [
        Patient("P001", date(2024, 1,  8), 28, 10.0, (52.22, 6.90)),
        Patient("P002", date(2024, 1,  8), 14,  6.0, (52.20, 6.85)),
        Patient("P003", date(2024, 1, 10), 21, 14.0, (52.25, 6.95)),
        Patient("P004", date(2024, 1, 10), 35,  8.0, (52.18, 6.88)),
        Patient("P005", date(2024, 1, 12), 14, 12.0, (52.23, 6.92)),
        Patient("P006", date(2024, 1, 15), 28,  5.0, (52.19, 6.87)),
        Patient("P007", date(2024, 1, 15), 21,  9.0, (52.26, 6.93)),
        Patient("P008", date(2024, 1, 18), 14, 11.0, (52.21, 6.89)),
        Patient("P009", date(2024, 1, 20), 28,  7.0, (52.24, 6.91)),
        Patient("P010", date(2024, 1, 22), 21, 13.0, (52.17, 6.86)),
    ]

    # Voorbeelddata: thuiszorgorganisaties (in Enschede omgeving)
    providers = [
        Provider("ThuiszorgA", (52.21, 6.89), capacity_hrs_per_week=80.0),
        Provider("ThuiszorgB", (52.24, 6.93), capacity_hrs_per_week=80.0),
        Provider("ThuiszorgC", (52.19, 6.86), capacity_hrs_per_week=80.0),
    ]

    # Draai het algoritme
    # alpha=0.6: lichte voorkeur voor gelijke spreiding boven minimale reistijd
    result = rolling_horizon_assignment(
        patients      = patients,
        providers     = providers,
        alpha         = 0.6,
        lookahead_days= 7,
        avg_speed_kmh = 30.0
    )

    # Print resultaten
    print_results(result, patients, providers)