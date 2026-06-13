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

              // Bezettingsgraad: hoe vol is de provider al? (mag > 1 worden)
              load     = MAX over active_weeks of
                         (1 - remaining_capacity[o][w]
                              / o.capacity_hrs_per_week)
              distance = travel_hrs[p][o]

              // OVERCAPACITY PENALTY (zachte grens, geen afwijzing)
              // Als toewijzing capaciteit zou overschrijden, voeg
              // zware straf toe i.p.v. provider uit te sluiten.
              overcap_penalty = 0
              FOR each week w in p.active_weeks:
                  needed  = p.visit_hours + travel_hrs[p][o]
                  deficit = needed - remaining_capacity[o][w]
                  IF deficit > 0:
                      overcap_penalty = MAX(overcap_penalty,
                                             PENALTY_WEIGHT * deficit)

              score = α * load + (1 - α) * distance + overcap_penalty

              IF score < best_score:
                  best_score    = score
                  best_provider = o

          // Patiënt wordt ALTIJD toegewezen (geen afwijzing mogelijk)
          FOR each week w in p.active_weeks:
              remaining_capacity[best_provider][w] -=
                  (p.visit_hours + travel_hrs[p][best_provider])
              // Let op: kan negatief worden = overschrijding capaciteit
          assignment_map[p] = best_provider

  ────────────────────────────────────────────────────
  STEP 6 — Output & KPIs
  ────────────────────────────────────────────────────
  RETURN assignment_map
  RETURN remaining_capacity   // negatieve waarden = overschrijding

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


# =============================================================================
# CSV INLEZEN
# =============================================================================

def load_providers_from_csv(filepath: str) -> list:
    """
    Laadt thuiszorgorganisaties in vanuit een CSV-bestand.

    Verwacht formaat (kolomkoppen exact zo benoemd):

        provider_id,latitude,longitude,capacity_hrs_per_week
        ThuiszorgA,52.21,6.89,80.0
        ThuiszorgB,52.24,6.93,80.0
        ThuiszorgC,52.19,6.86,75.0

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

    providers = []
    for _, row in df.iterrows():
        providers.append(Provider(
            provider_id=row['provider_id'],
            coords=(float(row['latitude']), float(row['longitude'])),
            capacity_hrs_per_week=float(row['capacity_hrs_per_week'])
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

                # Bereken score
                active_w = patient_active_weeks[p.patient_id]

                if active_w:
                    # Bezettingsgraad: hoe vol is de provider al?
                    # Kan > 1 worden bij overschrijding (zachte grens)
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

                # OVERCAPACITY PENALTY
                # Als deze toewijzing de capaciteit zou overschrijden,
                # voegen we een zware extra straf toe zodat het algoritme
                # overschrijding alleen kiest als er geen alternatief is.
                overcapacity_penalty = 0.0
                for w in active_w:
                    if w in remaining_capacity[o.provider_id]:
                        needed = (p.visit_hours +
                                  travel_hrs[p.patient_id][o.provider_id])
                        deficit = needed - remaining_capacity[o.provider_id][w]
                        if deficit > 0:
                            # Straf proportioneel aan het tekort,
                            # met grote constante om altijd te domineren
                            # boven load/afstand verschillen
                            overcapacity_penalty = max(
                                overcapacity_penalty,
                                OVERCAPACITY_PENALTY_WEIGHT * deficit
                            )

                score = (alpha * load
                         + (1 - alpha) * distance_normalized
                         + overcapacity_penalty)

                if score < best_score:
                    best_score    = score
                    best_provider = o

            # Wijs altijd toe (patiënt kan nooit afgewezen worden)
            # Boek capaciteit; remaining_capacity mag negatief worden
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

            processed.add(p.patient_id)

    # -------------------------------------------------------------------------
    # STEP 6: Bereken KPIs
    # -------------------------------------------------------------------------
    kpis = compute_kpis(
        assignment_map, remaining_capacity,
        providers, all_weeks, travel_log, patients
    )

    return {
        'assignments':        assignment_map,
        'remaining_capacity': remaining_capacity,
        'kpis':               kpis
    }


# =============================================================================
# KPI BEREKENING
# =============================================================================

def compute_kpis(assignment_map, remaining_capacity,
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

    # Aantal weken met overschrijding (>100% bezetting) per provider
    overcapacity_weeks = {}
    for o in providers:
        overcapacity_weeks[o.provider_id] = sum(
            1 for v in utilization[o.provider_id].values() if v > 100
        )

    return {
        'total_assigned':        len(assignment_map),
        'avg_travel_hrs':        avg_travel,
        'avg_utilization_%':     avg_utilization,
        'utilization_std_dev_%': std_util,
        'overcapacity_weeks':    overcapacity_weeks,
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

    # KPIs
    kpis = result['kpis']

    # Waarschuwing bij overschrijding van capaciteit
    overcap = kpis['overcapacity_weeks']
    if any(v > 0 for v in overcap.values()):
        print(f"\n⚠️  CAPACITEITSOVERSCHRIJDING (weken > 100%):")
        for oid, weeks in overcap.items():
            if weeks > 0:
                print(f"    - {oid}: {weeks} week(en)")

    print("\n📊 KPIs:")
    print(f"  Totaal toegewezen       : {kpis['total_assigned']}")
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

    # Patiënten en thuiszorgorganisaties inladen vanuit CSV
    patients  = load_patients_from_csv("patients.csv")
    providers = load_providers_from_csv("providers.csv")

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