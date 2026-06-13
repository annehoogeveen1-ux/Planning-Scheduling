"""
=============================================================================
PATIENT DATA GENERATOR — aansluitend op rolling_horizon_assignment()
=============================================================================

Doel:
  Genereert een patients.csv met exact de kolommen die het toewijzings-
  algoritme verwacht:

      patient_id, discharge_date, length_of_stay,
      visit_hours, latitude, longitude

  De originele distributies (P_Nurse_Skills.csv, P_Type_Care.csv,
  P_Visit_Frequency.csv) en de locatiegenerator blijven gebruikt.
  nurse_skill en type_care worden mee opgeslagen voor later gebruik,
  maar spelen nog geen rol in het algoritme.

=============================================================================
"""

import numpy as np
import pandas as pd
import random
from datetime import date, timedelta
from geopy.distance import geodesic

from Archive_Old_Files.Config import location_centre


# =============================================================================
# CONFIG: aanpasbare generatie-parameters
# =============================================================================

# Planningshorizon voor ontslagdatums
HORIZON_START_DATE = date(2026, 1, 1)
HORIZON_LENGTH_DAYS = 30          # patiënten worden ontslagen binnen dit venster

# Lengte van het thuiszorgtraject (in dagen), uniform tussen min en max
LENGTH_OF_STAY_MIN_DAYS = 14
LENGTH_OF_STAY_MAX_DAYS = 42


# -----------------------------
# Load distributions
# -----------------------------
dist_nurse_skill = pd.read_csv('P_Nurse_Skills.csv', sep=';', decimal=',')
dist_type_care = pd.read_csv('P_Type_Care.csv', sep=';', decimal=',', float_precision='round_trip')
dist_visit_frequency = pd.read_csv('P_Visit_Frequency.csv', sep=';', decimal=',')

dist_nurse_skill['Dist'] = dist_nurse_skill['Dist'] / dist_nurse_skill['Dist'].sum()
dist_type_care['Distribution'] = dist_type_care['Distribution'] / dist_type_care['Distribution'].sum()
dist_visit_frequency['CDF Bucket'] = dist_visit_frequency['CDF Bucket'] / dist_visit_frequency['CDF Bucket'].sum()

# -----------------------------
# Counter
# -----------------------------
patient_counter = 0


# -----------------------------
# Location generator (ongewijzigd)
# -----------------------------
def generate_patient_location():
    radius_km = 50
    distance = random.uniform(0, radius_km)
    bearing = random.uniform(0, 360)

    destination = geodesic(kilometers=distance).destination(
        (location_centre["centre_lat"], location_centre["centre_lon"]),
        bearing
    )

    return {
        "latitude": destination.latitude,
        "longitude": destination.longitude
    }


# -----------------------------
# Ontslagdatum generator (nieuw)
# -----------------------------
def generate_discharge_date():
    """
    Genereert een willekeurige ontslagdatum binnen de gedefinieerde horizon.
    """
    offset_days = random.randint(0, HORIZON_LENGTH_DAYS - 1)
    return HORIZON_START_DATE + timedelta(days=offset_days)


# -----------------------------
# Verblijfsduur generator (nieuw)
# -----------------------------
def generate_length_of_stay():
    """
    Genereert de duur van het thuiszorgtraject in dagen,
    uniform verdeeld tussen LENGTH_OF_STAY_MIN_DAYS en _MAX_DAYS.
    """
    return random.randint(LENGTH_OF_STAY_MIN_DAYS, LENGTH_OF_STAY_MAX_DAYS)


# -----------------------------
# Patient generator
# -----------------------------
def generate_patient():
    global patient_counter
    patient_counter += 1

    nurse_skill = int(np.random.choice(
        dist_nurse_skill['Niveau'],
        p=dist_nurse_skill['Dist']
    ))

    type_care = str(np.random.choice(
        dist_type_care['Diagnose'],
        p=dist_type_care['Distribution']
    ))

    # "uur/week" sluit direct aan op visit_hours in het algoritme
    visit_hours = int(np.random.choice(
        dist_visit_frequency['uur/week'],
        p=dist_visit_frequency['CDF Bucket']
    ))

    location = generate_patient_location()
    discharge_date = generate_discharge_date()
    length_of_stay = generate_length_of_stay()

    return {
        "patient_id": f"P{patient_counter:04d}",
        "discharge_date": discharge_date,
        "length_of_stay": length_of_stay,
        "visit_hours": visit_hours,
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        # Onderstaande velden worden nog niet gebruikt door het algoritme,
        # maar blijven beschikbaar voor latere uitbreiding
        "nurse_skill": nurse_skill,
        "type_care": type_care,
    }


# -----------------------------
# Batch generator
# -----------------------------
def generate_dataset(n=500):
    patients = []

    for _ in range(n):
        patients.append(generate_patient())

    df = pd.DataFrame(patients)

    # Sorteer op ontslagdatum, zoals het algoritme verwacht
    # voor de rolling horizon volgorde
    # df = df.sort_values("discharge_date").reset_index(drop=True)

    return df


# =============================================================================
# MAIN: genereer en sla op als patients.csv
# =============================================================================

if __name__ == "__main__":
    n_patients = 500

    df = generate_dataset(n=n_patients)
    df.to_csv("patients.csv", index=False)

    print(f"{n_patients} patiënten gegenereerd en opgeslagen in patients.csv")
    print(df.head())