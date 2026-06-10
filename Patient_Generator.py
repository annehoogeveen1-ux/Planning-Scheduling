import numpy as np
import pandas as pd
import random
from geopy.distance import geodesic

from config import location_centre

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
# Location generator
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

    visit_frequency = int(np.random.choice(
        dist_visit_frequency['uur/week'],
        p=dist_visit_frequency['CDF Bucket']
    ))

    location = generate_patient_location()

    return {
        "patient_id": patient_counter,
        "nurse_skill": nurse_skill,
        "type_care": type_care,
        "visit_frequency": visit_frequency,
        "latitude": location["latitude"],
        "longitude": location["longitude"]
    }

# -----------------------------
# Optional batch generator
# -----------------------------
def generate_dataset(n=100):
    patients = []

    for _ in range(n):
        patients.append(generate_patient())

    return pd.DataFrame(patients)