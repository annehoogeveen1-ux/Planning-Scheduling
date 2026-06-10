import pandas as pd
import numpy as np
import random
from geopy.distance import geodesic

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
# Location generator (50 km radius)
# -----------------------------
def generate_patient_location():
    centre_lat = 52.5136992436934
    centre_lon = 6.123670119684271
    radius_km = 50

    distance = random.uniform(0, radius_km)
    bearing = random.uniform(0, 360)

    destination = geodesic(kilometers=distance).destination((centre_lat, centre_lon),bearing)

    return {
        "latitude": destination.latitude,
        "longitude": destination.longitude
    }


# -----------------------------
# Patient generator
# -----------------------------
patient_counter = 0

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

    # ✔ USE YOUR OWN LOCATION FUNCTION
    location = generate_patient_location()

    return {
        'patient_id': patient_counter,
        'nurse_skill': nurse_skill,
        'type_care': type_care,
        'visit_frequency': visit_frequency,
        'latitude': location['latitude'],
        'longitude': location['longitude']
    }
# -----------------------------
# Generate dataset
# -----------------------------
patients_list = []
print("Starting patient generation...\n")

for i in range(100):
    patient_data = generate_patient()
    patients_list.append(patient_data)

    print(f"Generated patient {i+1}/100")

# -----------------------------
# Save to CSV
# -----------------------------
df = pd.DataFrame(patients_list)
df.to_csv("patients.csv", index=False)

print("\nSuccess! Generated 100 patients and saved to patients.csv")