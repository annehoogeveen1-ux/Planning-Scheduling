import pandas as pd
import numpy as np
import Random_Address


dist_nurse_skill = pd.read_csv('P_Nurse_Skills.csv', sep=';', decimal=',')
dist_type_care = pd.read_csv('P_Type_Care.csv', sep=';', decimal=',', float_precision='round_trip')
dist_visit_frequency = pd.read_csv('P_Visit_Frequency.csv', sep=';', decimal=',')

dist_nurse_skill['Dist'] = dist_nurse_skill['Dist'] / dist_nurse_skill['Dist'].sum()
dist_type_care['Distribution'] = dist_type_care['Distribution'] / dist_type_care['Distribution'].sum()
dist_visit_frequency['CDF Bucket'] = dist_visit_frequency['CDF Bucket'] / dist_visit_frequency['CDF Bucket'].sum()

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

    # Get random location
    location = Random_Address.generate_random_address()

    if location is None:
        latitude = None
        longitude = None
    else:
        latitude = location['latitude']
        longitude = location['longitude']

    return {
        'patient_id': patient_counter,
        'nurse_skill': nurse_skill,
        'type_care': type_care,
        'visit_frequency': visit_frequency,
        'latitude': latitude,
        'longitude': longitude
    }

patients = []
for _ in range(100):
    patients.append(generate_patient())
df = pd.DataFrame(patients)
df.to_csv("patients.csv", index=False)
print("Generated 100 patients and saved to patients.csv")
print(df.to_string(index=False))