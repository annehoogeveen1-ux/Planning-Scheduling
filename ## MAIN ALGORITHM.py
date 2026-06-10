import pandas as pd
import numpy as np
import Random Address.py

dist_nurse_skill = pd.read_csv('P_Nurse_Skills.csv', sep=';', decimal=',')
dist_type_care = pd.read_csv('P_Type_Care.csv', sep=';', decimal=',', float_precision='round_trip')
dist_visit_frequency = pd.read_csv('P_Visit_Frequency.csv', sep=';', decimal=',')

# Normalize all distributions to ensure they sum to 1
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
    
    return {
        'nurse_skill': nurse_skill,
        'type_care': type_care,
        'visit_frequency': visit_frequency
    }

def reset_patient_counter():
    global patient_counter
    patient_counter = 0


while True:
    command = input("\nEnter command (new / reset / quit): ").strip().lower()

    if command == 'new':
        patient = generate_patient()
        print(f"Patient {patient_counter}: {patient}")

    elif command == 'reset':
        reset_patient_counter()
        print("Patient counter reset to 0.")

    elif command == 'quit':
        print("Exiting.")
        break

    else:
        print("Unknown command. Use 'new', 'reset', or 'quit'.")

