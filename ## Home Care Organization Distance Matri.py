import pandas as pd
import numpy as np


def generate_distance_matrix_organizations(file_path, separator=';'):
    """
    Leest een zorgorganisatie CSV in en genereert een afstandmatrix in kilometers.
    """
    # 1. # 1. Read existing organization data
    df = pd.read_csv(file_path, sep=separator)
    df.columns = df.columns.str.strip()
    
    lon_col = 'Longditude' if 'Longditude' in df.columns else 'Longitude'
    
    # 2. convert degrees -> radial
    lat_rad = np.radians(df['Latitude'].values)
    lon_rad = np.radians(df[lon_col].values)
    
    # 3. Calc difference between pairs
    dlat = lat_rad[:, np.newaxis] - lat_rad
    dlon = lon_rad[:, np.newaxis] - lon_rad
    
    # 4. the Haversine formula
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_rad[:, np.newaxis]) * np.cos(lat_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    km_matrix = 6371 * c               #radius of the earth = 6371
    km_matrix = np.round(km_matrix, 2) #round on 2 decimals
    
    # 6. make dataframe
    matrix_df = pd.DataFrame(km_matrix, index=df['Organization'], columns=df['Organization'])
    
    return matrix_df

def find_closest_organizations(file_path, num_patients=1, separator=';'):
    # 1. Read existing organization data
    df = pd.read_csv(file_path, sep=separator)
    df.columns = df.columns.str.strip()
    lon_col = 'Longditude' if 'Longditude' in df.columns else 'Longitude'
    
    # 2. Get bounding box of organizations to keep random patients in the same region
    min_lat, max_lat = df['Latitude'].min(), df['Latitude'].max()
    min_lon, max_lon = df[lon_col].min(), df[lon_col].max()
    
    # 3. Generate random patient coordinates
    # np.random.uniform picks a random float between the min and max limits
    patient_lats = np.random.uniform(min_lat, max_lat, num_patients)
    patient_lons = np.random.uniform(min_lon, max_lon, num_patients)
    
    # Convert organization coordinates to radians for the math
    org_lats_rad = np.radians(df['Latitude'].values)
    org_lons_rad = np.radians(df[lon_col].values)
    
    # 4. Loop through each generated patient and calculate distances
    for i in range(num_patients):
        p_lat, p_lon = patient_lats[i], patient_lons[i]
        
        # Convert current patient to radians
        p_lat_rad = np.radians(p_lat)
        p_lon_rad = np.radians(p_lon)
        
        # Haversine formula between this 1 patient and ALL organizations
        dlat = org_lats_rad - p_lat_rad
        dlon = org_lons_rad - p_lon_rad
        
        a = np.sin(dlat / 2.0)**2 + np.cos(p_lat_rad) * np.cos(org_lats_rad) * np.sin(dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances_km = 6371 * c
        
        # 5. Attach distances to a temporary copy of the dataframe
        temp_df = df.copy()
        temp_df['Distance_KM'] = np.round(distances_km, 2)
        
        # Sort by distance and grab the top 2 closest
        closest_two = temp_df.sort_values(by='Distance_KM').head(2)
        
        # 6. Print the results cleanly
        print(f"--- Patient {i+1} Located at ({p_lat:.5f}, {p_lon:.5f}) ---")
        for idx, row in closest_two.iterrows():
            print(f" -> Closest #{closest_two.index.get_loc(idx)+1}: {row['Organization']} ({row['HQ Town / City']}) - {row['Distance_KM']} km away")
        print("-" * 50)



file_path = "Generated Data.csv"

print("=== ORGANIZATION DISTANCE MATRIX (KM) ===")
org_matrix = generate_distance_matrix_organizations(file_path)
print(org_matrix)
print("\n" + "="*50 + "\n")

print("=== NEAREST ORGANIZATIONS FOR 10 RANDOM PATIENTS ===")
find_closest_organizations(file_path, num_patients=10)