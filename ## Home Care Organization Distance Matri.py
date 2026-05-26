import pandas as pd
import numpy as np


def generate_distance_matrix_organizations(file_path, separator=';'):
    """
    Leest een zorgorganisatie CSV in en genereert een afstandmatrix in kilometers.
    """
    # 1. Lees het CSV-bestand in
    df = pd.read_csv(file_path, sep=separator)
    df.columns = df.columns.str.strip()
    
    lon_col = 'Longditude' if 'Longditude' in df.columns else 'Longitude'
    
    # 2. Haal de Wetenschappelijke Graden (lat/lon) op en zet ze om naar radialen
    lat_rad = np.radians(df['Latitude'].values)
    lon_rad = np.radians(df[lon_col].values)
    
    # 3. Bereken het verschil tussen alle paren (Matrix Broadcasting)
    # Dit maakt handig gebruik van dimensies om alle combinaties in één keer te doen
    dlat = lat_rad[:, np.newaxis] - lat_rad
    dlon = lon_rad[:, np.newaxis] - lon_rad
    
    # 4. De Haversine formule
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_rad[:, np.newaxis]) * np.cos(lat_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    km_matrix = 6371 * c               #Straal van de aarde in kilometers is 6371
    km_matrix = np.round(km_matrix, 2) #Afronden op 2 decimalen voor de netheid
    
    # 6. Formatteer naar een overzichtelijke DataFrame
    matrix_df = pd.DataFrame(km_matrix, index=df['Organization'], columns=df['Organization'])
    
    return matrix_df

file_path = "Generated Data.csv"
print(generate_distance_matrix_organizations(file_path))  #print distance matrix