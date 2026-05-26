import pandas as pd
from scipy.spatial import distance_matrix

# 1. Gebruik het volledige absolute pad naar je CSV-bestand
file_path = "Generated Data.csv"

def generate_distance_matrix(file_path):
    df = pd.read_csv(file_path, sep=';')

    # Schoon de kolomnamen op (verwijdert eventuele onzichtbare spaties aan het begin/eind)
    df.columns = df.columns.str.strip()

    # 2. Dynamisch de juiste lengtegraad-kolom pakken, hoe hij ook gespeld staat
    lon_col = 'Longditude' if 'Longditude' in df.columns else 'Longitude'

    # Bereken de afstandmatrix
    coords = df[['Latitude', lon_col]].values
    dist_matrix = distance_matrix(coords, coords)

    # 3. Zet de matrix om in een overzichtelijke DataFrame met de organisatienamen
    matrix_df = pd.DataFrame(dist_matrix, index=df['Organization'], columns=df['Organization'])

    # Toon het resultaat in je terminal
    return matrix_df
print(generate_distance_matrix(file_path))