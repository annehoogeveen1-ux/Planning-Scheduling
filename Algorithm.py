from Patient_Generator import generate_patient
from Config import home_centres

home_centres = [
    {"id": 1, "lat": 52.27818005978114, "lon": 5.972706090677909},
    {"id": 2, "lat": 52.52965432815976, "lon": 5.917148356828077},
    {"id": 3, "lat": 52.171816247239505, "lon": 5.744777726409898}
]

def allocate_patient():

    patient = generate_patient()

    best_centre = None
    best_distance = float("inf")

    for c in home_centres:
        dist = geodesic(
            (patient["latitude"], patient["longitude"]),
            (c["lat"], c["lon"])
        ).km

        if dist < best_distance:
            best_distance = dist
            best_centre = c

    return {
        **patient,
        "assigned_centre": best_centre["id"],
        "distance_km": best_distance
    }