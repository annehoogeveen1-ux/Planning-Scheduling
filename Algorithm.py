from geopy.distance import geodesic
from Patient_Generator import generate_patient
from config import home_centres

# -----------------------------
# Allocation function
# -----------------------------
def allocate_patient():

    patient = generate_patient()

    best_centre = None
    best_distance = float("inf")

    for centre in home_centres:
        dist = geodesic(
            (patient["latitude"], patient["longitude"]),
            (centre["lat"], centre["lon"])
        ).km

        if dist < best_distance:
            best_distance = dist
            best_centre = centre

    return {
        **patient,
        "assigned_centre": best_centre["id"],
        "distance_km": best_distance
    }

# -----------------------------
# Batch simulation (optional)
# -----------------------------
def run_simulation(n=100):
    results = []

    for _ in range(n):
        results.append(allocate_patient())

    return results


# -----------------------------
# Test run
# -----------------------------
if __name__ == "__main__":
    print(allocate_patient())