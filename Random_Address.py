import random
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# ----------------------------------------------------------------------
# Define your circles here: (latitude, longitude, radius_km)
# ----------------------------------------------------------------------
circles = [
    (52.512153, 6.092912, 20),  # Zwolle
    (52.2215,   6.8937,   20),  # Enschede
    (52.0907,   5.1214,   20),  # Utrecht
]

# ----------------------------------------------------------------------
# Build a geodesic circle (accurate on Earth's surface)
# ----------------------------------------------------------------------
def geodesic_circle(lat, lon, radius_km, n_points=360):
    points = []

    for bearing in range(n_points):
        destination = geodesic(kilometers=radius_km).destination(
            (lat, lon),
            bearing
        )

        # Shapely expects (longitude, latitude)
        points.append((destination.longitude, destination.latitude))

    return Polygon(points)


# ----------------------------------------------------------------------
# Create the union of all circles
# ----------------------------------------------------------------------
circle_polygons = [
    geodesic_circle(lat, lon, radius)
    for lat, lon, radius in circles
]

search_area = unary_union(circle_polygons)

# Bounding box of the union
min_lon, min_lat, max_lon, max_lat = search_area.bounds

# ----------------------------------------------------------------------
# Initialize geocoder
# ----------------------------------------------------------------------
geolocator = Nominatim(user_agent="random_address")

# ----------------------------------------------------------------------
# Generate a single random address
# ----------------------------------------------------------------------
def generate_random_address(max_attempts=10):
    """
    Generate a random address within the defined geographic area.
    
    Parameters:
    -----------
    max_attempts : int
        Maximum geocoding attempts before returning None
    
    Returns:
    --------
    dict or None
        Dictionary with keys: address, latitude, longitude
        Returns None if geocoding fails after max_attempts
    """
    for attempt in range(max_attempts):
        try:
            # Generate a random point inside the bounding box
            lon = random.uniform(min_lon, max_lon)
            lat = random.uniform(min_lat, max_lat)

            point = Point(lon, lat)

            # Reject if outside the union of circles
            if not search_area.contains(point):
                continue

            # Reverse geocode the point
            location = geolocator.reverse(
                (lat, lon),
                exactly_one=True,
                language="en",
                timeout=10
            )

            if location and location.address:
                return {
                    'address': location.address,
                    'latitude': location.latitude,
                    'longitude': location.longitude
                }

        except Exception as e:
            continue

    # Fallback: return None if all attempts fail
    return None


# ----------------------------------------------------------------------
# Demo: Generate and display a random address
# ----------------------------------------------------------------------
if __name__ == "__main__":
    address_data = generate_random_address()
    
    if address_data:
        print("Address:")
        print(address_data['address'])

        print("\nCoordinates:")
        print(f"Latitude:  {address_data['latitude']}")
        print(f"Longitude: {address_data['longitude']}")

        print("\nGoogle Maps:")
        print(
            f"https://www.google.com/maps?q="
            f"{address_data['latitude']},{address_data['longitude']}"
        )
    else:
        print("Failed to generate address")