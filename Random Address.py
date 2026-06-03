import random
import math
import geopy
from geopy.geocoders import Nominatim

# Center point (Binnenstad Zwolle)
center_lat = 52.512153
center_lon = 6.092912

# Radius in kilometers
radius_km = 20

geolocator = Nominatim(user_agent="random_address")

# Earth's radius (km)
earth_radius = 6371

while True:
    # Random distance and angle
    distance = radius_km * math.sqrt(random.random())
    angle = random.uniform(0, 2 * math.pi)

    # Convert distance to latitude/longitude offsets
    delta_lat = (distance / earth_radius) * (180 / math.pi)
    delta_lon = (
        (distance / earth_radius)
        * (180 / math.pi)
        / math.cos(math.radians(center_lat))
    )

    lat = center_lat + delta_lat * math.sin(angle)
    lon = center_lon + delta_lon * math.cos(angle)

    # Find nearest address
    location = geolocator.reverse((lat, lon), exactly_one=True)

    if location and location.address:
        print("Address:")
        print(location.address)

        print("\nCoordinates:")
        print(f"Latitude: {location.latitude}")
        print(f"Longitude: {location.longitude}")

        print("\nGoogle Maps:")
        print(
            f"https://www.google.com/maps?q="
            f"{location.latitude},{location.longitude}"
        )

        break