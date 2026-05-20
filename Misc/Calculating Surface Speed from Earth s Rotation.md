## Calculating Surface Speed from Earth's Rotation:
Slide 1: Earth's Rotation and Surface Speed

This presentation explores the problem of calculating the estimated speed at various latitudes due to Earth's rotation. We'll develop a mathematical approach to solve this problem, considering the Earth's shape and rotation rate.

Slide 2: Background: Earth's Rotation

Earth rotates on its axis once every 24 hours, causing the apparent movement of celestial bodies across the sky. This rotation also affects the speed at which points on the Earth's surface move through space, with the speed varying depending on latitude.

Slide 3: Problem Assumptions

To simplify our calculations, we'll make the following assumptions:

1. Earth is a perfect sphere
2. Earth's radius is approximately 6,371 km
3. Earth rotates exactly once every 24 hours
4. We'll ignore the effects of Earth's orbital motion around the Sun

Slide 4: Mathematical Formulation

To calculate the surface speed at a given latitude, we need to consider the circumference of the circle formed by that latitude line. The formula for surface speed is:

Surface Speed = (2 \* π \* r \* cos(θ)) / T

Where: r = Earth's radius θ = latitude angle in radians T = Earth's rotation period (24 hours)

Slide 5: Logical Reasoning and Pseudocode

To solve this problem, we'll follow these steps:

1. Define constants (Earth's radius, rotation period)
2. Convert latitude from degrees to radians
3. Calculate the circumference of the latitude circle
4. Divide the circumference by the rotation period to get speed
5. Convert speed from km/h to mph

Pseudocode:

```
function calculate_surface_speed(latitude_degrees):
    earth_radius = 6371  # km
    rotation_period = 24  # hours
    
    latitude_radians = convert_to_radians(latitude_degrees)
    circumference = 2 * pi * earth_radius * cos(latitude_radians)
    speed_km_h = circumference / rotation_period
    speed_mph = convert_km_h_to_mph(speed_km_h)
    
    return speed_mph
```

Slide 6: Python Implementation (Part 1)

```python
import math

def calculate_surface_speed(latitude_degrees):
    # Constants
    EARTH_RADIUS_KM = 6371
    ROTATION_PERIOD_HOURS = 24
    KM_TO_MILES = 0.621371

    # Convert latitude to radians
    latitude_radians = math.radians(latitude_degrees)

    # Calculate circumference of latitude circle
    circumference_km = 2 * math.pi * EARTH_RADIUS_KM * math.cos(latitude_radians)

    # Calculate speed in km/h
    speed_km_h = circumference_km / ROTATION_PERIOD_HOURS

    # Convert speed to mph
    speed_mph = speed_km_h * KM_TO_MILES

    return speed_mph
```

Slide 7: Python Implementation (Part 2)

```python
# Test the function with various latitudes
latitudes = [0, 15, 30, 45, 60, 75, 90]

print("Latitude | Surface Speed (mph)")
print("---------|--------------------")
for lat in latitudes:
    speed = calculate_surface_speed(lat)
    print(f"{lat:8.0f} | {speed:18.2f}")

# Calculate speed at the equator for comparison
equator_speed = calculate_surface_speed(0)
print(f"\nSpeed at the equator: {equator_speed:.2f} mph")
```

Slide 8: Real-World Applications

1. Satellite orbit calculations: Understanding Earth's rotation is crucial for precise satellite positioning and tracking.
2. Global Positioning System (GPS): Accurate surface speed calculations improve GPS accuracy.
3. Weather forecasting: Earth's rotation affects wind patterns and ocean currents.
4. Ballistic missile trajectories: Surface speed affects launch calculations for long-range projectiles.
5. Geophysics research: Studying Earth's rotation helps understand plate tectonics and core dynamics.

Slide 9: Historical Trivia: Foucault Pendulum

In 1851, French physicist Léon Foucault demonstrated Earth's rotation using a pendulum. The plane of the pendulum's swing appeared to rotate over time, providing visual proof of Earth's rotation. This experiment can be used to estimate latitude based on the pendulum's rotation rate.

Slide 10: Made-up Trivia Question

If Earth suddenly stopped rotating, how long would it take for a person at the equator to reach the North Pole, assuming they could walk at a constant speed of 5 km/h?

Slide 11: Solving the Made-up Trivia Question

To solve this, we need to calculate the distance from the equator to the North Pole along Earth's surface:

```python
import math

EARTH_RADIUS_KM = 6371
WALKING_SPEED_KM_H = 5

# Calculate distance from equator to North Pole
distance_km = (math.pi / 2) * EARTH_RADIUS_KM

# Calculate time to walk this distance
time_hours = distance_km / WALKING_SPEED_KM_H

# Convert time to days
time_days = time_hours / 24

print(f"Distance to walk: {distance_km:.2f} km")
print(f"Time to reach North Pole: {time_days:.2f} days")
```

Slide 12: Additional Resources

1. "The Physics of Earth's Rotation" - [https://arxiv.org/abs/1204.4449](https://arxiv.org/abs/1204.4449)
2. "Earth Rotation: Physical Models and Theoretical Aspects" - [https://arxiv.org/abs/1702.03062](https://arxiv.org/abs/1702.03062)
3. "Variations in the rotation of the Earth" - [https://royalsocietypublishing.org/doi/10.1098/rsta.2011.0032](https://royalsocietypublishing.org/doi/10.1098/rsta.2011.0032)
4. NASA Earth Observatory: Earth's Rotation - [https://earthobservatory.nasa.gov/features/EON/eon3.php](https://earthobservatory.nasa.gov/features/EON/eon3.php)
5. NOAA Earth System Research Laboratory: Earth's Rotation - [https://www.esrl.noaa.gov/gmd/outreach/info\_activities/pdfs/TBI\_earth\_rotation.pdf](https://www.esrl.noaa.gov/gmd/outreach/info_activities/pdfs/TBI_earth_rotation.pdf)

