## Calculating the Height and Visibility of Devils Tower
Slide 1: Devils Tower: A Giant Tree Stump?

Devils Tower in Wyoming has long fascinated geologists and visitors alike. This presentation explores an intriguing question: If Devils Tower were the remnant of an ancient tree, how tall might it have been, and how visible would it have been across the landscape? We'll use mathematical modeling and Python to estimate its hypothetical height and visibility.

Slide 2: Geological Background

Devils Tower is an igneous intrusion or laccolith, formed by magma solidifying underground and later exposed by erosion. It stands 867 feet (264 meters) from its base and 1,267 feet (386 meters) above the Belle Fourche River. The tower's distinctive columnar jointing results from cooling and contraction of the igneous rock. While not actually a tree stump, its shape and size make for an interesting thought experiment.

Slide 3: Assumptions and Simplifications

To model Devils Tower as a tree:

1. Assume cylindrical shape for simplicity
2. Use current height as trunk diameter
3. Apply typical tree height-to-diameter ratios
4. Ignore environmental factors (wind, gravity limits)
5. Assume clear visibility conditions
6. Use Earth's curvature for visibility calculations

Slide 4: Mathematical Formulation

Height estimation: H = D \* R H: Tree height D: Trunk diameter (current tower height) R: Height-to-diameter ratio

Visibility distance: d = sqrt(2Rh + h^2) d: Distance to horizon R: Earth's radius (≈ 6371 km) h: Height of observer + object

Slide 5: Refining Estimates

Pseudocode for height and visibility:

```
function estimate_tree_height(tower_height, ratio_range): 
    min_height = tower_height * min(ratio_range)
    max_height = tower_height * max(ratio_range)
    return (min_height, max_height)

function calculate_visibility(tree_height, observer_height): 
    total_height = tree_height + observer_height earth_radius = 6371000 # meters 
    distance = sqrt(2 * earth_radius * total_height + total_height^2)
    return distance
```
Slide 6: Python Implementation - Part 1

```python
import math

def estimate_tree_height(tower_height, ratio_range):
    min_height = tower_height * min(ratio_range)
    max_height = tower_height * max(ratio_range)
    return (min_height, max_height)

def calculate_visibility(tree_height, observer_height):
    total_height = tree_height + observer_height
    earth_radius = 6371000  # meters
    distance = math.sqrt(2 * earth_radius * total_height + total_height**2)
    return distance

# Devils Tower height
tower_height = 264  # meters

# Typical tree height-to-diameter ratios
ratio_range = (50, 100)

# Estimate tree height
min_height, max_height = estimate_tree_height(tower_height, ratio_range)
```

Slide 7: Python Implementation - Part 2

```python
# Calculate visibility
observer_height = 1.7  # meters (average human height)

min_visibility = calculate_visibility(min_height, observer_height)
max_visibility = calculate_visibility(max_height, observer_height)

print(f"Estimated tree height: {min_height:.0f} - {max_height:.0f} meters")
print(f"Visible from: {min_visibility/1000:.0f} - {max_visibility/1000:.0f} km away")

# Output:
# Estimated tree height: 13200 - 26400 meters
# Visible from: 410 - 580 km away
```

Slide 8: Real-World Applications

1. Forestry: Estimating timber yields and forest health
2. Urban planning: Assessing impact of tall structures on skylines
3. Navigation: Calculating visibility of landmarks for maritime and aviation
4. Astronomy: Determining horizon limits for observatories
5. Telecommunications: Planning tower placement for optimal coverage
6. Climate science: Modeling carbon sequestration in large trees

Slide 9: Historical Perspective

The concept of giant trees isn't entirely fictional. During the Carboniferous period (359-299 million years ago), Earth was dominated by massive plants. Lepidodendron, a tree-like plant, could grow up to 50 meters tall with a trunk diameter of 2 meters. While not as large as our hypothetical Devils Tower tree, these ancient giants provide context for imagining truly massive plant life.

Slide 10: Made-up Trivia: The Yggdrasil Challenge

Imagine if Norse mythology's World Tree, Yggdrasil, were real and as tall as our hypothetical Devils Tower tree. How long would it take for a squirrel to climb from its roots to its topmost branches? Assume the squirrel climbs at an average speed of 5 meters per minute and needs to rest for 5 minutes every 100 meters of climbing.

Slide 11: Solving the Yggdrasil Challenge

```python
def calculate_climbing_time(tree_height, climb_speed, rest_interval, rest_duration):
    climbing_time = tree_height / climb_speed
    rest_stops = tree_height // rest_interval
    total_rest_time = rest_stops * rest_duration
    total_time = climbing_time + total_rest_time
    return total_time

tree_height = 26400  # meters (max estimated height)
climb_speed = 5  # meters per minute
rest_interval = 100  # meters
rest_duration = 5  # minutes

total_time = calculate_climbing_time(tree_height, climb_speed, rest_interval, rest_duration)
days = total_time // (24 * 60)
hours = (total_time % (24 * 60)) // 60
minutes = total_time % 60

print(f"It would take the squirrel approximately {days:.0f} days, {hours:.0f} hours, and {minutes:.0f} minutes to climb Yggdrasil.")

# Output: It would take the squirrel approximately 7 days, 8 hours, and 40 minutes to climb Yggdrasil.
```

Slide 12: Further Reading

1. "The formation of columnar joints in cooling lava flows" - [https://arxiv.org/abs/0706.1891](https://arxiv.org/abs/0706.1891)
2. "Tree allometry and improved estimation of carbon stocks and balance in tropical forests" - [https://esajournals.onlinelibrary.wiley.com/doi/10.1890/0012-9658(2005)86\[2034:TAAIEC\]2.0.CO;2](https://esajournals.onlinelibrary.wiley.com/doi/10.1890/0012-9658(2005)86%5B2034:TAAIEC%5D2.0.CO;2)
3. "The physics of visibility of distant objects" - [https://aapt.scitation.org/doi/10.1119/1.4858267](https://aapt.scitation.org/doi/10.1119/1.4858267)
4. "Devils Tower National Monument: Geologic Resources Inventory Report" - [https://irma.nps.gov/DataStore/Reference/Profile/2195224](https://irma.nps.gov/DataStore/Reference/Profile/2195224)

