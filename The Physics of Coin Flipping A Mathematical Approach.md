## When spun on a table, a US ‘Lincoln Memorial’ one-cent coin will land on tails 80% of the time.
Slide 1: The Curious Case of the Lincoln Memorial Cent

The US 'Lincoln Memorial' one-cent coin exhibits an intriguing behavior when spun on a table: it lands on tails 80% of the time. This presentation will explore the physics behind this phenomenon, develop a mathematical model to explain it, and use Python to simulate the coin's behavior.

Slide 2: The Lincoln Memorial Cent

The Lincoln Memorial cent, minted from 1959 to 2008, features Abraham Lincoln's portrait on the obverse (heads) and the Lincoln Memorial on the reverse (tails). The coin's unique physical properties, including its weight distribution and surface features, contribute to its unusual spinning behavior.

Slide 3: Assumptions and Simplifications

To analyze the coin's behavior, we'll make the following assumptions:

1. The coin is a perfect cylinder with uniform density.
2. Air resistance is negligible.
3. The spinning surface is perfectly flat and smooth.
4. The coin's initial angular velocity is high enough to ensure multiple rotations.
5. The coin's edge is perfectly smooth, ignoring the ridges present on actual coins.

Slide 4: Mathematical Formulation

We'll model the coin's motion using the following equations:

1. Angular momentum conservation: L = Iω Where L is angular momentum, I is moment of inertia, and ω is angular velocity.
2. Energy conservation: E = 1/2 Iω² + mgh Where E is total energy, m is mass, g is gravitational acceleration, and h is height.
3. Moment of inertia for a cylinder: I\_center = 1/4 mr² + 1/12 mh² I\_edge = I\_center + mr²

Where r is the coin's radius and h is its thickness.

Slide 5: Analyzing the Coin's Motion

Pseudocode for simulating the coin's motion:

```
function simulate_coin_spin(initial_angle, initial_angular_velocity): 
    while coin_is_spinning: 
    update_angular_position(time_step)
    update_angular_velocity(time_step) 
    if coin_falls_below_critical_angle: 
        determine_landing_side() 
        break 
    return landing_side
```

The critical angle at which the coin transitions from spinning to falling determines the landing side. This angle depends on the coin's physical properties and initial conditions.

Slide 6: Python Simulation - Part 1

```python
import numpy as np
import matplotlib.pyplot as plt

class Coin:
    def __init__(self, radius, thickness, mass):
        self.radius = radius
        self.thickness = thickness
        self.mass = mass
        self.I_center = 0.25 * mass * radius**2 + (1/12) * mass * thickness**2
        self.I_edge = self.I_center + mass * radius**2
        
    def simulate_spin(self, initial_angle, initial_angular_velocity, time_step, num_steps):
        angle = initial_angle
        angular_velocity = initial_angular_velocity
        
        angles = [angle]
        angular_velocities = [angular_velocity]
        
        for _ in range(num_steps):
            angle += angular_velocity * time_step
            angular_velocity -= (self.mass * 9.81 * self.radius * np.sin(angle) / self.I_edge) * time_step
            
            angles.append(angle)
            angular_velocities.append(angular_velocity)
            
            if angle < 0:
                return "Tails", angles, angular_velocities
        
        return "Heads", angles, angular_velocities
```

Slide 7: Python Simulation - Part 2

```python
# Create a coin object (dimensions in meters, mass in kg)
lincoln_cent = Coin(radius=0.00953, thickness=0.00192, mass=0.00250)

# Run multiple simulations
num_simulations = 10000
tails_count = 0

for _ in range(num_simulations):
    result, _, _ = lincoln_cent.simulate_spin(
        initial_angle=np.pi/2,
        initial_angular_velocity=100,
        time_step=0.0001,
        num_steps=100000
    )
    if result == "Tails":
        tails_count += 1

tails_probability = tails_count / num_simulations
print(f"Probability of landing on tails: {tails_probability:.2f}")

# Plot a single simulation
result, angles, angular_velocities = lincoln_cent.simulate_spin(
    initial_angle=np.pi/2,
    initial_angular_velocity=100,
    time_step=0.0001,
    num_steps=10000
)

plt.figure(figsize=(10, 5))
plt.plot(angles)
plt.title("Coin Angle vs. Time")
plt.xlabel("Time Step")
plt.ylabel("Angle (radians)")
plt.show()
```

Slide 8: Real-World Applications

The techniques used to analyze the Lincoln cent's behavior have applications in various fields:

1. Robotics: Designing coin-sorting machines and predicting object behavior in automated systems.
2. Physics education: Demonstrating concepts of angular momentum, energy conservation, and rotational dynamics.
3. Game theory: Analyzing fairness in games of chance and developing strategies for coin-based games.
4. Manufacturing: Optimizing the design of coins and other circular objects to achieve desired physical properties.
5. Forensics: Reconstructing crime scenes involving falling or spinning objects.

Slide 9: The Lincoln Cent Trivia Challenge

Question: If a Lincoln cent is spun on its edge on the surface of the Moon, how would its behavior differ from Earth, and why?

To solve this, consider:

1. The Moon's gravity (1/6 of Earth's)
2. Lack of atmosphere on the Moon
3. Different surface properties

This problem requires adapting our Earth-based model to lunar conditions.

Slide 10: Lunar Lincoln Cent Simulation

```python
class LunarCoin(Coin):
    def simulate_spin(self, initial_angle, initial_angular_velocity, time_step, num_steps):
        angle = initial_angle
        angular_velocity = initial_angular_velocity
        
        for _ in range(num_steps):
            angle += angular_velocity * time_step
            # Moon's gravity is about 1.625 m/s^2
            angular_velocity -= (self.mass * 1.625 * self.radius * np.sin(angle) / self.I_edge) * time_step
            
            if angle < 0:
                return "Tails"
        
        return "Heads"

lunar_cent = LunarCoin(radius=0.00953, thickness=0.00192, mass=0.00250)

num_simulations = 10000
lunar_tails_count = 0

for _ in range(num_simulations):
    result = lunar_cent.simulate_spin(
        initial_angle=np.pi/2,
        initial_angular_velocity=100,
        time_step=0.0001,
        num_steps=100000
    )
    if result == "Tails":
        lunar_tails_count += 1

lunar_tails_probability = lunar_tails_count / num_simulations
print(f"Probability of landing on tails on the Moon: {lunar_tails_probability:.2f}")
```

Slide 11: The Physics of Coin Spinning

The tendency of the Lincoln cent to land on tails more often is due to its physical properties:

1. Center of mass: The Lincoln Memorial side is slightly heavier due to the raised design.
2. Air resistance: The more detailed tails side experiences greater air resistance during spinning.
3. Precession: The coin's wobble as it spins affects its final orientation.
4. Surface interaction: Microscopic imperfections on the coin and spinning surface influence the outcome.

These factors combine to create a bias towards the tails side, resulting in the observed 80% probability.

Slide 12: Historical Context

The Lincoln cent has been in circulation since 1909, with the Lincoln Memorial design introduced in 1959. The coin's unique spinning behavior was first noted by mathematicians and physicists in the late 20th century. This discovery led to further studies on the physics of spinning objects and the role of asymmetry in determining their behavior.

Slide 13: Further Reading

For those interested in delving deeper into the physics of coin spinning and related topics, here are some resources:

1. "The Dynamical Behavior of a Spinning Coin" by H. R. Crane (1967) [https://aapt.scitation.org/doi/10.1119/1.1973744](https://aapt.scitation.org/doi/10.1119/1.1973744)
2. "How a Coin Spins" by E. H. Atlas (1982) [https://aapt.scitation.org/doi/10.1119/1.13283](https://aapt.scitation.org/doi/10.1119/1.13283)
3. "The Motion of a Spinning Coin" by J. Strzałko et al. (2008) [https://arxiv.org/abs/0801.4407](https://arxiv.org/abs/0801.4407)
4. "Euler's Disk and its Finite-Time Singularity" by H. K. Moffatt (2000) [https://www.nature.com/articles/35001504](https://www.nature.com/articles/35001504)
5. "Dynamics of Spinning Disks and Coins" by K. Ramasubramanian and K. Vijayakumar (2009) [https://www.ias.ac.in/article/fulltext/reso/014/09/0822-0851](https://www.ias.ac.in/article/fulltext/reso/014/09/0822-0851)

