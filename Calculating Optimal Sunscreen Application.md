## Calculating Optimal Sunscreen Application:
Slide 1: The Sunscreen Dilemma

A Gen Z beachgoer plans to spend 10 hours at the beach, applying sunscreen with SPF 30 every 2 hours. We'll determine how many times they'll apply sunscreen and calculate their total SPF protection, assuming SPF values don't accumulate.

Slide 2: Understanding SPF

Sun Protection Factor (SPF) measures how well a sunscreen protects against UVB rays. SPF 30 blocks about 97% of UVB rays, allowing you to stay in the sun 30 times longer than without protection before burning.

Slide 3: Problem Assumptions

1. The beachgoer applies sunscreen immediately upon arrival
2. They reapply every 2 hours consistently
3. SPF effectiveness remains constant between applications
4. The 10-hour period is continuous
5. Environmental factors (water, sweat) don't affect reapplication timing

Slide 4: Mathematical Formulation

Let's define our variables: T = Total time at the beach (in hours) = 10 I = Reapplication interval (in hours) = 2 S = SPF value = 30

Number of applications (N) = Floor(T / I) + 1 Total SPF protection = S \* N

Slide 5: Logical Reasoning

To solve this problem, we'll use the following steps:

1. Calculate the number of 2-hour intervals in 10 hours
2. Add 1 to account for the initial application
3. Multiply the number of applications by the SPF value

Pseudocode:

```
total_time = 10
interval = 2
spf = 30

applications = floor(total_time / interval) + 1
total_protection = applications * spf

return applications, total_protection
```

Slide 6: Python Implementation (Part 1)

```python
import math

def calculate_sunscreen_applications(total_time, interval, spf):
    # Calculate the number of applications
    applications = math.floor(total_time / interval) + 1
    
    # Calculate total SPF protection
    total_protection = applications * spf
    
    return applications, total_protection
```

Slide 7: Python Implementation (Part 2)

```python
# Set parameters
total_time = 10  # hours
interval = 2     # hours
spf = 30

# Call the function
applications, total_protection = calculate_sunscreen_applications(total_time, interval, spf)

# Print results
print(f"Number of applications: {applications}")
print(f"Total SPF protection: {total_protection}")
```

Slide 8: Real-world Applications

This problem-solving approach can be applied to:

1. Medication dosage scheduling
2. Industrial maintenance planning
3. Crop protection in agriculture
4. Time management in project planning
5. Battery life optimization in mobile devices

Slide 9: Sunscreen History

The first sunscreen was developed in 1938 by Franz Greiter, a Swiss chemistry student. It had an SPF of 2. The SPF rating system was introduced in 1962. Modern sunscreens now offer SPF values up to 100+.

Slide 10: UV Index and Sunscreen Effectiveness

The UV Index, developed by Canadian scientists in 1992, measures the strength of ultraviolet radiation. Understanding the UV Index can help determine how often to reapply sunscreen:

UV Index 0-2: Low exposure, reapply every 2-3 hours UV Index 3-5: Moderate exposure, reapply every 1-2 hours UV Index 6+: High exposure, reapply every 60-90 minutes

Slide 11: Trivia Question

How many bottles of SPF 30 sunscreen would our Gen Z beachgoer need to use in a year if they spent every weekend day at the beach for a full year?

Assumptions:

* 52 weekends per year (104 days)
* 10 hours per beach day
* 1 oz of sunscreen per application
* 8 oz bottle of sunscreen

Let's solve this with Python!

Slide 12: Trivia Solution Code

```python
import math

def calculate_yearly_sunscreen_bottles(beach_days, hours_per_day, interval, oz_per_application, oz_per_bottle):
    applications_per_day = math.floor(hours_per_day / interval) + 1
    total_applications = applications_per_day * beach_days
    total_oz_needed = total_applications * oz_per_application
    bottles_needed = math.ceil(total_oz_needed / oz_per_bottle)
    return bottles_needed

beach_days = 104  # 52 weekends * 2 days
hours_per_day = 10
interval = 2
oz_per_application = 1
oz_per_bottle = 8

bottles = calculate_yearly_sunscreen_bottles(beach_days, hours_per_day, interval, oz_per_application, oz_per_bottle)
print(f"Bottles of sunscreen needed for a year: {bottles}")
```

Slide 13: Additional Resources

1. "Sunscreens: an overview and update" - [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3543289/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3543289/)
2. "The UV Index: A Guide to Sun Safety" - [https://www.epa.gov/sunsafety/uv-index-scale-0](https://www.epa.gov/sunsafety/uv-index-scale-0)
3. "History of Sunscreen" - [https://www.skincancer.org/blog/sunscreen-history/](https://www.skincancer.org/blog/sunscreen-history/)
4. "Ultraviolet radiation as a hazard in the workplace" - [https://www.who.int/uv/publications/en/occupational\_risk.pdf](https://www.who.int/uv/publications/en/occupational_risk.pdf)

