## Analyzing the $21 Million Lifetime Airline Ticket
Slide 1: The Lifetime Ticket Conundrum

In 1987, Steve Rothstein purchased a lifetime American Airlines ticket for $250,000. This ticket allowed him unlimited travel, which he used extensively, even flying to other countries just for lunch. The airline company claims this ended up costing them $21 million. Our task is to estimate how many flights Rothstein took to accumulate such a significant cost.

Slide 2: Historical Context

American Airlines introduced the AAirpass program in 1981, offering unlimited first-class travel for a one-time fee. The program was designed to generate quick cash for the airline during a period of financial difficulty. However, the long-term consequences of this program were not fully anticipated, leading to substantial losses for the company.

Slide 3: Assumptions and Simplifications

To approach this problem, we'll make the following assumptions:

1. The average cost of a flight to American Airlines is constant over time.
2. We'll use an average flight cost, considering both domestic and international flights.
3. We'll ignore inflation and changes in fuel prices over the years.
4. We'll assume Rothstein's travel patterns remained consistent throughout the period.

Slide 5: Problem Breakdown and Mathematical Formulation

Let's break down the problem into these components:

1. Total cost to American Airlines: $21,000,000
2. Purchase price of the ticket: $250,000
3. Net loss to American Airlines: $21,000,000 - $250,000 = $20,750,000
4. Unknown variable: Number of flights taken (x)
5. Unknown variable: Average cost per flight to American Airlines (y)

Our equation: 20,750,000 = x \* y

Slide 6: Logical Reasoning and Estimation

To estimate the average cost per flight:

1. Consider a mix of domestic and international flights.
2. Factor in first-class amenities and services.
3. Account for operational costs like fuel, staff, and maintenance.

Let's estimate the average cost per flight to be $500.

Now we can solve for x: 20,750,000 = x \* 500 x = 20,750,000 / 500 x ≈ 41,500 flights

Slide 7: Python Code - Part 1

```python
# Constants
TOTAL_COST = 21_000_000
TICKET_PRICE = 250_000
NET_LOSS = TOTAL_COST - TICKET_PRICE

# Estimated average cost per flight
AVG_COST_PER_FLIGHT = 500

# Calculate estimated number of flights
estimated_flights = NET_LOSS / AVG_COST_PER_FLIGHT

print(f"Estimated number of flights: {estimated_flights:.0f}")
```

Slide 8: Python Code - Part 2

```python
# Function to calculate flights based on varying average costs
def calculate_flights(avg_cost_range):
    results = {}
    for avg_cost in avg_cost_range:
        flights = NET_LOSS / avg_cost
        results[avg_cost] = round(flights)
    return results

# Calculate for a range of average costs
avg_costs = range(300, 701, 50)
flight_estimates = calculate_flights(avg_costs)

for cost, flights in flight_estimates.items():
    print(f"At ${cost} per flight: {flights} flights")
```

Slide 9: Real-World Applications

This estimation technique has various applications:

1. Business: Analyzing customer lifetime value and loyalty program profitability.
2. Transportation: Optimizing route networks and pricing strategies for airlines and other transport companies.
3. Economics: Studying the long-term effects of promotional offers and their impact on company finances.
4. Risk Management: Assessing potential losses from unlimited use products or services.

Slide 10: The AAirpass Legacy

The AAirpass program's unintended consequences led to significant changes in how airlines approach loyalty programs. Today, most airline miles programs have more restrictions and expiration dates to prevent similar situations. This case study is often used in business schools to illustrate the importance of long-term planning and risk assessment in promotional strategies.

Slide 11: Sensitivity Analysis

Let's examine how our estimate changes with different average flight costs:

1. At $300 per flight: ~69,167 flights
2. At $400 per flight: ~51,875 flights
3. At $500 per flight: ~41,500 flights
4. At $600 per flight: ~34,583 flights
5. At $700 per flight: ~29,643 flights

This analysis shows the importance of accurate cost estimation in our calculations.

Slide 12: Made-up Trivia: The Cosmic Commuter

Imagine an interplanetary travel company offers unlimited travel between Earth and Mars for $10 billion. If each round trip costs the company $50 million and takes 2 years, how many years would a passenger need to travel to cost the company $100 billion?

Slide 13: Solving the Cosmic Commuter Problem

```python
TICKET_PRICE = 10_000_000_000
TOTAL_COST = 100_000_000_000
NET_LOSS = TOTAL_COST - TICKET_PRICE
COST_PER_TRIP = 50_000_000
YEARS_PER_TRIP = 2

trips = NET_LOSS / COST_PER_TRIP
years = trips * YEARS_PER_TRIP

print(f"Number of trips: {trips:.0f}")
print(f"Number of years: {years:.0f}")
```

This cosmic commuter would need to make 1,800 trips over 3,600 years to cost the company $100 billion!

Slide 14: Further Reading

1. "The Man Who Took Unlimited Flights on American Airlines for 21 Years" - [https://www.theguardian.com/lifeandstyle/2019/sep/19/american-airlines-aairpass-golden-ticket](https://www.theguardian.com/lifeandstyle/2019/sep/19/american-airlines-aairpass-golden-ticket)
2. "Loyalty Programs: Design and Effectiveness" - [https://arxiv.org/abs/1810.03481](https://arxiv.org/abs/1810.03481)
3. "The Economics of Airline Frequent Flyer Programs" - [https://www.nber.org/papers/w28087](https://www.nber.org/papers/w28087)

