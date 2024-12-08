import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 500
initial_inventory = 100
num_simulations = 1000
prices = [200, 100, 50]
probabilities = [0.1, 0.5, 0.8]

# Initialize
V = np.zeros((T + 1, initial_inventory + 1))
policy = np.zeros((T, initial_inventory + 1), dtype=int)

# DP
for t in range(T - 1, -1, -1):
    for x in range(initial_inventory + 1):
        max_value = 0
        best_price_index = 0
        for i, price in enumerate(prices):
            prob = probabilities[i]
            value_if_sold = prob * (price + V[t + 1][x - 1]) if x > 0 else 0
            value_if_not_sold = (1 - prob) * V[t + 1][x]
            expected_value = value_if_sold + value_if_not_sold
            if expected_value > max_value:
                max_value = expected_value
                best_price_index = i

        V[t][x] = max_value  # Best revenue for this state
        policy[t][x] = best_price_index  # Store optimal price choice

total_revenues = []

for sim in range(num_simulations):
    inventory = initial_inventory
    total_revenue = 0
    for t in range(T):
        if inventory == 0:
            break

        price_index = policy[t][inventory]
        price = prices[price_index]
        prob = probabilities[price_index]

        if np.random.rand() < prob:
            total_revenue += price
            inventory -= 1  # Reduce inventory

    total_revenues.append(total_revenue)


plt.hist(total_revenues, bins=30, edgecolor='black')
plt.title("distribution_of_total_revenue_1000")
plt.xlabel("Total")
plt.ylabel("Occurrence")
plt.show()
