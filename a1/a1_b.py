import numpy as np
import matplotlib.pyplot as plt

# Params
T = 500
initial_inventory = 100
prices = [200, 100, 50]
probabilities = [0.1, 0.5, 0.8]  # Probability

# Initialize
V = np.zeros((T + 1, initial_inventory + 1))
policy = np.zeros((T, initial_inventory + 1), dtype=int)

# DP
for t in range(T - 1, -1, -1):  # backwards from t-1 to 0
    for x in range(initial_inventory + 1):
        max_value = 0
        best_price_index = 0
        for i, price in enumerate(prices):
            prob = probabilities[i]
            #item is sold
            value_if_sold = prob * (price + V[t + 1][x - 1]) if x > 0 else 0
            #if item is not sold
            value_if_not_sold = (1 - prob) * V[t + 1][x]
            # Total expected revenue
            expected_value = value_if_sold + value_if_not_sold
            if expected_value > max_value:
                max_value = expected_value
                best_price_index = i  # Store the best price index

        V[t][x] = max_value  # Store value for state (t, x)
        policy[t][x] = best_price_index  # Store p_optimal

# Expected maximal revenue
expected_maximal_revenue = V[0][initial_inventory]
print("Expected Maximal Revenue:", expected_maximal_revenue)

optimal_policy = np.zeros((T, initial_inventory + 1))
for t in range(T):
    for x in range(initial_inventory + 1):
        optimal_price_index = policy[t][x]
        optimal_policy[t, x] = prices[optimal_price_index]

# Plot
plt.figure(figsize=(12, 8))
plt.imshow(optimal_policy, aspect='auto', origin='lower', cmap='viridis',
           extent=[0, T, 0, initial_inventory])
plt.colorbar(label="Optimal Price")
plt.xlabel("Time Period")
plt.ylabel("Remaining Inventory")
plt.title("Optimal Policy (Price)")
plt.show()
