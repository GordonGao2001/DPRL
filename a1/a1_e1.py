import numpy as np

# Params
T = 500
initial_inventory = 100
prices = [200, 100, 50]
probabilities = [0.1, 0.5, 0.8]

V = np.zeros((T + 1, initial_inventory + 1, len(prices)))
policy = np.zeros((T, initial_inventory + 1, len(prices)), dtype=int)

for t in range(T - 1, -1, -1):
    for x in range(initial_inventory + 1):
        for p_prev_idx, p_prev in enumerate(prices):
            max_value = 0
            best_price_index = p_prev_idx
            for i, price in enumerate(prices):
                if price > p_prev:
                    continue  # Skip if p increase
                prob = probabilities[i]
                # Expected if item sold
                value_if_sold = prob * (price + V[t + 1][x - 1][i]) if x > 0 else 0
                # Expected if item not sold
                value_if_not_sold = (1 - prob) * V[t + 1][x][p_prev_idx]
                # Total expected
                expected_value = value_if_sold + value_if_not_sold
                if expected_value > max_value:
                    max_value = expected_value
                    best_price_index = i

            V[t][x][p_prev_idx] = max_value
            policy[t][x][p_prev_idx] = best_price_index

initial_price_index = 0
expected_maximal_revenue = V[0][initial_inventory][initial_price_index]
print("Expected Maximal Revenue 2:", expected_maximal_revenue)
