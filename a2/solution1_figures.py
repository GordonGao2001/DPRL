import matplotlib.pyplot as plt

# Plotting Simulation Results (Average Cost Over Time)
def plot_simulation():
    state = (ORDER_LEVEL, ORDER_LEVEL)
    costs = []
    cumulative_cost = 0
    np.random.seed(42)

    for t in range(NUM_SIMULATION_STEPS_SMALL):
        # Simulate demand
        demand_1 = np.random.choice([0, 1], p=[1 - DEMAND_PROBABILITY, DEMAND_PROBABILITY])
        demand_2 = np.random.choice([0, 1], p=[1 - DEMAND_PROBABILITY, DEMAND_PROBABILITY])

        # Update inventory after demand
        next_state = (max(1, state[0] - demand_1), max(1, state[1] - demand_2))
        order = next_state[0] == 1 or next_state[1] == 1
        if order:
            next_state = (ORDER_LEVEL, ORDER_LEVEL)

        # Compute cost
        step_cost = get_state_cost(next_state)
        cumulative_cost += step_cost
        costs.append(cumulative_cost / (t + 1))  # Average cost so far

        # Move to the next state
        state = next_state

    plt.figure(figsize=(10, 6))
    plt.plot(range(NUM_SIMULATION_STEPS_SMALL), costs, label="Simulation Average Cost")
    plt.xlabel("Timestep")
    plt.ylabel("Average Cost")
    plt.title("Simulation: Convergence of Average Cost Over Time")
    plt.legend()
    plt.grid()
    plt.show()


plot_simulation()


import numpy as np
import matplotlib.pyplot as plt

# Value iteration for Bellman Equation
def bellman_optimal_policy(P, max_iterations=1000):
    J = np.zeros(num_states)  # Initialize differential costs
    policy = np.zeros(num_states, dtype=int)  # Store optimal actions
    g = 0  # Long-run average cost

    for iteration in range(max_iterations):
        J_new = np.zeros_like(J)

        for i, state in enumerate(states):
            # Cost-to-go for action A=0 (No Order)
            cost_no_order = get_state_cost(state) + np.sum(P[i, :] * J)
            
            # Cost-to-go for action A=1 (Order)
            reset_state = (ORDER_LEVEL, ORDER_LEVEL)
            reset_index = state_index[reset_state]
            cost_order = FIXED_ORDER_COST + get_state_cost(reset_state) + J[reset_index]

            # Choose the action that minimizes cost
            if cost_no_order <= cost_order:
                J_new[i] = cost_no_order
                policy[i] = 0  # No order
            else:
                J_new[i] = cost_order
                policy[i] = 1  # Order

        # Update long-run average cost
        g = np.mean(J_new - J)

        # Check for convergence
        if np.max(np.abs(J_new - J)) < CONVERGENCE_THRESHOLD:
            break

        # Update differential costs for the next iteration
        J = J_new - g

    return J, g, policy

# Solve Bellman equation and find optimal policy
J_bellman, average_cost_bellman, optimal_policy_bellman = bellman_optimal_policy(P)

# Plotting the optimal policy
def plot_optimal_policy(policy):
    policy_matrix = np.zeros((MAX_CAPACITY, MAX_CAPACITY))
    for state, action in zip(states, policy):
        policy_matrix[state[0] - 1, state[1] - 1] = action

    plt.figure(figsize=(10, 8))
    plt.imshow(policy_matrix, cmap="coolwarm", origin="lower")
    plt.colorbar(label="Action (0: No Order, 1: Order)")
    plt.xlabel("Inventory Level of Product 2")
    plt.ylabel("Inventory Level of Product 1")
    plt.title("Optimal Policy Heatmap")
    plt.show()

# Plot the optimal policy
plot_optimal_policy(optimal_policy_bellman)
