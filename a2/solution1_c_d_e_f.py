import numpy as np

# Parameters
MAX_CAPACITY = 20
ORDER_LEVEL = 5
FIXED_ORDER_COST = 5
HOLDING_COST_1 = 1
HOLDING_COST_2 = 2
DEMAND_PROBABILITY = 0.5
NUM_SIMULATION_STEPS_SMALL = 10000
CONVERGENCE_THRESHOLD = 1e-6

# Define state space
states = [(i, j) for i in range(1, MAX_CAPACITY + 1) for j in range(1, MAX_CAPACITY + 1)]
state_index = {state: idx for idx, state in enumerate(states)}
num_states = len(states)

# Cost function
def get_state_cost(state):
    holding_cost = state[0] * HOLDING_COST_1 + state[1] * HOLDING_COST_2
    order_cost = FIXED_ORDER_COST if state[0] == 1 or state[1] == 1 else 0
    return holding_cost + order_cost

# Transition probabilities
def get_transition_probabilities():
    P = np.zeros((num_states, num_states))
    for i, state in enumerate(states):
        for demand_1 in [0, 1]:
            for demand_2 in [0, 1]:
                prob = 0.25  # Each combination of demands has a probability of 0.25
                next_state = (max(1, state[0] - demand_1), max(1, state[1] - demand_2))
                if next_state[0] == 1 or next_state[1] == 1:
                    next_state = (ORDER_LEVEL, ORDER_LEVEL)
                next_index = state_index[next_state]
                P[i, next_index] += prob
    return P

# c) Simulation-based long-run average cost
def simulate_fixed_policy(steps):
    total_cost = 0
    state = (ORDER_LEVEL, ORDER_LEVEL)  # Initial state
    np.random.seed(42)  # For reproducibility
    for _ in range(steps):
        # Simulate demand for both products
        demand_1 = np.random.choice([0, 1], p=[1 - DEMAND_PROBABILITY, DEMAND_PROBABILITY])
        demand_2 = np.random.choice([0, 1], p=[1 - DEMAND_PROBABILITY, DEMAND_PROBABILITY])
        # Update inventory after demand
        next_state = (max(1, state[0] - demand_1), max(1, state[1] - demand_2))
        order = next_state[0] == 1 or next_state[1] == 1
        if order:
            next_state = (ORDER_LEVEL, ORDER_LEVEL)
        step_cost = get_state_cost(next_state)
        total_cost += step_cost
        state = next_state
    return total_cost / steps

# d) Limiting distribution and long-run average cost
def compute_limiting_distribution(P):
    pi = np.ones(num_states) / num_states  # Start with uniform distribution
    while True:
        new_pi = pi @ P
        if np.max(np.abs(new_pi - pi)) < CONVERGENCE_THRESHOLD:
            break
        pi = new_pi
    average_cost = sum(pi[i] * get_state_cost(states[i]) for i in range(num_states))
    return pi, average_cost

# e) Solve Poisson equation using value iteration
def solve_poisson_equation_corrected(P, max_iterations=1000):
    J = np.zeros(num_states)  # Initialize differential costs
    g = 0  # Initial average cost (we will extract this later)
    for iteration in range(max_iterations):
        J_new = np.zeros_like(J)
        for i in range(num_states):
            # Compute the new J(S) based on transition probabilities and current J values
            J_new[i] = get_state_cost(states[i]) + np.sum(P[i, :] * J)
        # Extract average cost from differential costs
        g = np.mean(J_new - J)
        # Check convergence
        if np.max(np.abs(J_new - J)) < CONVERGENCE_THRESHOLD:
            break
        J = J_new - g  # Normalize by subtracting average cost to stabilize iteration
    return J, g


# Define the Bellman Equation and solve using value iteration
def value_iteration(P, max_iterations=1000):
    J = np.zeros(num_states)  # Initialize differential costs
    policy = np.zeros(num_states, dtype=int)  # Store optimal actions (0: no order, 1: order)
    g = 0  # Long-run average cost

    for iteration in range(max_iterations):
        J_new = np.zeros_like(J)

        for i, state in enumerate(states):
            # Compute costs for two possible actions: no order (A=0) and order (A=1)
            cost_no_order = get_state_cost(state) + np.sum(P[i, :] * J)
            
            # Action: Place an order to reset inventory to ORDER_LEVEL
            reset_state = (ORDER_LEVEL, ORDER_LEVEL)
            reset_index = state_index[reset_state]
            cost_order = FIXED_ORDER_COST + get_state_cost(reset_state) + J[reset_index]

            # Select the action with the minimum cost
            if cost_no_order <= cost_order:
                J_new[i] = cost_no_order
                policy[i] = 0  # No order
            else:
                J_new[i] = cost_order
                policy[i] = 1  # Order

        # Compute the average cost from differential costs
        g = np.mean(J_new - J)
        
        # Check for convergence
        if np.max(np.abs(J_new - J)) < CONVERGENCE_THRESHOLD:
            break

        # Update J for the next iteration
        J = J_new - g

    return J, g, policy



# Main execution
P = get_transition_probabilities()

# Question c: Simulation-based average cost
average_cost_c = simulate_fixed_policy(NUM_SIMULATION_STEPS_SMALL)

# Question d: Limiting distribution and long-run average cost
limiting_distribution, average_cost_d = compute_limiting_distribution(P)

# Question e: Poisson equation and long-run average cost
J_corrected, average_cost_e = solve_poisson_equation_corrected(P)


# Question f: Solve the Bellman equation using value iteration
J_optimal, average_cost_optimal, optimal_policy = value_iteration(P)



# Results

print(f"C. Long-run average cost (Simulation-based): {average_cost_c}")
print(f"D. Long-run average cost (Limiting distribution): {average_cost_d}")
print(f"E. Long-run average cost (Poisson equation): {average_cost_e}")
print(f"F. Long-run average cost (Bellman equation): {average_cost_optimal}")
