import numpy as np

# Parameters
states = [(i, j) for i in range(1, 6) for j in range(1, 6)]  # Inventory levels
actions = [(a1, a2) for a1 in range(0, 6) for a2 in range(0, 6)]  # Restocking actions
num_states = len(states)
num_actions = len(actions)
state_to_index = {state: idx for idx, state in enumerate(states)}
action_to_index = {action: idx for idx, action in enumerate(actions)}

# Transition probabilities
def get_transition_probabilities():
    P = np.zeros((num_states, num_states, num_actions))
    for s_idx, state in enumerate(states):
        for a_idx, action in enumerate(actions):
            for d1, d2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                prob = 0.25
                new_s1 = min(5, max(1, state[0] - d1 + action[0]))
                new_s2 = min(5, max(1, state[1] - d2 + action[1]))
                next_idx = state_to_index[(new_s1, new_s2)]
                P[s_idx, next_idx, a_idx] += prob
    return P

# Cost function
def cost_function(state, action):
    s1, s2 = state
    holding_cost = s1 + 2 * s2
    order_cost = 5 if action[0] > 0 or action[1] > 0 else 0
    return holding_cost + order_cost

# Value iteration
def value_iteration(P, max_iterations=1000, tolerance=1e-6):
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    for iteration in range(max_iterations):
        V_new = np.zeros_like(V)
        for s_idx, state in enumerate(states):
            action_costs = []
            for a_idx, action in enumerate(actions):
                immediate_cost = cost_function(state, action)
                future_cost = sum(P[s_idx, next_s_idx, a_idx] * V[next_s_idx] for next_s_idx in range(num_states))
                action_costs.append(immediate_cost + future_cost)
            V_new[s_idx] = min(action_costs)
            policy[s_idx] = np.argmin(action_costs)
        if np.max(np.abs(V_new - V)) < tolerance:
            break
        V = V_new
    return V, policy

# Main execution
P = get_transition_probabilities()
V, optimal_policy_indices = value_iteration(P)

# Extract optimal policy
optimal_policy = [actions[a_idx] for a_idx in optimal_policy_indices]

# Display results
print("Optimal Policy and Value Function:")
for state, action, value in zip(states, optimal_policy, V):
    print(f"State: {state}, Action: {action}, Value: {value:.2f}")
