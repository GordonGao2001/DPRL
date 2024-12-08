import numpy as np
import matplotlib.pyplot as plt

# Connect 4 setup (unchanged parts omitted for brevity)
ROWS, COLS = 6, 7

def create_board():
    return np.zeros((ROWS, COLS), dtype=int)

def available_moves(board):
    return [c for c in range(COLS) if board[0, c] == 0]

def drop_piece(board, col, player):
    for r in range(ROWS - 1, -1, -1):
        if board[r, col] == 0:
            board[r, col] = player
            break
    return board

def check_win(board, player):
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r, c:c + 4] == player):
                return True
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r:r + 4, c] == player):
                return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i, c + i] == player for i in range(4)):
                return True
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r - i, c + i] == player for i in range(4)):
                return True
    return False

# Purely random opponent
def purely_random(board):
    moves = available_moves(board)
    if moves:
        return np.random.choice(moves)
    return None

# UCB formula
def UCB(wins, n_i, N, c=np.sqrt(2)):
    if np.any(n_i == 0):
        return np.argwhere(n_i == 0)[0][0]
    return np.argmax(wins / n_i + c * np.sqrt(np.log(N) / n_i))

# Modified UCT with tracking
def UCT_with_tracking(board, player, num_simulations=1000):
    k = len(available_moves(board))
    if k == 0:
        return None
    n_i = np.zeros(k)
    wins = np.zeros(k)
    probabilities_over_time = []
    visits_over_time = []

    for t in range(num_simulations):
        a_t = UCB(wins, n_i, t + 1)
        n_i[a_t] += 1

        # Simulate game
        sim_board = board.copy()
        sim_board = drop_piece(sim_board, available_moves(sim_board)[a_t], player)

        current_player = 3 - player
        win = False
        while not win and available_moves(sim_board):
            a = purely_random(sim_board)
            sim_board = drop_piece(sim_board, a, current_player)
            win = check_win(sim_board, current_player)
            current_player = 3 - current_player

        if check_win(sim_board, player):
            wins[a_t] += 1

        probabilities_over_time.append(wins / n_i)  # Track probabilities
        visits_over_time.append(n_i.copy())         # Track visit counts

    return np.argmax(wins / n_i), probabilities_over_time, visits_over_time

# Generate convergence data
board = create_board()
best_action, probabilities, visits = UCT_with_tracking(board, 1, num_simulations=1000)

# Plot winning probabilities
plt.figure()
for col in range(len(probabilities[0])):
    plt.plot([p[col] if not np.isnan(p[col]) else 0 for p in probabilities], label=f'Column {col + 1}')
plt.xlabel('Simulation')
plt.ylabel('Winning Probability')
plt.title('Convergence of Winning Probabilities')
plt.legend()
plt.savefig('winning_probabilities_convergence.png')
plt.close()

# Plot visit counts
final_visits = visits[-1]
plt.figure()
plt.bar(range(1, len(final_visits) + 1), final_visits)
plt.xlabel('Column')
plt.ylabel('Number of Visits')
plt.title('Node Visitation Counts')
plt.savefig('node_visitation_counts.png')
plt.close()
