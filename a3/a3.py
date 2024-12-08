import numpy as np

# Define Connect 4 dimensions
ROWS, COLS = 6, 7

# Initialize board with zeros (empty)
def create_board():
    return np.zeros((ROWS, COLS), dtype=int)

# Check for a win in Connect 4
def check_win(board, player):
    # Check horizontal locations
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r, c:c + 4] == player):
                return True

    # Check vertical locations
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r:r + 4, c] == player):
                return True

    # Check positively sloped diagonals
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i, c + i] == player for i in range(4)):
                return True

    # Check negatively sloped diagonals
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r - i, c + i] == player for i in range(4)):
                return True

    return False

# Get available moves (columns that are not full)
def available_moves(board):
    return [c for c in range(COLS) if board[0, c] == 0]

# Drop a piece in a column
def drop_piece(board, col, player):
    for r in range(ROWS - 1, -1, -1):  # Start from the bottom row
        if board[r, col] == 0:
            board[r, col] = player
            break
    return board

# Purely random opponent logic
def purely_random(board):
    moves = available_moves(board)
    if moves:
        return np.random.choice(moves)
    return None

# Upper Confidence Bound (UCT formula)
def UCB(wins, n_i, N, c=np.sqrt(2)):
    if np.any(n_i == 0):
        return np.argwhere(n_i == 0)[0][0]
    return np.argmax(wins / n_i + c * np.sqrt(np.log(N) / n_i))

# Monte Carlo Tree Search with UCT for Connect 4
def UCT(board, player, num_simulations=1000):
    k = len(available_moves(board))
    if k == 0:
        return None
    n_i = np.zeros(k)
    wins = np.zeros(k)

    for t in range(num_simulations):
        a_t = UCB(wins, n_i, t + 1)
        n_i[a_t] += 1

        # Simulate game
        sim_board = board.copy()
        sim_board = drop_piece(sim_board, available_moves(sim_board)[a_t], player)

        current_player = 3 - player  # Switch player
        win = False
        while not win and available_moves(sim_board):
            a = purely_random(sim_board)
            sim_board = drop_piece(sim_board, a, current_player)
            win = check_win(sim_board, current_player)
            current_player = 3 - current_player

        if check_win(sim_board, player):
            wins[a_t] += 1

    return available_moves(board)[np.argmax(wins / n_i)]

# Simulate a Connect 4 game
def simulate_game():
    board = create_board()
    current_player = 1  # Player 1 starts

    while True:
        if current_player == 1:
            col = UCT(board, current_player)
        else:
            col = purely_random(board)

        if col is None:  # No valid moves left (draw)
            print("Game ended in a draw.")
            break

        board = drop_piece(board, col, current_player)
        if check_win(board, current_player):
            print(f"Player {current_player} wins!")
            break

        current_player = 3 - current_player  # Switch players

    print(board)

# Run simulation
simulate_game()
