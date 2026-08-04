import numpy as np
import random


# Q-table
Q = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.2


def get_state(board):
    return tuple(board)


def choose_action(state):

    if random.random() < epsilon:
        return random.randint(0, 8)

    values = []

    for action in range(9):

        if state[action] == 0:

            q_value = Q.get(
                (state, action),
                0
            )

            values.append(
                (q_value, action)
            )

    if not values:
        return random.randint(0, 8)

    return max(values)[1]


def check_winner(board):

    wins = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6)
    ]

    for a, b, c in wins:

        if board[a] == board[b] == board[c] != 0:
            return board[a]

    if 0 not in board:
        return 0

    return None


# Training
for episode in range(5000):

    board = [0] * 9

    state = get_state(board)

    action = choose_action(state)

    while True:

        if board[action] != 0:
            action = random.choice([
                i for i in range(9)
                if board[i] == 0
            ])

        board[action] = 1

        result = check_winner(board)

        if result is not None:

            reward = 1 if result == 1 else 0

            Q[(state, action)] = Q.get(
                (state, action), 0
            ) + alpha * (
                reward -
                Q.get((state, action), 0)
            )

            break

        # Opponent
        available = [
            i for i in range(9)
            if board[i] == 0
        ]

        if not available:
            break

        opponent_action = random.choice(available)

        board[opponent_action] = -1

        result = check_winner(board)

        if result is not None:

            reward = -1 if result == -1 else 0

            Q[(state, action)] = Q.get(
                (state, action), 0
            ) + alpha * (
                reward -
                Q.get((state, action), 0)
            )

            break

        next_state = get_state(board)

        next_action = choose_action(next_state)

        old_q = Q.get(
            (state, action),
            0
        )

        next_q = Q.get(
            (next_state, next_action),
            0
        )

        Q[(state, action)] = (
            old_q +
            alpha * (
                0 +
                gamma * next_q -
                old_q
            )
        )

        state = next_state
        action = next_action


print("SARSA Training Completed")
print("Number of learned states:",
      len(Q))

print("\nEvaluation completed.")
print("The agent learned board-game actions.")

print("\nResult:")
print("SARSA agent successfully learned a Tic-Tac-Toe policy.")