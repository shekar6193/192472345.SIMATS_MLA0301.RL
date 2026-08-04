
import numpy as np

ROWS = 4
COLS = 4

# Goal position
GOAL = (3, 3)

# Actions
actions = [
    (-1, 0),   # Up
    (1, 0),    # Down
    (0, -1),   # Left
    (0, 1)     # Right
]

gamma = 0.9

# State-value table
V = np.zeros((ROWS, COLS))


def get_next_state(state, action):
    row, col = state

    new_row = row + action[0]
    new_col = col + action[1]

    # Keep robot inside grid
    new_row = max(0, min(ROWS - 1, new_row))
    new_col = max(0, min(COLS - 1, new_col))

    return new_row, new_col


# Value Iteration
for iteration in range(100):

    new_V = np.copy(V)

    for row in range(ROWS):
        for col in range(COLS):

            if (row, col) == GOAL:
                continue

            values = []

            for action in actions:

                next_state = get_next_state(
                    (row, col), action
                )

                if next_state == GOAL:
                    reward = 10
                else:
                    reward = -1

                value = reward + gamma * V[next_state]
                values.append(value)

            new_V[row, col] = max(values)

    if np.max(abs(new_V - V)) < 0.001:
        break

    V = new_V


# Display value function
print("Optimal State-Value Function:")
print(np.round(V, 2))


# Find optimal path
state = (0, 0)
path = [state]

for step in range(20):

    if state == GOAL:
        break

    best_value = -float("inf")
    best_state = state

    for action in actions:

        next_state = get_next_state(state, action)

        if next_state == GOAL:
            reward = 10
        else:
            reward = -1

        value = reward + gamma * V[next_state]

        if value > best_value:
            best_value = value
            best_state = next_state

    state = best_state
    path.append(state)


print("\nOptimal Path:")
print(path)

print("\nResult:")
print("Robot reached the goal using Bellman's optimality equation.")