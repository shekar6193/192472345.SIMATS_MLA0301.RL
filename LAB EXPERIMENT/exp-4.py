import numpy as np

# Grid size
ROWS = 5
COLS = 5

# Goal
GOAL = (4, 4)

# Initialize value table
V = np.full(
    (ROWS, COLS),
    1000.0
)

# Goal cost
V[GOAL] = 0

# Actions
ACTIONS = [
    (-1, 0),  # Up
    (1, 0),   # Down
    (0, -1),  # Left
    (0, 1)    # Right
]

# Cost for each move
MOVE_COST = 1

# Bellman Value Iteration
for iteration in range(100):

    new_V = V.copy()

    for row in range(ROWS):
        for col in range(COLS):

            if (row, col) == GOAL:
                continue

            values = []

            for dr, dc in ACTIONS:

                new_row = row + dr
                new_col = col + dc

                if (
                    0 <= new_row < ROWS and
                    0 <= new_col < COLS
                ):
                    cost = MOVE_COST + V[
                        new_row,
                        new_col
                    ]

                    values.append(cost)

            new_V[row, col] = min(values)

    # Check convergence
    if np.max(np.abs(new_V - V)) < 0.001:
        break

    V = new_V


# Find optimal path
state = (0, 0)
path = [state]

while state != GOAL:

    row, col = state

    best_value = float("inf")
    best_state = state

    for dr, dc in ACTIONS:

        new_row = row + dr
        new_col = col + dc

        if (
            0 <= new_row < ROWS and
            0 <= new_col < COLS
        ):

            value = V[
                new_row,
                new_col
            ]

            if value < best_value:
                best_value = value
                best_state = (
                    new_row,
                    new_col
                )

    if best_state == state:
        break

    state = best_state
    path.append(state)


print("AUTONOMOUS DELIVERY ROBOT")
print("--------------------------")

print("\nValue Table:")

print(np.round(V, 2))

print("\nOptimal Path:")

for i, position in enumerate(path):
    print("Step", i, ":", position)

print("\nTotal Travel Cost:", len(path) - 1)