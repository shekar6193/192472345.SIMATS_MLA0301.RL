import numpy as np
import random

# Grid size
ROWS = 5
COLS = 5

# Start and goal
START = (0, 0)
GOAL = (4, 4)

# Actions
ACTIONS = [
    (-1, 0),  # Up
    (1, 0),   # Down
    (0, -1),  # Left
    (0, 1)    # Right
]

# Q-table
q_table = np.zeros((ROWS, COLS, 4))

# Parameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

episodes = 1000


# Get legal actions
def get_actions(state):
    row, col = state
    actions = []

    for i, (dr, dc) in enumerate(ACTIONS):
        new_row = row + dr
        new_col = col + dc

        if 0 <= new_row < ROWS and 0 <= new_col < COLS:
            actions.append(i)

    return actions


# Training
for episode in range(episodes):

    state = START

    for step in range(100):

        row, col = state
        legal_actions = get_actions(state)

        # Epsilon-greedy selection
        if random.random() < epsilon:
            action = random.choice(legal_actions)
        else:
            action = max(
                legal_actions,
                key=lambda a: q_table[row, col, a]
            )

        # New position
        dr, dc = ACTIONS[action]

        next_state = (
            row + dr,
            col + dc
        )

        # Reward
        if next_state == GOAL:
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        # Q-learning update
        old_value = q_table[row, col, action]

        if done:
            next_value = 0
        else:
            next_actions = get_actions(next_state)

            next_value = max(
                q_table[
                    next_state[0],
                    next_state[1],
                    a
                ]
                for a in next_actions
            )

        q_table[row, col, action] = old_value + alpha * (
            reward + gamma * next_value - old_value
        )

        state = next_state

        if done:
            break

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# Test the robot
state = START
path = [state]

for step in range(50):

    if state == GOAL:
        break

    row, col = state
    legal_actions = get_actions(state)

    action = max(
        legal_actions,
        key=lambda a: q_table[row, col, a]
    )

    dr, dc = ACTIONS[action]

    state = (
        row + dr,
        col + dc
    )

    path.append(state)


print("Smart Home Robot Navigation")
print("----------------------------")

for i, position in enumerate(path):
    print("Step", i, ":", position)

if state == GOAL:
    print("\nRobot successfully reached the destination!")
else:
    print("\nRobot could not reach the destination.")