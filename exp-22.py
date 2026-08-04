import numpy as np
import random

GRID_SIZE = 4

FOOD = (2, 1)
GHOST = (1, 2)

# Q-table
q_table = np.zeros(
    (GRID_SIZE, GRID_SIZE, 4)
)

# Actions
# 0 = Up
# 1 = Down
# 2 = Left
# 3 = Right

alpha = 0.1
gamma = 0.9
epsilon = 0.2


def move(state, action):

    row, col = state

    if action == 0:
        row = max(0, row - 1)

    elif action == 1:
        row = min(GRID_SIZE - 1, row + 1)

    elif action == 2:
        col = max(0, col - 1)

    elif action == 3:
        col = min(GRID_SIZE - 1, col + 1)

    return row, col


# Training
for episode in range(1000):

    state = (0, 0)

    for step in range(50):

        # Epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state[0], state[1]])

        next_state = move(state, action)

        # Reward
        if next_state == FOOD:
            reward = 10
            done = True

        elif next_state == GHOST:
            reward = -10
            done = True

        else:
            reward = -1
            done = False

        # Q-learning update
        old_value = q_table[
            state[0], state[1], action
        ]

        next_max = np.max(
            q_table[
                next_state[0],
                next_state[1]
            ]
        )

        new_value = old_value + alpha * (
            reward + gamma * next_max - old_value
        )

        q_table[
            state[0],
            state[1],
            action
        ] = new_value

        state = next_state

        if done:
            break


# Evaluation
print("Q-Learning Training Completed")
print("\nEvaluation:")

state = (0, 0)

for step in range(20):

    print("Step:", step + 1, "Position:", state)

    if state == FOOD:
        print("Food collected!")
        break

    if state == GHOST:
        print("Agent hit the ghost!")
        break

    action = np.argmax(
        q_table[state[0], state[1]]
    )

    state = move(state, action)