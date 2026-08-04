import random
import numpy as np

# Board size
BOARD_SIZE = 8

# Actions
ACTIONS = [
    (-1, 0),   # Up
    (1, 0),    # Down
    (0, -1),   # Left
    (0, 1),    # Right
    (-1, -1),  # Up-Left
    (-1, 1),   # Up-Right
    (1, -1),   # Down-Left
    (1, 1)     # Down-Right
]

# Start and goal positions
START = (0, 0)
GOAL = (7, 7)

# Q-table
# State = (row, column)
q_table = np.zeros((8, 8, len(ACTIONS)))

# Parameters
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 1.0     # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

episodes = 5000


# Get legal actions
def get_legal_actions(position):
    row, col = position
    legal_actions = []

    for i, (dr, dc) in enumerate(ACTIONS):
        new_row = row + dr
        new_col = col + dc

        if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
            legal_actions.append(i)

    return legal_actions


# Training
for episode in range(episodes):

    position = START

    for step in range(100):

        row, col = position

        legal_actions = get_legal_actions(position)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(legal_actions)
        else:
            q_values = q_table[row, col, legal_actions]
            action = legal_actions[np.argmax(q_values)]

        # Perform action
        dr, dc = ACTIONS[action]

        new_position = (
            row + dr,
            col + dc
        )

        # Reward
        if new_position == GOAL:
            reward = 100
            done = True

        else:
            reward = -1
            done = False

        # Current Q-value
        current_q = q_table[row, col, action]

        # Next maximum Q-value
        if done:
            max_next_q = 0
        else:
            next_actions = get_legal_actions(new_position)

            max_next_q = max(
                q_table[
                    new_position[0],
                    new_position[1],
                    next_actions
                ]
            )

        # Q-learning update
        q_table[row, col, action] = current_q + alpha * (
            reward + gamma * max_next_q - current_q
        )

        # Move to next state
        position = new_position

        if done:
            break

    # Reduce exploration
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# Test the trained agent

print("\nOptimal Path Learned by Agent:")
print("--------------------------------")

position = START
path = [position]

for step in range(20):

    if position == GOAL:
        break

    row, col = position

    legal_actions = get_legal_actions(position)

    # Select best learned action
    action = legal_actions[
        np.argmax(
            q_table[row, col, legal_actions]
        )
    ]

    dr, dc = ACTIONS[action]

    position = (
        row + dr,
        col + dc
    )

    path.append(position)


# Display path
for i, position in enumerate(path):
    print("Move", i, ":", position)

if position == GOAL:
    print("\nResult: Agent reached the goal and WON!")
else:
    print("\nResult: Agent did not reach the goal.")