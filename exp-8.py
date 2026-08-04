import numpy as np
import random

# 3x3 room
size = 3
states = [(r, c) for r in range(size) for c in range(size)]

actions = ["up", "down", "left", "right"]

Q = {}
returns = {}

for s in states:
    for a in actions:
        Q[(s, a)] = 0
        returns[(s, a)] = []


def move(state, action):
    r, c = state

    if action == "up":
        r = max(0, r - 1)
    elif action == "down":
        r = min(size - 1, r + 1)
    elif action == "left":
        c = max(0, c - 1)
    elif action == "right":
        c = min(size - 1, c + 1)

    return (r, c)


def choose_action(state, epsilon=0.2):
    if random.random() < epsilon:
        return random.choice(actions)

    values = [Q[(state, a)] for a in actions]
    return actions[np.argmax(values)]


# Monte Carlo Control
episodes = 1000
gamma = 0.9

for episode in range(episodes):

    state = (0, 0)
    episode_data = []

    for step in range(30):

        action = choose_action(state)
        next_state = move(state, action)

        # Goal: reach bottom-right
        if next_state == (2, 2):
            reward = 10
            done = True
        else:
            reward = -1       # energy/movement cost
            done = False

        episode_data.append((state, action, reward))

        state = next_state

        if done:
            break

    G = 0
    visited = set()

    for state, action, reward in reversed(episode_data):

        G = gamma * G + reward

        if (state, action) not in visited:
            visited.add((state, action))

            returns[(state, action)].append(G)

            Q[(state, action)] = np.mean(
                returns[(state, action)]
            )


# Display learned policy
print("Monte Carlo Control completed")
print("\nLearned Cleaning Policy:")

for r in range(size):

    row = []

    for c in range(size):

        state = (r, c)

        if state == (2, 2):
            row.append("G")
        else:
            values = [Q[(state, a)] for a in actions]
            best = np.argmax(values)
            row.append(actions[best])

    print(row)