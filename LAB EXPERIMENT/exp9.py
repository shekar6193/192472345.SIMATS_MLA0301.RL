import numpy as np
import random

states = 6
actions = 2

# 0 = Left, 1 = Right
goal = 5

alpha = 0.1
gamma = 0.9
epsilon = 0.1


def step(state, action):

    if action == 1:
        next_state = min(state + 1, goal)
    else:
        next_state = max(state - 1, 0)

    if next_state == goal:
        reward = 10
        done = True
    else:
        reward = -1
        done = False

    return next_state, reward, done


def choose_action(Q, state):

    if random.random() < epsilon:
        return random.randint(0, 1)

    return np.argmax(Q[state])


# -----------------------------
# TD(0)
# -----------------------------
V = np.zeros(states)

for episode in range(500):

    state = 0

    for step_no in range(20):

        action = random.randint(0, 1)

        next_state, reward, done = step(state, action)

        V[state] = V[state] + alpha * (
            reward + gamma * V[next_state] - V[state]
        )

        state = next_state

        if done:
            break


# -----------------------------
# SARSA
# -----------------------------
Q_sarsa = np.zeros((states, actions))

for episode in range(500):

    state = 0
    action = choose_action(Q_sarsa, state)

    for step_no in range(20):

        next_state, reward, done = step(state, action)

        next_action = choose_action(Q_sarsa, next_state)

        Q_sarsa[state, action] += alpha * (
            reward +
            gamma * Q_sarsa[next_state, next_action] -
            Q_sarsa[state, action]
        )

        state = next_state
        action = next_action

        if done:
            break


# -----------------------------
# Q-Learning
# -----------------------------
Q_qlearning = np.zeros((states, actions))

for episode in range(500):

    state = 0

    for step_no in range(20):

        action = choose_action(Q_qlearning, state)

        next_state, reward, done = step(state, action)

        Q_qlearning[state, action] += alpha * (
            reward +
            gamma * np.max(Q_qlearning[next_state]) -
            Q_qlearning[state, action]
        )

        state = next_state

        if done:
            break


print("TD(0) State Values:")
print(np.round(V, 2))

print("\nSARSA Q-Table:")
print(np.round(Q_sarsa, 2))

print("\nQ-Learning Q-Table:")
print(np.round(Q_qlearning, 2))

print("\nBest actions using Q-Learning:")

for s in range(states - 1):
    action = np.argmax(Q_qlearning[s])

    if action == 0:
        print("State", s, "-> LEFT")
    else:
        print("State", s, "-> RIGHT")