import numpy as np
import random

# States
# 0 = Start
# 1 = Move to object
# 2 = Pick object
# 3 = Move to destination
# 4 = Place object
# 5 = Finished

states = 6
actions = 2

# 0 = slow movement
# 1 = efficient movement

theta = np.zeros((states, actions))

alpha = 0.01
gamma = 0.9


def softmax(x):

    exp_x = np.exp(x - np.max(x))

    return exp_x / np.sum(exp_x)


def step(state, action):

    if state == 0:
        next_state = 1
    elif state == 1:
        next_state = 2
    elif state == 2:
        next_state = 3
    elif state == 3:
        next_state = 4
    elif state == 4:
        next_state = 5
    else:
        next_state = 5

    if action == 1:
        reward = 2
    else:
        reward = 1

    if next_state == 5:
        reward += 10

    return next_state, reward


# Policy Gradient Training
for episode in range(1000):

    state = 0

    episode_data = []

    while state != 5:

        probabilities = softmax(theta[state])

        action = np.random.choice(
            actions,
            p=probabilities
        )

        next_state, reward = step(state, action)

        episode_data.append(
            (state, action, reward)
        )

        state = next_state

    G = 0

    for state, action, reward in reversed(episode_data):

        G = reward + gamma * G

        probabilities = softmax(theta[state])

        for a in range(actions):

            if a == action:
                theta[state, a] += alpha * G * (
                    1 - probabilities[a]
                )
            else:
                theta[state, a] -= alpha * G * probabilities[a]


print("Policy Gradient Training Completed")

print("\nLearned Policy:")

for state in range(5):

    probabilities = softmax(theta[state])

    best_action = np.argmax(probabilities)

    if best_action == 0:
        action_name = "Slow Movement"
    else:
        action_name = "Efficient Movement"

    print(
        "State", state,
        "->", action_name,
        "Probability:",
        round(probabilities[best_action], 2)
    )