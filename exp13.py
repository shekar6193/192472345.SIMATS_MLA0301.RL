import numpy as np
import random

# States represent parking positions
# 0 = Far left
# 1 = Left
# 2 = Center
# 3 = Right
# 4 = Parking position

states = 5
actions = 3

# Actions
# 0 = Left
# 1 = Straight
# 2 = Right

theta = np.zeros((states, actions))

alpha = 0.02
gamma = 0.9


def softmax(x):

    exp_x = np.exp(x - np.max(x))

    return exp_x / np.sum(exp_x)


def environment(state, action):

    if action == 0:
        next_state = max(0, state - 1)

    elif action == 1:
        next_state = state

    else:
        next_state = min(4, state + 1)

    if next_state == 4:
        reward = 20
        done = True
    else:
        reward = -1
        done = False

    return next_state, reward, done


# REINFORCE Training
for episode in range(1000):

    state = 0

    episode_data = []

    for step in range(20):

        probabilities = softmax(theta[state])

        action = np.random.choice(
            actions,
            p=probabilities
        )

        next_state, reward, done = environment(
            state, action
        )

        episode_data.append(
            (state, action, reward)
        )

        state = next_state

        if done:
            break


    # Calculate returns
    G = 0
    returns = []

    for _, _, reward in reversed(episode_data):

        G = reward + gamma * G

        returns.insert(0, G)


    # Policy update
    for i, (state, action, reward) in enumerate(
        episode_data
    ):

        probabilities = softmax(theta[state])

        for a in range(actions):

            if a == action:
                gradient = 1 - probabilities[a]
            else:
                gradient = -probabilities[a]

            theta[state, a] += (
                alpha *
                returns[i] *
                gradient
            )


print("REINFORCE Training Completed")

print("\nOptimal Parking Policy:")

names = ["LEFT", "STRAIGHT", "RIGHT"]

for state in range(4):

    probabilities = softmax(theta[state])

    action = np.argmax(probabilities)

    print(
        "Parking State",
        state,
        "->",
        names[action],
        "Probability:",
        round(probabilities[action], 2)
    )


# Test learned policy
state = 0

print("\nAutonomous Parking Path:")

for step in range(10):

    probabilities = softmax(theta[state])

    action = np.argmax(probabilities)

    print(
        "State:",
        state,
        "Action:",
        names[action]
    )

    next_state, reward, done = environment(
        state, action
    )

    state = next_state

    if done:
        print("Car parked successfully!")
        break