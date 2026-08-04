import numpy as np

# Traffic states
states = [
    "Low Traffic",
    "Medium Traffic",
    "High Traffic"
]

# Actions
actions = [
    "Short Green",
    "Long Green"
]

# Rewards
# Higher reward = lower waiting time
rewards = np.array([
    [-2, -1],
    [-5, -2],
    [-10, -3]
])

gamma = 0.9

# Initial policy
policy = np.zeros(3, dtype=int)

V = np.zeros(3)


# Policy Iteration
for iteration in range(20):

    # -------------------------
    # Policy Evaluation
    # -------------------------

    for _ in range(100):

        new_V = np.zeros(3)

        for state in range(3):

            action = policy[state]

            new_V[state] = (
                rewards[state][action]
                + gamma * V[state]
            )

        V = new_V

    # -------------------------
    # Policy Improvement
    # -------------------------

    stable = True

    for state in range(3):

        old_action = policy[state]

        values = []

        for action in range(2):

            value = (
                rewards[state][action]
                + gamma * V[state]
            )

            values.append(value)

        policy[state] = np.argmax(values)

        if old_action != policy[state]:
            stable = False

    if stable:
        break


print("Traffic Light Policy")
print("--------------------")

for i in range(3):

    print(
        states[i],
        "->",
        actions[policy[i]]
    )

print("\nOptimal State Values:")
print(np.round(V, 2))

print("\nResult:")
print("Traffic signal policy optimized successfully.")