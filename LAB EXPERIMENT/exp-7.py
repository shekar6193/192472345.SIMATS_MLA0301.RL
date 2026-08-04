import numpy as np

# 4x4 taxi/grid environment
rows, cols = 4, 4
states = rows * cols

# Actions: 0=Up, 1=Down, 2=Left, 3=Right
actions = [0, 1, 2, 3]

goal = 15
gamma = 0.9

# Obstacles
obstacles = [5, 6, 9, 10]

def next_state(state, action):
    r, c = divmod(state, cols)

    nr, nc = r, c

    if action == 0:
        nr -= 1
    elif action == 1:
        nr += 1
    elif action == 2:
        nc -= 1
    elif action == 3:
        nc += 1

    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
        return state

    ns = nr * cols + nc

    if ns in obstacles:
        return state

    return ns


def reward(state, next_s):
    if next_s == goal:
        return 10
    return -1


V = np.zeros(states)

# Value Iteration
for iteration in range(100):
    new_V = V.copy()

    for s in range(states):
        if s == goal or s in obstacles:
            continue

        values = []

        for a in actions:
            ns = next_state(s, a)
            values.append(reward(s, ns) + gamma * V[ns])

        new_V[s] = max(values)

    if np.max(abs(new_V - V)) < 0.001:
        break

    V = new_V


# Extract optimal policy
policy = []

symbols = ["↑", "↓", "←", "→"]

for s in range(states):
    if s == goal:
        policy.append("G")
    elif s in obstacles:
        policy.append("X")
    else:
        values = []

        for a in actions:
            ns = next_state(s, a)
            values.append(reward(s, ns) + gamma * V[ns])

        best_action = np.argmax(values)
        policy.append(symbols[best_action])


print("Value Iteration completed")
print("Iterations:", iteration + 1)

print("\nOptimal Policy:")
for r in range(rows):
    print(policy[r*cols:(r+1)*cols])

print("\nState Values:")
for r in range(rows):
    print([round(V[r*cols+c], 2) for c in range(cols)])