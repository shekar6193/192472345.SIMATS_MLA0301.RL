import numpy as np
import random

# States: drone position and battery
# Actions: 0=Left, 1=Right

class DQN:
    def __init__(self, input_size, output_size):
        self.W1 = np.random.randn(input_size, 16) * 0.1
        self.b1 = np.zeros(16)

        self.W2 = np.random.randn(16, output_size) * 0.1
        self.b2 = np.zeros(output_size)

    def predict(self, x):
        h = np.maximum(0, np.dot(x, self.W1) + self.b1)
        return np.dot(h, self.W2) + self.b2

    def train(self, x, target, lr=0.01):

        h = np.maximum(0, np.dot(x, self.W1) + self.b1)

        output = np.dot(h, self.W2) + self.b2

        error = output - target

        dW2 = np.outer(h, error)
        db2 = error

        dh = np.dot(error, self.W2.T)
        dh[h <= 0] = 0

        dW1 = np.outer(x, dh)
        db1 = dh

        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        self.W1 -= lr * dW1
        self.b1 -= lr * db1


def state_vector(position, battery):

    x = np.array([
        position / 4,
        battery / 10
    ])

    return x


model = DQN(2, 2)

gamma = 0.9
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

episodes = 1000

for episode in range(episodes):

    position = 0
    battery = 10

    for step in range(20):

        state = state_vector(position, battery)

        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(model.predict(state))

        # Move right
        if action == 1:
            new_position = min(4, position + 1)
        else:
            new_position = max(0, position - 1)

        battery -= 1

        if new_position == 4:
            reward = 20
            done = True

        elif battery <= 0:
            reward = -20
            done = True

        else:
            reward = -1
            done = False

        next_state = state_vector(new_position, max(battery, 0))

        target = model.predict(state).copy()

        if done:
            target[action] = reward
        else:
            target[action] = reward + gamma * np.max(
                model.predict(next_state)
            )

        model.train(state, target)

        position = new_position

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)


print("DQN Training Completed")

# Test
position = 0
battery = 10

print("\nDrone Route:")

while position != 4 and battery > 0:

    state = state_vector(position, battery)

    action = np.argmax(model.predict(state))

    if action == 1:
        position += 1
        print("Move RIGHT")
    else:
        position = max(0, position - 1)
        print("Move LEFT")

    battery -= 1

print("\nFinal Position:", position)
print("Remaining Battery:", battery)

if position == 4:
    print("Delivery completed successfully!")
else:
    print("Battery exhausted.")