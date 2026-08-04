import numpy as np
import random
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# --------------------------------
# Simple Highway Environment
# --------------------------------

class HighwayEnv:

    def __init__(self):
        self.position = 0
        self.speed = 1

    def reset(self):
        self.position = 0
        self.speed = 1

        return np.array(
            [self.position, self.speed],
            dtype=np.float32
        )

    def step(self, action):

        # 0 = Slow
        # 1 = Maintain
        # 2 = Accelerate

        if action == 0:
            self.speed = max(0, self.speed - 1)

        elif action == 2:
            self.speed = min(5, self.speed + 1)

        self.position += self.speed

        reward = self.speed

        if self.speed == 0:
            reward -= 5

        done = self.position >= 30

        return (
            np.array(
                [self.position, self.speed],
                dtype=np.float32
            ),
            reward,
            done
        )


# --------------------------------
# DQN Model
# --------------------------------

model = Sequential([
    Dense(32, activation="relu", input_shape=(2,)),
    Dense(32, activation="relu"),
    Dense(3, activation="linear")
])

model.compile(
    optimizer="adam",
    loss="mse"
)


env = HighwayEnv()

epsilon = 1.0
gamma = 0.9

# Training
for episode in range(50):

    state = env.reset()
    total_reward = 0

    for step in range(50):

        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            q_values = model.predict(
                state.reshape(1, -1),
                verbose=0
            )

            action = np.argmax(q_values[0])

        next_state, reward, done = env.step(action)

        target = reward

        if not done:

            next_q = model.predict(
                next_state.reshape(1, -1),
                verbose=0
            )

            target += gamma * np.max(next_q[0])

        target_q = model.predict(
            state.reshape(1, -1),
            verbose=0
        )

        target_q[0][action] = target

        model.fit(
            state.reshape(1, -1),
            target_q,
            epochs=1,
            verbose=0
        )

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = max(0.05, epsilon * 0.95)


print("DQN Training Completed")

# Evaluation
state = env.reset()
total_reward = 0

for step in range(30):

    q_values = model.predict(
        state.reshape(1, -1),
        verbose=0
    )

    action = np.argmax(q_values[0])

    state, reward, done = env.step(action)

    total_reward += reward

    if done:
        break

print("Evaluation Reward:", round(total_reward, 2))
print("Vehicle reached position:", state[0])

print("\nResult:")
print("DQN successfully trained the autonomous vehicle.")