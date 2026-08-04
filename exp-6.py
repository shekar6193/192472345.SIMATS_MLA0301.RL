import gymnasium as gym
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# Create environment
env = gym.make("CartPole-v1")

# Get state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


# Create neural network
model = Sequential([
    Dense(
        24,
        activation="relu",
        input_shape=(state_size,)
    ),

    Dense(
        24,
        activation="relu"
    ),

    Dense(
        action_size,
        activation="linear"
    )
])


# Compile model
model.compile(
    optimizer="adam",
    loss="mse"
)


# Parameters
episodes = 100
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995


# Training
for episode in range(episodes):

    state, info = env.reset()

    state = np.reshape(
        state,
        [1, state_size]
    )

    total_reward = 0

    for step in range(500):

        # Choose action
        if np.random.rand() <= epsilon:

            action = env.action_space.sample()

        else:

            q_values = model.predict(
                state,
                verbose=0
            )

            action = np.argmax(
                q_values[0]
            )

        # Perform action
        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        next_state = np.reshape(
            next_state,
            [1, state_size]
        )

        total_reward += reward

        # Calculate target
        target = reward

        if not terminated and not truncated:

            next_q_values = model.predict(
                next_state,
                verbose=0
            )

            target = reward + gamma * np.max(
                next_q_values[0]
            )

        # Current Q-values
        target_f = model.predict(
            state,
            verbose=0
        )

        target_f[0][action] = target

        # Train model
        model.fit(
            state,
            target_f,
            epochs=1,
            verbose=0
        )

        state = next_state

        if terminated or truncated:
            break

    # Reduce exploration
    if epsilon > epsilon_min:

        epsilon *= epsilon_decay

    print(
        "Episode:",
        episode + 1,
        "Reward:",
        total_reward,
        "Epsilon:",
        round(epsilon, 3)
    )


env.close()

print("\nTraining Completed!")