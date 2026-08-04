import numpy as np
import random


class HealthcareEnvironment:

    def __init__(self):

        self.patients = 5

    def reset(self):

        self.patients = 5

        return self.patients

    def step(self, action):

        # 0 = Normal treatment
        # 1 = Priority treatment

        if action == 0:

            outcome = 5
            cost = 2

        else:

            outcome = 10
            cost = 5

        reward = outcome - cost

        self.patients -= 1

        done = self.patients == 0

        return (
            self.patients,
            reward,
            done
        )


class RLAgent:

    def __init__(self):

        self.q_values = np.zeros(2)

        self.lr = 0.1

    def choose_action(self):

        if random.random() < 0.2:

            return random.randint(0, 1)

        return np.argmax(
            self.q_values
        )

    def update(self, action, reward):

        self.q_values[action] += (
            self.lr *
            (
                reward -
                self.q_values[action]
            )
        )


env = HealthcareEnvironment()

agent = RLAgent()


for episode in range(100):

    state = env.reset()

    while True:

        action = agent.choose_action()

        next_state, reward, done = env.step(
            action
        )

        agent.update(
            action,
            reward
        )

        if done:
            break


print("Healthcare RL Training Completed")

print("\nLearned Q Values:")
print(np.round(agent.q_values, 2))

actions = [
    "Normal Treatment",
    "Priority Treatment"
]

best = np.argmax(agent.q_values)

print("\nPreferred Action:",
      actions[best])

print("\nResult:")
print("Healthcare resource allocation policy learned.")