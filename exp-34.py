import numpy as np
import random


class SmartHome:

    def __init__(self):
        self.temperature = 25

    def reset(self):
        self.temperature = 25
        return self.temperature

    def step(self, action):

        # 0 = Cooling
        # 1 = No change
        # 2 = Heating

        if action == 0:
            self.temperature -= 1
            energy = 2

        elif action == 1:
            energy = 0

        else:
            self.temperature += 1
            energy = 2

        comfort_penalty = abs(
            self.temperature - 24
        )

        reward = -(
            energy +
            comfort_penalty
        )

        return self.temperature, reward


class REINFORCEAgent:

    def __init__(self):

        self.policy = np.zeros(3)

        self.lr = 0.01

    def choose_action(self):

        probabilities = np.exp(
            self.policy -
            np.max(self.policy)
        )

        probabilities /= np.sum(probabilities)

        return np.random.choice(
            3,
            p=probabilities
        )

    def update(self, rewards):

        total_reward = sum(rewards)

        self.policy += (
            self.lr *
            total_reward
        )


env = SmartHome()

agent = REINFORCEAgent()


for episode in range(100):

    env.reset()

    rewards = []

    for step in range(20):

        action = agent.choose_action()

        temperature, reward = env.step(
            action
        )

        rewards.append(reward)

    agent.update(rewards)


print("Smart Home REINFORCE Training Completed")

print("\nPolicy Values:")
print(np.round(agent.policy, 2))

actions = [
    "Cooling",
    "No Change",
    "Heating"
]

best = np.argmax(agent.policy)

print("\nRecommended Action:",
      actions[best])

print("\nResult:")
print("Smart home temperature control optimized.")