import numpy as np
import random


class Highway:

    def __init__(self):
        self.lane = 1
        self.speed = 50

    def reset(self):
        self.lane = 1
        self.speed = 50

    def step(self, action):

        # 0 = Stay
        # 1 = Change Left
        # 2 = Change Right

        if action == 1:
            self.lane = max(0, self.lane - 1)

        elif action == 2:
            self.lane = min(2, self.lane + 1)

        if self.lane == 1:
            reward = 2
        else:
            reward = 8

        return reward


class PPOAgent:

    def __init__(self):
        self.policy = np.zeros(3)
        self.lr = 0.01

    def choose_action(self):

        probability = np.exp(
            self.policy - np.max(self.policy)
        )

        probability /= np.sum(probability)

        return np.random.choice(
            3,
            p=probability
        )

    def update(self, action, reward):

        update = np.clip(
            self.lr * reward,
            -0.2,
            0.2
        )

        self.policy[action] += update


env = Highway()
agent = PPOAgent()

# Training
for episode in range(200):

    env.reset()

    for step in range(20):

        action = agent.choose_action()

        reward = env.step(action)

        agent.update(action, reward)


print("PPO Lane-Changing Training Completed")

print("\nPolicy Values:")
print(np.round(agent.policy, 2))

best_action = np.argmax(agent.policy)

actions = [
    "Stay",
    "Change Left",
    "Change Right"
]

print("\nBest Action:", actions[best_action])
print("Vehicle learned lane-changing behavior.")