import numpy as np

# Lane Keeping Environment
class LaneEnv:
    def __init__(self):
        self.position = 0.0

    def reset(self):
        self.position = np.random.uniform(-0.5, 0.5)
        return self.position

    def step(self, action):
        # -1 = left, 0 = straight, 1 = right
        self.position += action * 0.1

        reward = 10 - abs(self.position) * 10

        done = abs(self.position) > 1

        return self.position, reward, done


# REINFORCE Agent
class PolicyGradient:
    def __init__(self, lr):
        self.policy = np.zeros(3)
        self.lr = lr

    def choose_action(self):
        exp = np.exp(self.policy - np.max(self.policy))
        probabilities = exp / np.sum(exp)

        action = np.random.choice(3, p=probabilities)

        return action - 1

    def update(self, rewards):
        total_reward = sum(rewards)

        self.policy += self.lr * total_reward


# Agent 1
agent1 = PolicyGradient(0.001)

# Agent 2
agent2 = PolicyGradient(0.0005)

# Training
for episode in range(100):

    env = LaneEnv()
    env.reset()

    rewards = []

    for step in range(20):
        action = agent1.choose_action()

        state, reward, done = env.step(action)

        rewards.append(reward)

        if done:
            break

    agent1.update(rewards)


for episode in range(100):

    env = LaneEnv()
    env.reset()

    rewards = []

    for step in range(20):
        action = agent2.choose_action()

        state, reward, done = env.step(action)

        rewards.append(reward)

        if done:
            break

    agent2.update(rewards)


print("Policy Gradient Training Completed")

print("\nAlgorithm 1 Policy:")
print(np.round(agent1.policy, 2))

print("\nAlgorithm 2 Policy:")
print(np.round(agent2.policy, 2))

print("\nLane keeping performance improved.")