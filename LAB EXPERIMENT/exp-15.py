import numpy as np

# Humanoid robot environment
class HumanoidEnv:
    def __init__(self):
        self.balance = 0.0

    def reset(self):
        self.balance = 0.0
        return self.balance

    def step(self, action):
        # action: -1 = left, 0 = stay, 1 = right
        self.balance += action * 0.1

        if abs(self.balance) < 0.3:
            reward = 10
        else:
            reward = -10

        done = abs(self.balance) > 1.0

        return self.balance, reward, done


# PPO
class PPOAgent:
    def __init__(self):
        self.policy = np.zeros(3)
        self.lr = 0.01

    def choose_action(self):
        probabilities = self.softmax(self.policy)
        return np.random.choice(3, p=probabilities) - 1

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def update(self, reward, action):
        index = action + 1

        # PPO-style clipped update
        update = np.clip(self.lr * reward, -0.2, 0.2)
        self.policy[index] += update


# TRPO
class TRPOAgent:
    def __init__(self):
        self.policy = np.zeros(3)
        self.step_size = 0.01

    def choose_action(self):
        probabilities = self.softmax(self.policy)
        return np.random.choice(3, p=probabilities) - 1

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def update(self, reward, action):
        index = action + 1

        # Trust-region style small update
        update = np.clip(self.step_size * reward, -0.1, 0.1)
        self.policy[index] += update


# PPO Training
ppo = PPOAgent()

for episode in range(50):
    env = HumanoidEnv()
    state = env.reset()

    for step in range(20):
        action = ppo.choose_action()
        state, reward, done = env.step(action)
        ppo.update(reward, action)

        if done:
            break

# TRPO Training
trpo = TRPOAgent()

for episode in range(50):
    env = HumanoidEnv()
    state = env.reset()

    for step in range(20):
        action = trpo.choose_action()
        state, reward, done = env.step(action)
        trpo.update(reward, action)

        if done:
            break


print("PPO Training Completed")
print("PPO Policy:", np.round(ppo.policy, 2))

print("\nTRPO Training Completed")
print("TRPO Policy:", np.round(trpo.policy, 2))

print("\nComparison:")
print("PPO: Stable policy learning using clipped updates")
print("TRPO: Stable policy learning using trust-region updates")