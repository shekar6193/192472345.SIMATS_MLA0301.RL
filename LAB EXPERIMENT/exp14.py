import numpy as np

# Smart Elevator Environment
class ElevatorEnv:
    def __init__(self):
        self.state = 0
        self.waiting = 5

    def reset(self):
        self.state = 0
        self.waiting = 5
        return self.state

    def step(self, action):
        # action 0 = stay, action 1 = move up
        if action == 1:
            self.state += 1
            reward = 2
        else:
            reward = -2

        self.waiting -= 1

        if self.waiting <= 0:
            done = True
            reward += 10
        else:
            done = False

        return self.state, reward, done


# A2C Agent
class A2CAgent:
    def __init__(self):
        self.actor = np.zeros(2)
        self.critic = 0.0
        self.lr = 0.1

    def choose_action(self):
        probabilities = self.softmax(self.actor)
        return np.random.choice(2, p=probabilities)

    def softmax(self, x):
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)

    def train(self, states, actions, rewards):
        total_reward = sum(rewards)

        for action in actions:
            self.actor[action] += self.lr * total_reward

        self.critic += self.lr * (total_reward - self.critic)


# A3C-style multiple workers
def a3c_training():
    global_actor = np.zeros(2)

    for worker in range(3):
        env = ElevatorEnv()
        local_actor = np.zeros(2)

        for episode in range(10):
            state = env.reset()
            rewards = []

            for step in range(5):
                action = np.random.choice(2)
                next_state, reward, done = env.step(action)
                rewards.append(reward)

                local_actor[action] += 0.01 * reward

                if done:
                    break

        global_actor += local_actor

    return global_actor


# ---------------- A2C ----------------

env = ElevatorEnv()
agent = A2CAgent()

for episode in range(20):
    state = env.reset()

    states = []
    actions = []
    rewards = []

    for step in range(5):
        action = agent.choose_action()

        next_state, reward, done = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

        if done:
            break

    agent.train(states, actions, rewards)

print("A2C Training Completed")
print("Actor values:", agent.actor)
print("Critic value:", round(agent.critic, 2))

# ---------------- A3C ----------------

result = a3c_training()

print("\nA3C Training Completed")
print("Global Actor values:", result)
print("Elevator scheduling optimized.")