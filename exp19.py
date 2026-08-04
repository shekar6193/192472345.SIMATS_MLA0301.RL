import numpy as np


# Industrial Robot Environment
class RobotTask:

    def __init__(self, task_type):
        self.task_type = task_type

    def get_reward(self, action):

        if action == self.task_type:
            return 10

        return -5


# Meta RL Agent
class MetaRLAgent:

    def __init__(self):
        self.policy = np.zeros(3)

    def choose_action(self):
        return np.argmax(self.policy)

    def adapt(self, action, reward):

        self.policy[action] += 0.1 * reward


agent = MetaRLAgent()

# Training on different tasks
tasks = [0, 1, 2]

for task in tasks:

    environment = RobotTask(task)

    for episode in range(10):

        action = agent.choose_action()

        reward = environment.get_reward(action)

        agent.adapt(action, reward)


print("Meta-Reinforcement Learning Completed")

print("Learned Policy:")
print(np.round(agent.policy, 2))


# Test new manufacturing task
new_task = RobotTask(2)

action = agent.choose_action()

reward = new_task.get_reward(action)

print("\nNew Task Action:", action)
print("Reward:", reward)

if reward > 0:
    print("Robot adapted successfully.")
else:
    print("Robot needs more adaptation.")