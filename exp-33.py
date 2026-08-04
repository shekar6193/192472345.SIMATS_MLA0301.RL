import numpy as np


class ResourceEnvironment:

    def __init__(self):
        self.resources = 0

    def reset(self):
        self.resources = 0
        return self.resources

    def step(self, action):

        # Continuous action from 0 to 1
        action = np.clip(action, 0, 1)

        gathered = action * 10

        self.resources += gathered

        reward = gathered

        done = self.resources >= 100

        return self.resources, reward, done


class DDPGAgent:

    def __init__(self):
        self.actor_weight = 0.5
        self.critic_weight = 0.5

    def choose_action(self, state):

        action = self.actor_weight

        noise = np.random.normal(
            0,
            0.1
        )

        return np.clip(
            action + noise,
            0,
            1
        )

    def update(self, reward):

        self.actor_weight += (
            0.001 * reward
        )

        self.actor_weight = np.clip(
            self.actor_weight,
            0,
            1
        )


env = ResourceEnvironment()

agent = DDPGAgent()


for episode in range(50):

    state = env.reset()

    for step in range(30):

        action = agent.choose_action(state)

        next_state, reward, done = env.step(
            action
        )

        agent.update(reward)

        state = next_state

        if done:
            break


print("DDPG Training Completed")

print("Learned Actor Weight:",
      round(agent.actor_weight, 3))

print("Resources collected:",
      round(state, 2))

print("\nResult:")
print("DDPG agent learned resource gathering.")