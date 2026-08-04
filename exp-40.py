import numpy as np
import random


class StudentEnvironment:

    def __init__(self):

        self.level = 0

    def reset(self):

        self.level = random.randint(
            0, 2
        )

        return self.level

    def step(self, action):

        # Best content is approximately
        # equal to student's level

        difference = abs(
            self.level - action
        )

        if difference == 0:

            reward = 10

            # Student improves
            if self.level < 2:
                self.level += 1

        elif difference == 1:

            reward = 3

        else:

            reward = -5

        return self.level, reward


class EducationAgent:

    def __init__(self):

        self.q_table = np.zeros(
            (3, 3)
        )

        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def choose_action(self, state):

        if random.random() < self.epsilon:

            return random.randint(0, 2)

        return np.argmax(
            self.q_table[state]
        )

    def update(
        self,
        state,
        action,
        reward,
        next_state
    ):

        old_value = self.q_table[
            state,
            action
        ]

        next_value = np.max(
            self.q_table[next_state]
        )

        self.q_table[
            state,
            action
        ] = old_value + self.lr * (
            reward +
            self.gamma * next_value -
            old_value
        )


env = StudentEnvironment()

agent = EducationAgent()


# Training
for episode in range(500):

    state = env.reset()

    for step in range(20):

        action = agent.choose_action(
            state
        )

        next_state, reward = env.step(
            action
        )

        agent.update(
            state,
            action,
            reward,
            next_state
        )

        state = next_state


print("Personalized Education RL Training Completed")

print("\nLearned Q Table:")
print(np.round(agent.q_table, 2))


content = [
    "Easy Content",
    "Medium Content",
    "Difficult Content"
]


print("\nRecommended Content:")

for state in range(3):

    best_action = np.argmax(
        agent.q_table[state]
    )

    print(
        "Student Level",
        state,
        "->",
        content[best_action]
    )


print("\nResult:")
print("RL learned a personalized content recommendation policy.")