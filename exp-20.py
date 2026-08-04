import numpy as np


# Warehouse Environment
class Warehouse:

    def __init__(self):
        self.tasks = [0, 0, 0]

    def assign_task(self, robot, task):

        if self.tasks[task] == 0:
            self.tasks[task] = robot
            reward = 10
        else:
            reward = -5

        return reward


# MARL Agent
class RobotAgent:

    def __init__(self, name):
        self.name = name
        self.q_table = np.zeros(3)

    def choose_task(self):
        return np.argmax(self.q_table)

    def update(self, task, reward):
        self.q_table[task] += 0.1 * reward


# Create environment
warehouse = Warehouse()

# Create three robots
robot1 = RobotAgent("Robot 1")
robot2 = RobotAgent("Robot 2")
robot3 = RobotAgent("Robot 3")

robots = [robot1, robot2, robot3]


# Training
for episode in range(50):

    warehouse = Warehouse()

    for robot in robots:

        task = np.random.randint(0, 3)

        reward = warehouse.assign_task(
            robots.index(robot),
            task
        )

        robot.update(task, reward)


print("MARL Training Completed\n")


# Display learned policies
for robot in robots:

    task = robot.choose_task()

    print(robot.name)
    print("Q Values:", np.round(robot.q_table, 2))
    print("Selected Task:", task)
    print()


print("Cooperative warehouse task allocation completed.")