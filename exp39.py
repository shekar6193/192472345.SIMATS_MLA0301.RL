import random


class Robot:

    def __init__(self):

        self.position = 0

        # Hidden goal
        self.goal = 4

    def observe(self):

        distance = abs(
            self.position - self.goal
        )

        if distance <= 1:
            return "Goal Nearby"

        return "Unknown"

    def move(self):

        self.position += 1

        if self.position > 5:
            self.position = 5


robot = Robot()

print("POMDP Robot Navigation")
print("----------------------")

for step in range(10):

    observation = robot.observe()

    print(
        "Step:", step + 1,
        "| Position:", robot.position,
        "| Observation:", observation
    )

    if robot.position == robot.goal:

        print("Robot reached hidden goal!")
        break

    # Robot moves based on partial observation
    robot.move()


print("\nResult:")
print("Robot successfully navigated using partial observations.")