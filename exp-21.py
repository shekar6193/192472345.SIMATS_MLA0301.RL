import numpy as np

# Search and Rescue Environment
class RescueRobot:
    def __init__(self):
        self.position = 0
        self.victim_location = 4

    def reset(self):
        self.position = 0
        return self.position

    def observe(self):
        # Robot cannot directly know victim location
        if abs(self.position - self.victim_location) <= 1:
            return "victim_near"
        else:
            return "unknown"

    def step(self, action):
        # 0 = move left
        # 1 = move right
        # 2 = search

        if action == 0:
            self.position = max(0, self.position - 1)

        elif action == 1:
            self.position = min(5, self.position + 1)

        elif action == 2:
            if self.position == self.victim_location:
                return 20, True
            else:
                return -2, False

        return -1, False


robot = RescueRobot()

state = robot.reset()

print("POMDP Search-and-Rescue Simulation")
print("-----------------------------------")

for step in range(10):

    observation = robot.observe()

    print(
        "Step:", step + 1,
        "| Position:", robot.position,
        "| Observation:", observation
    )

    if observation == "victim_near":
        action = 2
    else:
        action = 1

    reward, done = robot.step(action)

    print("Reward:", reward)

    if done:
        print("Victim successfully rescued!")
        break