# Household Robot HRL
# Tasks:
# 1 = Clean
# 2 = Pick Object
# 3 = Go Home


class HouseholdRobot:

    def __init__(self):
        self.location = "Home"

    def clean(self):
        print("Robot is cleaning the room")
        return 10

    def pick_object(self):
        print("Robot picked up the object")
        return 8

    def go_home(self):
        print("Robot returned home")
        self.location = "Home"
        return 5


# ---------------- HAM ----------------

class HAM:

    def execute(self, robot):

        print("\nHAM Hierarchical Controller")

        reward = 0

        reward += robot.clean()
        reward += robot.pick_object()
        reward += robot.go_home()

        return reward


# ---------------- MAXQ ----------------

class MAXQ:

    def execute(self, robot):

        print("\nMAXQ Hierarchical Controller")

        tasks = [
            robot.clean,
            robot.pick_object,
            robot.go_home
        ]

        total_reward = 0

        for task in tasks:
            total_reward += task()

        return total_reward


robot = HouseholdRobot()

ham = HAM()
ham_reward = ham.execute(robot)

maxq = MAXQ()
maxq_reward = maxq.execute(robot)


print("\nHAM Total Reward:", ham_reward)
print("MAXQ Total Reward:", maxq_reward)

print("\nHRL execution completed.")