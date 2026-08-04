import random


class Agent:

    def __init__(self, name):
        self.name = name
        self.completed_tasks = 0

    def move(self):
        print(self.name, "moves to task location")

    def collect(self):
        print(self.name, "collects resource")
        self.completed_tasks += 1

    def deliver(self):
        print(self.name, "delivers resource")


# -------------------------
# MAXQ Hierarchy
# -------------------------

class MAXQ:

    def execute_task(self, agent):

        print("\nExecuting MAXQ task for",
              agent.name)

        self.navigate(agent)

        self.collect_resource(agent)

        self.deliver_resource(agent)

    def navigate(self, agent):

        print("Subtask: Navigate")

        agent.move()

    def collect_resource(self, agent):

        print("Subtask: Collect")

        agent.collect()

    def deliver_resource(self, agent):

        print("Subtask: Deliver")

        agent.deliver()


agents = [
    Agent("Agent 1"),
    Agent("Agent 2"),
    Agent("Agent 3")
]

maxq = MAXQ()


for agent in agents:

    maxq.execute_task(agent)


print("\n-------------------------")
print("MAXQ Task Summary")
print("-------------------------")

total = 0

for agent in agents:

    print(
        agent.name,
        "completed:",
        agent.completed_tasks,
        "task(s)"
    )

    total += agent.completed_tasks


print("\nTotal Tasks Completed:",
      total)

print("\nResult:")
print("Hierarchical cooperative task completed.")