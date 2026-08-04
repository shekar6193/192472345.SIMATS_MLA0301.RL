import random

# -------------------------------------------------
# Autonomous Car Road Network
# -------------------------------------------------

road = {
    "Start": ["A"],
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["Destination"],
    "Destination": []
}

# Safe routes
safe_routes = [
    ["Start", "A", "B", "C", "Destination"],
    ["Start", "A", "C", "Destination"]
]


# -------------------------------------------------
# Random Policy
# -------------------------------------------------

def random_policy(current):
    choices = road[current]

    if len(choices) == 0:
        return None

    return random.choice(choices)


# -------------------------------------------------
# Safe Policy
# -------------------------------------------------

def safe_policy(current):
    if current == "Start":
        return "A"

    elif current == "A":
        # Choose B because it is a safe intermediate road
        return "B"

    elif current == "B":
        return "C"

    elif current == "C":
        return "Destination"

    return None


# -------------------------------------------------
# Evaluate Policy
# -------------------------------------------------

def evaluate_policy(policy, episodes=10):

    successful = 0
    total_steps = 0

    for episode in range(episodes):

        current = "Start"
        route = [current]

        for step in range(10):

            next_node = policy(current)

            if next_node is None:
                break

            current = next_node
            route.append(current)

            if current == "Destination":
                successful += 1
                total_steps += step + 1
                break

    success_rate = (successful / episodes) * 100

    if successful > 0:
        average_steps = total_steps / successful
    else:
        average_steps = 0

    return success_rate, average_steps


# -------------------------------------------------
# Display Road Network
# -------------------------------------------------

print("AUTONOMOUS CAR NAVIGATION")
print("-------------------------")

print("Road Network:")
print("Start -> A")
print("A -> B or C")
print("B -> C")
print("C -> Destination")


# -------------------------------------------------
# Demonstrate Safe Policy
# -------------------------------------------------

print("\nSafe Policy Route:")

current = "Start"
route = [current]

while current != "Destination":

    next_node = safe_policy(current)

    if next_node is None:
        break

    current = next_node
    route.append(current)

print(" -> ".join(route))


# -------------------------------------------------
# Evaluate Safe Policy
# -------------------------------------------------

success_rate, avg_steps = evaluate_policy(
    safe_policy,
    10
)

print("\nSafe Policy Evaluation")
print("----------------------")
print("Successful trips:", int(success_rate / 10), "/ 10")
print("Success Rate:", success_rate, "%")
print("Average Steps:", round(avg_steps, 2))


# -------------------------------------------------
# Evaluate Random Policy
# -------------------------------------------------

success_rate_random, avg_steps_random = evaluate_policy(
    random_policy,
    100
)

print("\nRandom Policy Evaluation")
print("------------------------")
print("Success Rate:",
      round(success_rate_random, 2), "%")

print("Average Steps:",
      round(avg_steps_random, 2))


# -------------------------------------------------
# Final Result
# -------------------------------------------------

print("\nResult:")
print("The autonomous car successfully follows")
print("traffic-safe routes and reaches the destination.")