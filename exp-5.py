import random

# Number of advertisements
num_ads = 4

# True engagement probabilities
true_probabilities = [
    0.20,
    0.50,
    0.80,
    0.30
]

# Number of trials
trials = 1000

# Epsilon
epsilon = 0.1

# Estimated rewards
estimated_rewards = [0] * num_ads

# Number of times each ad was selected
counts = [0] * num_ads

# Total engagement
total_engagement = 0


# Epsilon-Greedy Algorithm
for trial in range(trials):

    # Exploration
    if random.random() < epsilon:
        ad = random.randint(
            0,
            num_ads - 1
        )

    # Exploitation
    else:
        ad = estimated_rewards.index(
            max(estimated_rewards)
        )

    # Simulate user engagement
    if random.random() < true_probabilities[ad]:
        reward = 1
    else:
        reward = 0

    # Update count
    counts[ad] += 1

    # Update estimated reward
    estimated_rewards[ad] += (
        reward - estimated_rewards[ad]
    ) / counts[ad]

    total_engagement += reward


# Display results
print("ONLINE ADVERTISEMENT RECOMMENDATION")
print("------------------------------------")

for i in range(num_ads):

    print(
        "Ad", i + 1,
        "Selected:", counts[i],
        "times",
        "Estimated Engagement:",
        round(
            estimated_rewards[i],
            3
        )
    )

best_ad = estimated_rewards.index(
    max(estimated_rewards)
)

print(
    "\nBest Advertisement: Ad",
    best_ad + 1
)

print(
    "Total Engagement:",
    total_engagement
)

print(
    "Engagement Rate:",
    round(
        total_engagement / trials,
        3
    )
)