import numpy as np
import random

# True click probabilities
true_ctr = [0.10, 0.20, 0.15, 0.30, 0.25]

NUM_ADS = len(true_ctr)
TRIALS = 500


# ---------------- EPSILON GREEDY ----------------

def epsilon_greedy(epsilon):

    clicks = np.zeros(NUM_ADS)
    counts = np.zeros(NUM_ADS)

    total_clicks = 0

    for t in range(TRIALS):

        if random.random() < epsilon:
            ad = random.randint(0, NUM_ADS - 1)
        else:
            values = clicks / (counts + 1e-9)
            ad = np.argmax(values)

        click = np.random.rand() < true_ctr[ad]

        counts[ad] += 1

        if click:
            clicks[ad] += 1
            total_clicks += 1

    return total_clicks / TRIALS


# ---------------- UCB ----------------

def ucb():

    clicks = np.zeros(NUM_ADS)
    counts = np.zeros(NUM_ADS)

    total_clicks = 0

    for t in range(TRIALS):

        if t < NUM_ADS:
            ad = t

        else:
            average = clicks / (counts + 1e-9)

            confidence = np.sqrt(
                2 * np.log(t + 1) /
                (counts + 1e-9)
            )

            ad = np.argmax(
                average + confidence
            )

        click = np.random.rand() < true_ctr[ad]

        counts[ad] += 1

        if click:
            clicks[ad] += 1
            total_clicks += 1

    return total_clicks / TRIALS


# ---------------- THOMPSON SAMPLING ----------------

def thompson_sampling():

    successes = np.ones(NUM_ADS)
    failures = np.ones(NUM_ADS)

    total_clicks = 0

    for t in range(TRIALS):

        samples = np.random.beta(
            successes,
            failures
        )

        ad = np.argmax(samples)

        click = np.random.rand() < true_ctr[ad]

        if click:
            successes[ad] += 1
            total_clicks += 1
        else:
            failures[ad] += 1

    return total_clicks / TRIALS


eg = epsilon_greedy(0.1)
ucb_result = ucb()
ts = thompson_sampling()


print("Advertisement Bandit Comparison")
print("--------------------------------")

print("Epsilon-Greedy CTR:",
      round(eg, 3))

print("UCB CTR:",
      round(ucb_result, 3))

print("Thompson Sampling CTR:",
      round(ts, 3))

results = {
    "Epsilon-Greedy": eg,
    "UCB": ucb_result,
    "Thompson Sampling": ts
}

best = max(
    results,
    key=results.get
)

print("\nBest Algorithm:", best)