import numpy as np
import random

# Different prices
prices = [50, 70, 90, 110, 130]

# Probability of customer buying
buy_probability = [
    0.80,
    0.65,
    0.50,
    0.35,
    0.20
]

NUM_PRICES = len(prices)
TRIALS = 500


# ---------------- EPSILON GREEDY ----------------

def epsilon_greedy(epsilon):

    revenue = np.zeros(NUM_PRICES)
    counts = np.zeros(NUM_PRICES)

    total_revenue = 0

    for t in range(TRIALS):

        if random.random() < epsilon:
            price = random.randint(
                0, NUM_PRICES - 1
            )
        else:
            average = revenue / (
                counts + 1e-9
            )

            price = np.argmax(average)

        sale = (
            np.random.rand()
            < buy_probability[price]
        )

        counts[price] += 1

        if sale:
            revenue[price] += prices[price]
            total_revenue += prices[price]

    return total_revenue


# ---------------- UCB ----------------

def ucb():

    revenue = np.zeros(NUM_PRICES)
    counts = np.zeros(NUM_PRICES)

    total_revenue = 0

    for t in range(TRIALS):

        if t < NUM_PRICES:
            price = t

        else:

            average = revenue / (
                counts + 1e-9
            )

            confidence = np.sqrt(
                2 * np.log(t + 1) /
                (counts + 1e-9)
            )

            price = np.argmax(
                average + confidence
            )

        sale = (
            np.random.rand()
            < buy_probability[price]
        )

        counts[price] += 1

        if sale:
            revenue[price] += prices[price]
            total_revenue += prices[price]

    return total_revenue


# ---------------- THOMPSON SAMPLING ----------------

def thompson_sampling():

    success = np.ones(NUM_PRICES)
    failure = np.ones(NUM_PRICES)

    total_revenue = 0

    for t in range(TRIALS):

        samples = np.random.beta(
            success,
            failure
        )

        price = np.argmax(
            samples * np.array(prices)
        )

        sale = (
            np.random.rand()
            < buy_probability[price]
        )

        if sale:
            success[price] += 1
            total_revenue += prices[price]
        else:
            failure[price] += 1

    return total_revenue


eg = epsilon_greedy(0.1)
ucb_result = ucb()
ts = thompson_sampling()


print("Dynamic Pricing Comparison")
print("---------------------------")

print("Epsilon-Greedy Revenue: ₹",
      round(eg, 2))

print("UCB Revenue: ₹",
      round(ucb_result, 2))

print("Thompson Sampling Revenue: ₹",
      round(ts, 2))


results = {
    "Epsilon-Greedy": eg,
    "UCB": ucb_result,
    "Thompson Sampling": ts
}

best = max(
    results,
    key=results.get
)

print("\nBest Pricing Strategy:", best)