import numpy as np


# Simulated historical returns
np.random.seed(10)

stock_returns = np.random.normal(
    0.08, 0.15, 100
)

bond_returns = np.random.normal(
    0.04, 0.05, 100
)

gold_returns = np.random.normal(
    0.05, 0.08, 100
)


# Portfolio weights
portfolios = {

    "Conservative": [
        0.2, 0.6, 0.2
    ],

    "Balanced": [
        0.5, 0.3, 0.2
    ],

    "Aggressive": [
        0.7, 0.1, 0.2
    ]
}


print("Portfolio Performance Analysis")
print("--------------------------------")


results = {}


for name, weights in portfolios.items():

    portfolio_return = (
        weights[0] * stock_returns +
        weights[1] * bond_returns +
        weights[2] * gold_returns
    )

    average_return = np.mean(
        portfolio_return
    )

    risk = np.std(
        portfolio_return
    )

    predicted_value = (
        100000 *
        (1 + average_return) ** 5
    )

    results[name] = predicted_value

    print("\n", name)

    print("Average Return:",
          round(average_return * 100, 2),
          "%")

    print("Risk:",
          round(risk * 100, 2),
          "%")

    print("Predicted 5-Year Value: ₹",
          round(predicted_value, 2))


best = max(
    results,
    key=results.get
)

print("\nBest Predicted Portfolio:",
      best)

print("\nResult:")
print("Alternative portfolios were analyzed and compared.")