import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Dense,
    Lambda
)

from tensorflow.keras.models import Model


# State size = 4
# Actions = 4

state_size = 4
action_size = 4


# --------------------------------
# Standard DQN
# --------------------------------

def create_dqn():

    inputs = Input(shape=(state_size,))

    x = Dense(32, activation="relu")(inputs)

    x = Dense(32, activation="relu")(x)

    outputs = Dense(
        action_size,
        activation="linear"
    )(x)

    model = Model(inputs, outputs)

    return model


# --------------------------------
# Dueling DQN
# --------------------------------

def create_dueling_dqn():

    inputs = Input(shape=(state_size,))

    x = Dense(32, activation="relu")(inputs)

    x = Dense(32, activation="relu")(x)

    # Value stream
    value = Dense(1)(x)

    # Advantage stream
    advantage = Dense(action_size)(x)

    # Combine streams
    outputs = Lambda(
        lambda x:
        x[0] +
        (x[1] - tf.reduce_mean(
            x[1],
            axis=1,
            keepdims=True
        ))
    )([value, advantage])

    model = Model(inputs, outputs)

    return model


dqn = create_dqn()

dueling = create_dueling_dqn()

dqn.compile(
    optimizer="adam",
    loss="mse"
)

dueling.compile(
    optimizer="adam",
    loss="mse"
)


print("Standard DQN Created")
print("Dueling DQN Created")

print("\nStandard DQN Parameters:",
      dqn.count_params())

print("Dueling DQN Parameters:",
      dueling.count_params())

print("\nDueling DQN uses separate")
print("Value and Advantage streams.")

print("\nResult:")
print("Dueling DQN architecture implemented successfully.")