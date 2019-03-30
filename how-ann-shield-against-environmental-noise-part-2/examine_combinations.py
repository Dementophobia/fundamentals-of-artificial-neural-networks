import numpy as np
from os import environ
from tensorflow import keras
from sys import argv

def act(model, state):
    state = shape_state(state)
    action_values = model.predict(state)
    action = np.argmax(action_values)
    return action

def inspect_act(model, state, output):
    state = shape_state(state)
    action_values = model.predict(state)
    return action_values[0][output]
        
def shape_state(state):
    return np.reshape(state, [1, 2])

def print_output_matrix_after_activation(model, samples=10):
    results = np.zeros((samples, samples), dtype=int)

    for speed in range(samples):
        for position in range(samples):
            p = -1.2 + position * (0.6 + 1.2) / (samples - 1)
            s = -0.7 + speed    * (0.7 + 0.7) / (samples - 1)
            results[speed][position] = act(model, np.array((p, s)))

    print(results)

def print_output_matrix_before_activation(model, output, samples=10):
    layer_name = 'dense_11'
    inspect_model = keras.Model(inputs=model.input,
                                outputs=model.get_layer(layer_name).output)

    results = np.zeros((samples, samples), dtype=int)

    for speed in range(samples):
        for position in range(samples):
            p = -1.2 + position * (0.6 + 1.2) / (samples - 1)
            s = -0.7 + speed    * (0.7 + 0.7) / (samples - 1)
            results[speed][position] = inspect_act(model, np.array((p, s)), output)

    print(results)

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=2)

if len(argv) <= 1:
    id = 0
else:
    id = argv[1]

model = keras.models.load_model(f"trained_model_{id}.h5")

print_output_matrix_after_activation(model, 10)

for output in range(3):
    print("\n--------------------------------------------\n")
    print_output_matrix_before_activation(model, output, 10)

