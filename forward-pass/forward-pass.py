import tensorflow as tf
from tensorflow import keras
import numpy as np
from os import environ

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(2, input_shape=(2,), activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    weights_layer = [[np.array([[0.5, 0.2], [0.1, 0.2]]),
                      np.array([0.1, 0.1])],
                     [np.array([[0.3, -0.1], [0.2, 0.1]]),
                      np.array([0.2, -0.1])],
                     [np.array([[0.1, -0.2], [0.1, 0.2]]),
                      np.array([0.1, 0.])]]

    for layer in range(3):
        model.layers[layer].set_weights(weights_layer[layer])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model

def relu(value):
    return(max(0., value))

def softmax(values):
    return np.exp(values) / np.sum(np.exp(values))

def forward_pass(input, model):
    values = np.zeros([2, 4])  # Setting all values of layers incl. input and output layer to zero
    values[:, 0] = input       # Setting the input layer

    for layer in range(3):  # Going through all 3 layers of the ANN
        for i in range(2):  # Calculating values from top to bottom of the layer
            for i_prev_layer in range(2):  # Update the value based on the input of each neuron in the previous layer
                weight = model.layers[layer].get_weights()[0][i_prev_layer][i]
                prev_value = values[i_prev_layer][layer]
                values[i][layer+1] += prev_value * weight
            
            bias = model.layers[layer].get_weights()[1][i]
            values[i][layer+1] += bias
            
            if model.layers[layer].get_config()["activation"] == "relu":
                values[i][layer+1] = relu(values[i][layer+1])
               
        if model.layers[layer].get_config()["activation"] == "softmax":
            values[:, layer+1] = softmax(values[:, layer+1])

    print("\nAll the values of the layers created during the forward pass:\n")
    print(values)
    print("\n-----")
            
    return np.array([values[:,-1]])
   
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=2)

model = build_model()

print("\nThe following weights and biases have been set for the model:\n")
for i in range(3):
    print(model.layers[i].get_weights())
print("\n-----")

input = np.array([np.array([1., 1.])])

values_from_framework = model.predict(input)
values_from_function  = forward_pass(input, model)

print("\nValues calculated by the framework with the function 'predict':")
print(values_from_framework)
print("\nValues calculated by the self-made function 'forward_pass':")
print(values_from_function)
