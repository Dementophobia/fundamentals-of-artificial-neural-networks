import gym
import numpy as np
from os import environ
from tensorflow import keras
from sys import argv

def act(model, state):
    state = shape_state(state)
    action_values = model.predict(state)
    action = np.argmax(action_values)
    return action
        
def shape_state(state):
    return np.reshape(state, [1, 2])

def solve(model):
    for i in range(10):
        state = env.reset()
        terminal = False

        while not terminal:
            env.render()
            action = act(model, state)
            state, _, terminal, _ = env.step(action)
            
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
env = gym.make("MountainCar-v0")

if len(argv) <= 1:
    id = 0
else:
    id = argv[1]

model = keras.models.load_model(f"trained_model_{id}.h5")
solve(model)