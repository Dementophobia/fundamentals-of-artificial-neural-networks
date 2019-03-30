import random
import gym
import numpy as np
from os import environ
from collections import deque
from tensorflow import keras

class ANN_Factory:
    def __init__(self):
        self.exploration_rate  = 1.0
        self.batch_size        = 20
        self.discount          = 0.99
        self.exploration_decay = 0.997
        self.exploration_min   = 0.01
    
    def init_new_model(self, name):
        self.name    = name
        self.model   = self.create_model()
        self.storage = deque(maxlen=10**6)

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Dense(20, input_shape=(2,), activation="relu"),
            keras.layers.Dense(20, activation="relu"),
            keras.layers.Dense(3, activation="linear")
        ])
        model.compile(loss="mse", optimizer="Adam")
       
        return model
    
    def store(self, state, action, reward, next_state, terminal):
        self.storage.append((state, action, reward, next_state, terminal))
        
    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(3)

        action_values = self.model.predict(state)
        action = np.argmax(action_values)
        return action
        
    def replay(self):
        if len(self.storage) < self.batch_size:
            return

        batch = random.sample(self.storage, self.batch_size)
        for state, action, reward, next_state, terminal in batch:
            if not terminal:
                reward += self.discount * np.amax(self.model.predict(next_state)[0])
            q = self.model.predict(state)
            q[0][action] = reward
            self.model.fit(state, q, verbose=0)
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    def save(self):
        self.model.save(f"models\trained_model_{self.name}.h5")


def shape_state(state):
    return np.reshape(state, [1, 2])

def train_ann(factory, name):
    factory.init_new_model(name)
    
    run = 0
    successes_in_a_row = 0

    while run < 10000:  # We assume that the model won't converge if it doesn't succeed within 10000 runs
        run  += 1
        state = env.reset()
        state = shape_state(state)
        step  = 0
        terminal = False
        reward = -1

        while not terminal:
            step += 1
            action = factory.act(state)
            next_state, _, terminal, _ = env.step(action)
            
            reward = max(reward, next_state[0])
            next_state = shape_state(next_state)
            
            factory.store(state, action, reward, next_state, terminal)
            
            state = next_state

        if step < 200:
            successes_in_a_row += 1
        else:
            successes_in_a_row = 0

        print(f"Model: {name}, Successes in a Row: {successes_in_a_row}, ER: {factory.exploration_rate}, Run: {run}, Reward: {reward}")
            
        if successes_in_a_row == 5:
            factory.save()
            return True
            
        factory.replay()
    return False
        
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
env = gym.make("MountainCar-v0")

factory = ANN_Factory()

finished_models = 0

while finished_models < 10:
    name = str(finished_models)
    if train_ann(factory, name):
        finished_models += 1