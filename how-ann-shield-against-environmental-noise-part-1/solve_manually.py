import gym

def act(state):
	speed = state[1]
	
	if speed < 0:
		return 0  # Push car to the left
	elif speed > 0:
		return 2  # Push car to the right
	
	return 1  # Do not push
	
env = gym.make("MountainCar-v0")
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space)

run = 0

while True:
    run  += 1
    state = env.reset()
    step  = 0
    terminal = False
    
    while not terminal:
        step += 1
        env.render()
        action = act(state)
        state, _, terminal, _ = env.step(action)

        if terminal:
            print(f"Run: {run}, Score: {200-step}")