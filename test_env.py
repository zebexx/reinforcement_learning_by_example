import gym
import random
import numpy as np
from IPython.display import clear_output
import time



env = gym.make("Taxi-v3").env
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# hyperparameters
learning_rate = 0.1
discount_rate = 0.6
exploration_rate = 0.1

# plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < exploration_rate:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values
        
        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_rate * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print("Episode: {}".format(i))

print("Training finished.\n")
print(q_table[328])