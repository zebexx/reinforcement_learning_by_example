import gym
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import matplotlib.pyplot as plt


env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n

def build_model(n_states, n_actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,n_states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    return model

model = build_model(states, actions)
memory = SequentialMemory(limit=50000, window_length=1)


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-3), metrics= ['mae'])

history = dqn.fit(env, nb_steps=20000, visualize=False, verbose=1)



# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['episode_reward'])
plt.plot(history.history['nb_episode_steps'])
plt.title('Reward / num episodes')
plt.ylabel('Reward')
plt.xlabel('Episodes')

plt.show()


