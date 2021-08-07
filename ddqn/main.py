import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import gym
from gym import wrappers
from gym import envs
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import plot
from utils import saveGraphData

from tf_agents.environments import suite_gym

from ddqn import DDQNAgent
from train import train


if __name__ == '__main__':
    

    ### HYPERPARAMETERS ###
    alpha = 0.001
    gamma = 0.99

    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_discount_rate = 0.999

    replace_target = 500
 
    memory_size = 1000000
    batch_size = 32
    

    timeSteps = 400000


    example_data_location = "example_data/MountainCar-v0"
    #[lowerbound, upperbound, max quantity]
    example_episode_range = [0,20,50]
    primesteps = 100000

    env_name = "CartPole-v0"
    
    

    #Initialising agents
    env = gym.make(env_name)

    #ddqn_agent = DDQNAgent(alpha=alpha, gamma=gamma, n_actions=2, input_dims=4, epsilon=epsilon_start, batch_size=batch_size, batch_step=learn_from_batch, use_examples=False, epsilon_dec=epsilon_discount_rate, epsilon_end=epsilon_end, mem_size=memory_size, replace_target=replace_target)

    ddqn_agent_example = DDQNAgent(alpha=alpha, gamma=gamma, n_actions=2, input_dims=4, epsilon=epsilon_start, batch_size=batch_size, use_examples=True, primesteps=primesteps, episode_range=example_episode_range, example_location=example_data_location, epsilon_dec=epsilon_discount_rate, epsilon_end=epsilon_end, mem_size=memory_size, replace_target=replace_target)
    
    #Initialising agents graphing history

    #ddqn_scores = []
    #episode_timestep = []
    #eps_history = []
    #name = "Normal"

    example_ddqn_scores = []
    example_episode_timestep = []
    example_eps_history = []
    example_name = "Example"
    
    agents = []
    agents.append([ddqn_agent_example, example_ddqn_scores, example_episode_timestep, example_eps_history, example_name])
    #agents.append([ddqn_agent, ddqn_scores, episode_timestep, eps_history, name])


    
    for agent in agents:
        train(agent, timeSteps, env)
    

    #ddqn_agent.save_model()
    #ddqn_agent.memory.save_memory()

    
    name = "{}_ddqn_{}ts_Normal_{}_bs-{}_lr-{}_g-{}_edr-{}_em-{}_rt-{}".format(env_name, timeSteps, example_episode_range, batch_size, alpha, gamma, epsilon_discount_rate, epsilon_end, replace_target)
    
    saveGraphData(agents, "graphData", name)
    
    filename = "{}.png".format(name) 
    
    
    plot(agents, filename)