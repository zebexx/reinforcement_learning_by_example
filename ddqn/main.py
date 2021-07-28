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
    alpha = 0.00025
    gamma = 0.99

    epsilon_start = 1
    epsilon_end = 0.1
    epsilon_discount_rate = 0.9999

    replace_target = 500

    memory_size = 1000000
    batch_size = 32

    timeSteps = 200000

    #[lowerbound, upperbound, max quantity]
    example_episode_range = [0,50,50]

    env = gym.make('CartPole-v1')
    

    #Initialising agents
    
    ddqn_agent = DDQNAgent(alpha=alpha, gamma=gamma, n_actions=2, epsilon=epsilon_start, batch_size=batch_size, input_dims=4, use_examples=False, epsilon_dec=epsilon_discount_rate, epsilon_end=epsilon_end, mem_size=memory_size, replace_target=replace_target)

    ddqn_agent_example = DDQNAgent(alpha=alpha, gamma=gamma, n_actions=2, epsilon=1, batch_size=batch_size, input_dims=4, use_examples=True, episode_range=example_episode_range, example_location="example_data/CartPole-v1", epsilon_dec=epsilon_discount_rate, epsilon_end=epsilon_end, mem_size=memory_size, replace_target=replace_target)
    
    #Initialising agents graphing history

    ddqn_scores = []
    episode_timestep = []
    eps_history = []

    example_ddqn_scores = []
    example_episode_timestep = []
    example_eps_history = []
    
    agents = []
    agents.append([ddqn_agent_example, example_ddqn_scores, example_episode_timestep, example_eps_history])
    agents.append([ddqn_agent, ddqn_scores, episode_timestep, eps_history])


    
    for agent in agents:
        train(agent, timeSteps, env)
    

    ddqn_agent.save_model()
    ddqn_agent.memory.save_memory()

    saveGraphData(agents, "graphData")
    filename = 'Cartpole-v1_ddqn_200000ts_Normal_[500-500]_bs-32_bu-4_lr-0.00025_g-0.99_edr-0.9999_em-0.1_rt-500.png'
       
    
    
    plot(episode_timestep, ddqn_scores, eps_history, example_episode_timestep, example_ddqn_scores, example_eps_history, filename)