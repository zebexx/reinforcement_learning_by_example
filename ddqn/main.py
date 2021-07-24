import os
# for keras the CUDA commands must come before importing the keras libraries
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import gym
from gym import wrappers
from gym import envs
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import plot

from tf_agents.environments import suite_gym

from ddqn import DDQNAgent






if __name__ == '__main__':

    ### HYPERPARAMETERS ###
    alpha = 0.00025
    gamma = 0.99

    epsilon_start = 1
    epsilon_end = 0.001
    epsilon_discount_rate = 0.999

    replace_target = 500

    memory_size = 1000000
    batch_size = 32

    timeSteps = 100000

    env = gym.make('CartPole-v1')
    
    
    ddqn_agent = DDQNAgent(alpha=alpha, gamma=gamma, n_actions=2, epsilon=epsilon_start, batch_size=batch_size, input_dims=4, use_examples=False, epsilon_dec=epsilon_discount_rate, epsilon_end=epsilon_end, mem_size=memory_size, replace_target=replace_target)

    ddqn_agent_example = DDQNAgent(alpha=alpha, gamma=gamma, n_actions=2, epsilon=1, batch_size=batch_size, input_dims=4, use_examples=True, example_location="example_data/CartPole-v1", epsilon_dec=epsilon_discount_rate, epsilon_end=epsilon_end, mem_size=memory_size, replace_target=replace_target)

    ddqn_scores = []
    episode_timestep = []
    eps_history = []

    example_ddqn_scores = []
    example_episode_timestep = []
    example_eps_history = []
    
    agents = []
    agents.append([ddqn_agent_example, example_ddqn_scores, example_episode_timestep, example_eps_history])
    agents.append([ddqn_agent, ddqn_scores, episode_timestep, eps_history])


    score = 0
    done = True

    #Training loop
    for agent in agents:

        for i in range(timeSteps):
            

            if agent[0].use_examples and agent[0].example_memory.mem_counter+1 <= agent[0].example_memory.num_examples and agent[0].epsilon <= 0.1 :
                
                if agent[0].example_memory.mem_counter == 0:
                    score =0
                
                if done:
                    observation = agent[0].example_memory.example_reset()

                
                action, observation_, reward, xdone, info = agent[0].example_memory.example_step()
                done = not xdone
                
                score+=reward
                agent[0].remember(observation, action, reward, observation_, int(done))
                observation = observation_
                agent[0].learn()
            else:
                
                if done:
                    observation = env.reset()
                    
                action = agent[0].choose_action(observation)
                
                observation_, reward, done, info = env.step(action)
                score += reward
                agent[0].remember(observation, action, reward, observation_, int(done))
                observation = observation_
                agent[0].learn()
            
            if done: 
                agent[1].append(score)
                agent[2].append(i+1)
                agent[3].append(agent[0].epsilon)
                

                avg_score = np.mean(agent[1][max(0, len(agent[1])-100):(len(agent[1])+1)])
                running_avg_score = np.mean(agent[1][max(0, len(agent[1])-10):(len(agent[1])+1)])
                print(agent[0].name , '-', 'Episode:', len(agent[1]),'Timestep:', i, '/', timeSteps, 'Score: %.2f' % score,' Running average: %.2f' %running_avg_score)
    
                score = 0

        score = 0
        done = True

        avg_score = 0
        running_avg_score = 0
        timeInMins = time.process_time()/60
        print("Process time-", "{:.1f}".format(timeInMins), "minutes")

    

    ddqn_agent.save_model()
    ddqn_agent.memory.save_memory()
    filename = 'Cartpole-v1_ddqn_200000ts_Normal_bs-32_bu-4_lr-0.00025_g-0.99_edr-0.9999_em-0.1_rt-500.png'

    
    
    
    plot(episode_timestep, ddqn_scores, eps_history, example_episode_timestep, example_ddqn_scores, example_eps_history, filename)