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

from tf_agents.environments import suite_gym

from ddqn import DDQNAgent


def plotLearning(x, scores, epsilons, x1, example_scores, example_epsilsons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="m", linewidth=0.5)
    #ax.plot(x1, example_epsilsons, color="C2", linewidth=0.5)
    
    
    ax.set_xlabel("Timesteps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-10):(t+1)])

    N = len(example_scores)
    example_running_avg = np.empty(N)
    for t in range(N):
	    example_running_avg[t] = np.mean(example_scores[max(0, t-10):(t+1)])

    ax2.plot(x, running_avg, color="r", label="Normal")
    ax2.plot(x, scores, color="lightcoral", linewidth=0.5)
    ax2.plot(x1, example_running_avg, color="b", label="Example")
    ax2.plot(x1, example_scores, color="cornflowerblue", linewidth=0.5)
    ax2.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2,
            borderaxespad=0, frameon=False, fontsize='x-small')

    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def isLast(agent, index):
    return agent.memory.terminal_memory[index] == 0



if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    
    
    ddqn_agent = DDQNAgent(alpha=0.00025, gamma=0.99, n_actions=2, epsilon=1, batch_size=32, input_dims=4, use_examples=False)

    ddqn_agent_example = DDQNAgent(alpha=0.00025, gamma=0.99, n_actions=2, epsilon=1, batch_size=32, input_dims=4, use_examples=True, example_location="example_data/CartPole-v1")


    

    timeSteps = 200000
    
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

    
    
    
    plotLearning(episode_timestep, ddqn_scores, eps_history, example_episode_timestep, example_ddqn_scores, example_eps_history, filename)