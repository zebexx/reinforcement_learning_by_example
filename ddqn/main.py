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

from tf_agents.environments import suite_gym

from ddqn import DDQNAgent


def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0", linewidth=0.5)
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1", linewidths=1)
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    #env = gym.make('MountainCar-v0')
    #env = gym.make('Acrobot-v1')
    env = gym.make('LunarLander-v2')
    #print(envs.registry.all())
    
    ddqn_agent = DDQNAgent(alpha=0.00025, gamma=0.99, n_actions=2, epsilon=1,
                  batch_size=32, input_dims=4, use_examples=False)
    num_episodes = 200
    #ddqn_agent.load_model()
    ddqn_scores = []
    eps_history = []
    #env = wrappers.Monitor(env, "tmp/lunar-lander-ddqn-2",
    #                         video_callable=lambda episode_id: True, force=True)
    
    for i in range(num_episodes):
        
        done = False
        score = 0
        if ddqn_agent.use_examples and ddqn_agent.example_memory.num_episodes > ddqn_agent.example_memory.episode_counter:
            print("Example Game")
            print("| Num Episodes: {} || Num_Examples: {} || Current Episode: {} |".format(ddqn_agent.example_memory.num_episodes, ddqn_agent.example_memory.num_examples, ddqn_agent.example_memory.episode_counter) )
            
            observation = ddqn_agent.example_memory.example_reset()

            while not done:
                action, observation_, reward, xdone, info = ddqn_agent.example_memory.example_step()
                done = not xdone
                #print(ddqn_agent.example_memory.example_step())
                score+=reward
                ddqn_agent.remember(observation, action, reward, observation_, int(done))
                observation = observation_
                ddqn_agent.learn()
        else:
            print("Real Game")
            observation = env.reset()
            while not done:
                #print(ddqn_agent.memory.mem_counter)
                #env.render()
                action = ddqn_agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                ddqn_agent.remember(observation, action, reward, observation_, int(done))
                observation = observation_
                ddqn_agent.learn()
        eps_history.append(ddqn_agent.epsilon)

        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])
        running_avg_score = np.mean(ddqn_scores[max(0, i-20):(i+1)])
        print('episode: ', i,'score: %.2f' % score,
              ' average score: %.2f' % avg_score, ' running average: %.2f' %running_avg_score)

        if i % 10 == 0 and i > 0:
            ddqn_agent.save_model()
    ddqn_agent.memory.save_memory()
    filename = 'LunarLander_ddqn_200ep_7-200_bs-32_bu-4_lr-0.00025_g-0.99_ledr-0.0001.png'

    x = [i+1 for i in range(num_episodes)]
    plotLearning(x, ddqn_scores, eps_history, filename)