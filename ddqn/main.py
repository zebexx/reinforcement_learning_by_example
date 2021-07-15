import os
# for keras the CUDA commands must come before importing the keras libraries
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

from ddqn import DDQNAgent


def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    #ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    #ax.plot(x, epsilons, color="C0")
    #ax.set_xlabel("Game", color="C0")
    #ax.set_ylabel("Epsilon", color="C0")
    #ax.tick_params(axis='x', colors="C0")
    #ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.plot(x, scores, color="C3", linewidth=1)
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_xlabel("Episode", color="C0")
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
    ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=2, epsilon=1,
                  batch_size=64, input_dims=4, use_examples=True)
    n_games = 10
    #ddqn_agent.load_model()
    ddqn_scores = []
    eps_history = []
    #env = wrappers.Monitor(env, "tmp/lunar-lander-ddqn-2",
    #                         video_callable=lambda episode_id: True, force=True)
    
    for i in range(n_games):
        done = False
        score = 0
        if ddqn_agent.use_examples and ddqn_agent.example_memory.num_examples > ddqn_agent.example_memory.mem_counter:
            observation = ddqn_agent.example_memory.sample_example()
            while not done:
                
        else:
            observation = env.reset()
            while not done:
                action = ddqn_agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                ddqn_agent.remember(observation, action, reward, observation_, int(done))
                observation = observation_
                ddqn_agent.learn()
        eps_history.append(ddqn_agent.epsilon)

        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score: %.2f' % score,
              ' average score %.2f' % avg_score)

        if i % 10 == 0 and i > 0:
            ddqn_agent.save_model()
    ddqn_agent.memory.save_memory()
    filename = 'CartPole-v0-ddqn-with-example.png'

    x = [i+1 for i in range(n_games)]
    plotLearning(x, ddqn_scores, eps_history, filename)