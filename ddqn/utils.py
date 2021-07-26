import matplotlib.pyplot as plt
import numpy as np
import os


def plot(x, scores, epsilons, x1, example_scores, example_epsilsons, filename, lines=None):
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
    #ax2.plot(x, scores, color="lightcoral", linewidth=0.5)

    ax2.plot(x1, example_running_avg, color="b", label="Example")
    #ax2.plot(x1, example_scores, color="cornflowerblue", linewidth=0.5)

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

def saveGraphData(agents, directory):
    i = 0
    while os.path.exists("{}/{}/scores.csv".format(directory, i)):
        i +=1
    for agent in agents:
        np.savetxt("{}/{}/scores.csv".format(directory, i), agent[1], delimiter=",")
        np.savetxt("{}/{}/timeSteps.csv".format(directory, i), agent[2], delimiter=",")
        np.savetxt("{}/{}/epsilon.csv".format(directory, i), agent[3], delimiter=",")