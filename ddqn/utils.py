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

    
    running_avg = average_over(scores, 10)
    avg = average_over(scores, 100)

    example_running_avg = average_over(example_scores, 10)
    example_avg = average_over(example_scores, 100)

    ax2.plot(x, running_avg, color="r", label="Normal", linewidth=0.5)
    ax2.plot(x, avg, color="lightcoral")

    ax2.plot(x1, example_running_avg, color="b", label="Example", linewidth=0.5)
    ax2.plot(x1, example_avg, color="cornflowerblue")

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

def average_over(data, avg_range):
        N = len(data)
        avg = np.empty(N)
        for t in range(N):
	        avg[t] = np.mean(data[max(0, t-avg_range):(t+1)])
        return avg

    
def saveGraphData(agents, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    i = 0
    while os.path.exists("{}/{}".format(directory, i)):
        i +=1
    os.mkdir("{}/{}".format(directory, i))
    for agent in agents:
        np.savetxt("{}/{}/scores.csv".format(directory, i), agent[1], delimiter=",")
        np.savetxt("{}/{}/timeSteps.csv".format(directory, i), agent[2], delimiter=",")
        np.savetxt("{}/{}/epsilon.csv".format(directory, i), agent[3], delimiter=",")

#TODO: function for generating graph and file name with hyperparameters, example settings and env details
#TODO: function for producing graphs from saved graphData