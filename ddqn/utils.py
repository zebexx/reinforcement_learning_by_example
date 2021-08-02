import matplotlib.pyplot as plt
import numpy as np
import os


def plot(agents, filename, lines=None):


    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)
    for agent in agents:

        ax.plot(agent[2], agent[3], color="m", linewidth=0.5)
        running_avg = average_over(agent[1], 10)
        avg = average_over(agent[1], 100)
        ax2.plot(agent[2], running_avg, label="{} 10 ep average".format(agent[4]), linewidth=0.5)
        ax2.plot(agent[2], avg, label="{} 100 ep average".format(agent[4]))
    
    
    ax.set_xlabel("Timesteps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

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

    
def saveGraphData(agents, directory, name):
    if not os.path.exists(directory):
        os.mkdir(directory)
    tmpname = name
    i = 0
    while os.path.exists("{}/{}".format(directory, tmpname)):
        i +=1
        tmpname = name + "_"+ i
    os.mkdir("{}/{}".format(directory, tmpname))
    j = 0
    for agent in agents:
        np.savetxt("{}/{}/{}scores.csv".format(directory, tmpname, "Agent{}_".format(j)), agent[1], delimiter=",")
        np.savetxt("{}/{}/{}timeSteps.csv".format(directory, tmpname,"Agent{}_".format(j)), agent[2], delimiter=",")
        np.savetxt("{}/{}/{}epsilon.csv".format(directory, tmpname, "Agent{}_".format(j)), agent[3], delimiter=",")
        j +=1


#TODO: function for producing graphs from saved graphData