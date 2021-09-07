import matplotlib.pyplot as plt
import numpy as np
import os


def plot(agents, filename, lines=None):


    fig=plt.figure()
    
    for agent in agents:
        avg = average_over(agent[1], 100)
        plt.plot(agent[2], avg, label="{}".format(agent[4]))
    
    
    plt.xlabel("Timesteps", color="C0")
    plt.ylabel("Score", color="C0")
    plt.tick_params(axis='x', colors="C0")
    plt.tick_params(axis='y', colors="C0")

    plt.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2,
            borderaxespad=0, frameon=False, fontsize='x-small')

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
        tmpname = "{}_{}".format(name, i)
    os.mkdir("{}/{}".format(directory, tmpname))
    j = 0
    for agent in agents:
        folder = "Agent{}".format(j)
        os.mkdir("{}/{}/{}".format(directory, tmpname, folder))

        np.savetxt("{}/{}/{}/scores.csv".format(directory, tmpname, folder), agent[1], delimiter=",")
        np.savetxt("{}/{}/{}/timeSteps.csv".format(directory, tmpname, folder), agent[2], delimiter=",")
        np.savetxt("{}/{}/{}/epsilon.csv".format(directory, tmpname, folder), agent[3], delimiter=",")
        j+=1
        



