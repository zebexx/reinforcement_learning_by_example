import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

import numpy as np
import os
from utils import plot
from ddqn import Example_Buffer

def plotFromData(folder, graphName):
    agents = []
    for filename in os.listdir(folder):
        tmpdirectory = "{}/{}".format(folder, filename)
        scores = np.loadtxt("{}/scores.csv".format(tmpdirectory), delimiter=",")
        ts = np.loadtxt("{}/timeSteps.csv".format(tmpdirectory), delimiter=",")
        ep = np.loadtxt("{}/epsilon.csv".format(tmpdirectory), delimiter=",")
        agents.append([filename, scores, ts, ep, filename])
    
    
    plot(agents, graphName)

def state_space_histogram(data, filteredData, graphName):
    font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 6}

    plt.rc('font', **font)

    fig, axs = plt.subplots(2,2)

    data = data[:len(filteredData)-1]

    state1 = []
    state2 = []
    state3 = []
    state4 = []
    for d in data:
        state1.append(d[0])
        state2.append(d[1])
        state3.append(d[2])
        state4.append(d[3])


    Fstate1 = []
    Fstate2 = []
    Fstate3 = []
    Fstate4 = []
    for d in filteredData:
        Fstate1.append(d[0])
        Fstate2.append(d[1])
        Fstate3.append(d[2])
        Fstate4.append(d[3])

    bins = 100
    axs[0,0].hist(state1, bins=bins, alpha=0.5, label="Average Episodes")
    axs[0,0].hist(Fstate1, bins=bins, alpha=0.5, label="Perfect Episodes")
    axs[0,0].legend(loc="upper right", fontsize="x-small")
    axs[0,0].set_title("Cart Position")

    axs[0,1].hist(state2, bins=bins, alpha=0.5, label="Average Episodes")
    axs[0,1].hist(Fstate2, bins=bins, alpha=0.5, label="Perfect Episodes")
    axs[0,1].legend(loc="upper right", fontsize="x-small")
    axs[0,1].set_title("Cart Velocity")

    axs[1,0].hist(state3, bins=bins, alpha=0.5, label="Average Episodes")
    axs[1,0].hist(Fstate3, bins=bins, alpha=0.5, label="Perfect Episodes")
    axs[1,0].legend(loc="upper right", fontsize="x-small")
    axs[1,0].set_title("Pole Angle")

    axs[1,1].hist(state4, bins=bins, alpha=0.5, label="Average Episodes")
    axs[1,1].hist(Fstate4, bins=bins, alpha=0.5, label="Perfect Episodes")
    axs[1,1].legend(loc="upper right", fontsize="x-small")
    axs[1,1].set_title("Pole Velocity At Tip")  

    plt.savefig(graphName) 
        

folder = "graphData\CartPole-v1_exploration_vs_no_exploration"
filename = "CartPole-v1_exploration_vs_no_exploration.png"
#plotFromData(folder, filename)

data = Example_Buffer(location="example_data/CartPole-v1", discrete=True, episode_range=[0,500,10000])
Fdata = Example_Buffer(location="example_data/CartPole-v1", discrete=True, episode_range=[500,500,200])

state_space_histogram(data.state_memory, Fdata.state_memory, "CartPole-v1_state_space_histogram.png")

