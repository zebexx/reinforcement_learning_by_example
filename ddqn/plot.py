import matplotlib.pyplot as plt
import numpy as np
import os
from utils import plot

def plotFromData(folder, graphName):
    agents = []
    for filename in os.listdir(folder):
        tmpdirectory = "{}/{}".format(folder, filename)
        scores = np.loadtxt("{}/scores.csv".format(tmpdirectory), delimiter=",")
        ts = np.loadtxt("{}/timeSteps.csv".format(tmpdirectory), delimiter=",")
        ep = np.loadtxt("{}/epsilon.csv".format(tmpdirectory), delimiter=",")
        agents.append([filename, scores, ts, ep, filename])
    
    
    plot(agents, graphName)


        

folder = "graphData\CartPole-v1_ddqn_200000ts_Normal_[500, 500, 200]_bs-32_lr-0.001_g-0.99_edr-0.999_em-0.01_rt-500"
filename = "Test_graph.png"
plotFromData(folder, filename)