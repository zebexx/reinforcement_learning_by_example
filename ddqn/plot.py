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


        

folder = "graphData\MountainCar-v0_ddqn_400000ts_Normal_[-110, 0, 200]_bs-32_bu-4_lr-0.00025_g-0.99_edr-0.9999_em-0.1_rt-500"
filename = "Test_graph.png"
plotFromData(folder, filename)