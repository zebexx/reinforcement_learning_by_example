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


        

folder = "graphData/12"
filename = "Test_graph.png"
plotFromData(folder, filename)