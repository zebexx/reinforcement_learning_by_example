import matplotlib.pyplot as plt
import numpy as np
import os
from utils import plot

def plotFromData(folder, graphName):
    agents = []
    for filename in os.listdir(folder):
        tmpdirectory = "{}/{}".format(folder, filename)
        scores = np.genfromtxt("{}/scores.csv".format(tmpdirectory), delimiter=",")
        ts = np.genfromtxt("{}/timeSteps.csv".format(tmpdirectory), delimiter=",")
        ep = np.genfromtxt("{}/epsilon.csv".format(tmpdirectory), delimiter=",")
        agents.append([filename, scores, ts, ep, filename])
    
    
    plot(agents, graphName)


        

folder = "graphData/11"
filename = "Test_graph.png"
plotFromData(folder, filename)