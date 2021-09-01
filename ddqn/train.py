import numpy as np
import time
from IPython.display import clear_output

def train(agent, timeSteps, env):
    score = 0
    done = True
    #Training loop
    finished_priming = False
    if agent[0].use_examples and finished_priming == False:
        print("Priming")
        for i in range(agent[0].primesteps):
            history = agent[0].prime()
            
            loss = history.history['loss']
            

            if i % 100 == 0:
                print("Priming: {}/{}  Loss:{}".format(i,agent[0].primesteps, loss))
        print("Finished priming")
        #agent[0].replay_add()
        finished_priming = True
            
    
    for i in range(timeSteps):
        
        if done:
            observation = env.reset()
            
        action = agent[0].choose_action(observation)
        
        observation_, reward, done, info = env.step(action)
        score += reward
        agent[0].remember(observation, action, reward, observation_, int(done))
        observation = observation_
        agent[0].learn()
    
        if done: 
            agent[1].append(score)
            agent[2].append(i+1)
            agent[3].append(agent[0].epsilon)
            

            avg_score = np.mean(agent[1][max(0, len(agent[1])-100):(len(agent[1])+1)])
            running_avg_score = np.mean(agent[1][max(0, len(agent[1])-10):(len(agent[1])+1)])
            average_score = np.mean(agent[1][max(0, len(agent[1])-100):(len(agent[1])+1)])
            print(agent[0].name , '-', 'Episode:', len(agent[1]),'Timestep:', i, '/', timeSteps, 'Score: %.2f' % score,' Running average: %.2f' %running_avg_score, 'Average: %.2f' %average_score)

            score = 0

    score = 0
    done = True

    avg_score = 0
    running_avg_score = 0
    timeInMins = time.process_time()/60
    print("Process time-", "{:.1f}".format(timeInMins), "minutes")