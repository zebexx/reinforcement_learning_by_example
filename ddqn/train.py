import numpy as np
import time

def train(agent, timeSteps, env):
    score = 0
    done = True
    #Training loop
    for i in range(timeSteps):
        
        if agent[0].use_examples and agent[0].example_memory.mem_counter+1 <= agent[0].example_memory.num_examples and agent[0].epsilon <= agent[0].epsilon_min :
            
            if agent[0].example_memory.mem_counter == 0:
                score =0
            
            if done:
                observation = agent[0].example_memory.example_reset()

            
            action, observation_, reward, xdone, info = agent[0].example_memory.example_step()
            done = not xdone
            
            score+=reward
            agent[0].remember(observation, action, reward, observation_, int(done))
            observation = observation_
            agent[0].learn()
        else:
            
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
            print(agent[0].name , '-', 'Episode:', len(agent[1]),'Timestep:', i, '/', timeSteps, 'Score: %.2f' % score,' Running average: %.2f' %running_avg_score)

            score = 0

    score = 0
    done = True

    avg_score = 0
    running_avg_score = 0
    timeInMins = time.process_time()/60
    print("Process time-", "{:.1f}".format(timeInMins), "minutes")