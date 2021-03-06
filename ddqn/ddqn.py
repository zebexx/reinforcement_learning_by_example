
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, load_model, save_model, Model
from tensorflow.keras.optimizers import Adam

import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_counter = 0
        self.discrete = discrete
        self.state_memory = np.empty((self.mem_size, input_shape))
        self.next_state_memory = np.empty((self.mem_size, input_shape))
        self.dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.empty((self.mem_size, n_actions), dtype=self.dtype)
        self.reward_memory = np.empty(self.mem_size)
        self.terminal_memory = np.empty(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = state_
        
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
    
    def save_memory(self):
        
        np.savetxt("history/state_history.csv", self.state_memory[:self.mem_counter], delimiter=",")
        np.savetxt("history/action_history.csv", self.action_memory[:self.mem_counter], delimiter=",")
        np.savetxt("history/reward_history.csv", self.reward_memory[:self.mem_counter], delimiter=",")
        np.savetxt("history/next_state_history.csv", self.next_state_memory[:self.mem_counter], delimiter=",")
        np.savetxt("history/terminal_history.csv", self.terminal_memory[:self.mem_counter], delimiter=",")

        

    
class Example_Buffer(object):
    def __init__(self, location, episode_range, discrete=False):
        self.episode_range = episode_range
        self.discrete = discrete
        self.dtype = np.int8 if self.discrete else np.float32

        self.state_memory = np.genfromtxt("{}/state_history.csv".format(location), delimiter=",")
        self.action_memory = np.genfromtxt("{}/action_history.csv".format(location), delimiter=",", dtype=self.dtype)
        self.reward_memory = np.genfromtxt("{}/reward_history.csv".format(location), delimiter=",")
        self.next_state_memory = np.genfromtxt("{}/next_state_history.csv".format(location), delimiter=",")
        self.terminal_memory = np.genfromtxt("{}/terminal_history.csv".format(location), delimiter=",", dtype=np.float32)

        
        
        self.episode_indexes = [0]
        
        for i in range(len(self.terminal_memory)):
            if self.terminal_memory[i] == 0:
                self.episode_indexes.append(i+1)
            
            
        
        self.episode_scores = self.example_parser()
        

        self.episode_choice = []
        for i in range(len(self.episode_scores)):
            if self.episode_scores[i] >= self.episode_range[0] and self.episode_scores[i] <= self.episode_range[1]:
                self.episode_choice.append(i)
            if len(self.episode_choice) >= self.episode_range[2]:
                break
        

        
        
        print("Loading {} Episodes...".format(len(self.episode_choice)))
        
        
        episode_state_memory = np.split(self.state_memory, self.episode_indexes[1:])
        episode_action_memory = np.split(self.action_memory, self.episode_indexes[1:])
        episode_reward_memory= np.split(self.reward_memory, self.episode_indexes[1:])
        episode_next_state_memory= np.split(self.next_state_memory, self.episode_indexes[1:])
        episode_terminal_memory = np.split(self.terminal_memory, self.episode_indexes[1:])

        self.state_memory = np.concatenate([episode_state_memory[self.episode_choice[i]] for i in range(len(self.episode_choice))])
        self.action_memory = np.concatenate([episode_action_memory[self.episode_choice[i]] for i in range(len(self.episode_choice))])
        self.reward_memory = np.concatenate([episode_reward_memory[self.episode_choice[i]] for i in range(len(self.episode_choice))])
        self.next_state_memory = np.concatenate([episode_next_state_memory[self.episode_choice[i]] for i in range(len(self.episode_choice))])
        self.terminal_memory = np.concatenate([episode_terminal_memory[self.episode_choice[i]] for i in range(len(self.episode_choice))])

        self.num_examples = len(self.action_memory)
        self.mem_counter = len(self.action_memory)
        self.episode_counter = 0

        self.episode_indexes = []

        for i in range(len(self.terminal_memory)):
            if self.terminal_memory[i] == 0:
                self.episode_indexes.append(i+1)
           
            
        self.num_episodes = len(self.episode_indexes)+1

        self.episode_scores = self.example_parser()
        
    def sample_example(self, batch_size):

        max_mem = self.mem_counter
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    def example_reset(self):
        return self.state_memory[self.mem_counter]

    def example_step(self):
        actions = self.action_memory[self.mem_counter]
        states_ = self.next_state_memory[self.mem_counter]
        rewards = self.reward_memory[self.mem_counter]
        terminal = self.terminal_memory[self.mem_counter]
        

        self.mem_counter += 1
        if terminal == 0:
            
            self.episode_counter += 1

        return actions, states_, rewards, terminal, {}
    
    def example_parser(self):
        episode_scores = []
        for i in range(1, len(self.episode_indexes)):
            episode_scores.append(np.sum(self.reward_memory[range(self.episode_indexes[i-1], self.episode_indexes[i])]))

        return episode_scores
    
    def choice_score_range(self, range, episode_scores):
        episode_choice = []
        for i in range(len(episode_scores)):
            if episode_scores[i] >= range[0] and episode_scores[i] <= range[1]:
                episode_choice.append(i)
        return episode_choice

    def replay_add(self, agent):
        for i in range(len(self.action_memory)):
            agent.memory.store_transition(self.state_memory[i], self.action_memory[i], self.reward_memory[i], self.next_state_memory[i], self.terminal_memory[i])
        print("Added prime data to experience replay")

    def analyse_state_space(self):
        minMax = []
        for i in self.state_memory:
            if len(minMax) == 0:
                for y in range(len(i)):
                    minMax.append([i[y], i[y]])
            for x in range(len(i)):
                if i[x] < minMax[x][0]:
                    minMax[x][0] = i[x]
                elif i[x] > minMax[x][1]:
                    minMax[x][1] = i[x]
        
        return minMax
                



# Heavily influenced by: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/ddqn_keras.py
class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, primesteps=0, episode_range=[0,0,0], example_location=None, epsilon_dec=0.9999,  epsilon_end=0.0001,
                 mem_size=1000000, fname='ddqn_model.h5', replace_target=500, use_examples=False):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,discrete=True)
        self.use_examples = use_examples
        if self.use_examples:
            self.example_memory = Example_Buffer(location=example_location, discrete=True, episode_range=episode_range)
            self.name = "Example Agent"
            self.primesteps = primesteps
        else:
            self.name = "Normal Agent"
        self.q_eval = self.build_dqn(alpha, n_actions, input_dims, 256, 256)
        
        self.q_target = self.build_dqn(alpha, n_actions, input_dims, 256, 256)

        self.learning_counter = 0
        
        self.primestep_counter = 0

        
        
    def build_dqn(self, lr, n_actions, input_dims, fc1_dims, fc2_dims):

        model = Sequential([
                    Dense(fc1_dims, input_shape=(input_dims,)),
                    Activation('relu'),
                    Dense(fc2_dims),
                    Activation('relu'),
                    Dense(n_actions)])

        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

        print(model.summary())

        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def replay_add(self):
        for i in range(len(self.example_memory.action_memory)):
           self.remember(self.example_memory.state_memory[i], self.example_memory.action_memory[i], self.example_memory.reward_memory[i], \
            self.example_memory.next_state_memory[i], self.example_memory.terminal_memory[i])
        print("Added prime data to experience replay")

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
            
        else:
            actions = self.q_eval.predict(state, use_multiprocessing=True) 
            action = np.argmax(actions)
        return action

    def learn(self):          
        if self.memory.mem_counter > self.batch_size:
            state, action, reward, new_state, done = \
                                        self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state, use_multiprocessing=True)
            q_eval = self.q_eval.predict(new_state, use_multiprocessing=True)
            q_pred = self.q_eval.predict(state, use_multiprocessing=True)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            loss = self.q_eval.fit(state, q_target, verbose=0, use_multiprocessing=True)

            self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > \
                        self.epsilon_min else self.epsilon_min
            if self.memory.mem_counter % self.replace_target == 0:
                self.update_network_parameters()
            
            return loss

    def prime(self):
        

        state, action, reward, new_state, done = self.example_memory.sample_example(self.batch_size)


        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        q_next = self.q_target.predict(new_state, use_multiprocessing=True)
        q_eval = self.q_eval.predict(new_state, use_multiprocessing=True)
        q_pred = self.q_eval.predict(state, use_multiprocessing=True)

        max_actions = np.argmax(q_eval, axis=1)

        q_target = q_pred

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + self.gamma*q_next[batch_index, max_actions.astype(int)]*done

        loss = self.q_eval.fit(state, q_target, verbose=0, use_multiprocessing=True)

        if self.primestep_counter % self.replace_target == 0:
                self.update_network_parameters()
        self.primestep_counter += 1

        return loss
        


    def update_network_parameters(self):
        for i in range(len(self.q_target.layers)):
            self.q_target.get_layer(index=i).set_weights(self.q_eval.get_layer(index=i).get_weights())
        

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        if self.epsilon == 0.0:
            self.update_network_parameters()