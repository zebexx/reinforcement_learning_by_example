
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, load_model
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
        # store one hot encoding of actions, if appropriate
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
        #self.state_memory = self.state_memory[:self.mem_counter]
        np.savetxt("history/state_history.csv", self.state_memory[:self.mem_counter], delimiter=",")

        #self.action_memory = self.action_memory[:self.mem_counter]
        np.savetxt("history/action_history.csv", self.action_memory[:self.mem_counter], delimiter=",")

        #self.reward_memory = self.reward_memory[:self.mem_counter]
        np.savetxt("history/reward_history.csv", self.reward_memory[:self.mem_counter], delimiter=",")

        #self.next_state_memory = self.next_state_memory[:self.mem_counter]
        np.savetxt("history/next_state_history.csv", self.next_state_memory[:self.mem_counter], delimiter=",")

        #self.terminal_memory = self.terminal_memory[:self.mem_counter]
        np.savetxt("history/terminal_history.csv", self.terminal_memory[:self.mem_counter], delimiter=",")

        

    
class Example_Buffer(object):
    def __init__(self, episode_choice=None, discrete=False):
        self.episode_choice = episode_choice
        self.discrete = discrete
        self.dtype = np.int8 if self.discrete else np.float32

        self.state_memory = np.genfromtxt("example_data/state_history.csv", delimiter=",")
        self.action_memory = np.genfromtxt("example_data/action_history.csv", delimiter=",", dtype=self.dtype)
        self.reward_memory = np.genfromtxt("example_data/reward_history.csv", delimiter=",")
        self.next_state_memory = np.genfromtxt("example_data/next_state_history.csv", delimiter=",")
        self.terminal_memory = np.genfromtxt("example_data/terminal_history.csv", delimiter=",", dtype=np.float32)

        
        
        episode_indexes = []
        
        for i in range(len(self.terminal_memory)):
            if self.terminal_memory[i] == 0 and i != len(self.terminal_memory)-1:
                episode_indexes.append(i+1)
            
        
        self.episode_indexes = episode_indexes
        
        self.num_episodes = len(self.episode_choice)
        
        

        if self.episode_choice != None:
            episode_state_memory = np.split(self.state_memory, self.episode_indexes)
            episode_action_memory = np.split(self.action_memory, self.episode_indexes)
            episode_reward_memory= np.split(self.reward_memory, self.episode_indexes)
            episode_next_state_memory= np.split(self.next_state_memory, self.episode_indexes)
            episode_terminal_memory = np.split(self.terminal_memory, self.episode_indexes)

            self.state_memory = np.concatenate([episode_state_memory[self.episode_choice[i]] for i in range(len(self.episode_choice))])
            self.action_memory = np.concatenate([episode_action_memory[self.episode_choice[i]] for i in range(len(self.episode_choice))])
            self.reward_memory = np.concatenate([episode_reward_memory[self.episode_choice[i]] for i in range(len(self.episode_choice))])
            self.state_memory = np.concatenate([episode_next_state_memory[self.episode_choice[i]] for i in range(len(self.episode_choice))])
            self.terminal_memory = np.concatenate([episode_terminal_memory[self.episode_choice[i]] for i in range(len(self.episode_choice))])


        self.num_examples = len(self.action_memory)
        self.mem_counter = 0
        self.episode_counter = 0

        episode_indexes1 = []

        for i in range(len(self.terminal_memory)):
            if self.terminal_memory[i] == 0 and i != len(self.terminal_memory)-1:
                episode_indexes1.append(i+1)
            
        
        self.episode_indexes = episode_indexes1

        

        

    def sample_example(self):

        states = self.state_memory[self.mem_counter]
        actions = self.action_memory[self.mem_counter]
        rewards = self.reward_memory[self.mem_counter]
        states_ = self.next_state_memory[self.mem_counter]
        terminal = self.terminal_memory[self.mem_counter]

        self.mem_counter += 1

        return states, actions, rewards, states_, terminal

    def example_reset(self):
        return self.state_memory[self.mem_counter]

    def example_step(self):
        actions = self.action_memory[self.mem_counter]
        states_ = self.next_state_memory[self.mem_counter]
        rewards = self.reward_memory[self.mem_counter]
        terminal = self.terminal_memory[self.mem_counter]
        #print(self.mem_counter)

        self.mem_counter += 1
        if terminal == 0:
            #print("done")
            self.episode_counter += 1

        return actions, states_, rewards, terminal, {}

            



def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    
    model = Sequential([
                Dense(fc1_dims, input_shape=(input_dims,)),
                Activation('relu'),
                Dense(fc2_dims),
                Activation('relu'),
                Dense(n_actions)])

    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    return model


class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.0001,  epsilon_end=0.1,
                 mem_size=1000000, fname='ddqn_model.h5', replace_target=100, use_examples=False):
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
            self.example_memory = Example_Buffer(discrete=True, episode_choice=[39,51,54,61,62,71,88])
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)
        self.q_eval.summary()
        self.q_target = build_dqn(alpha, n_actions, input_dims, 256, 256)
        

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        #if else for example data
        #function to choose similar states
        return action

    def learn(self):
        learning_counter = 0
        if self.memory.mem_counter > self.batch_size and learning_counter % 4 == 0:
            state, action, reward, new_state, done = \
                                          self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)
            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > \
                           self.epsilon_min else self.epsilon_min
            if self.memory.mem_counter % self.replace_target == 0:
                self.update_network_parameters()

        

    def update_network_parameters(self):
        for i in range(len(self.q_target.layers)):
            self.q_target.get_layer(index=i).set_weights(self.q_eval.get_layer(index=i).get_weights())
        

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        if self.epsilon == 0.0:
            self.update_network_parameters()