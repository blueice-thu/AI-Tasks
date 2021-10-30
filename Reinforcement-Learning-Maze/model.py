from envs import Env
import numpy as np

class Direct:
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

class RLModel(object):
    def __init__(self, env: Env):
        self.n_epoch = 100

        self.n_states = env.m * env.n
        self.actions = [ Direct.LEFT, Direct.RIGHT, Direct.UP, Direct.DOWN ]

        self.epsilon = 0.9  # epsilon-greedy
        self.learning_rate = 0.1
        self.gamma = 0.9  # discount

        self.current_score = 0.0
        self.current_step = 0
        self.current_epoch = 0
        
        self.q_table = np.zeros([self.n_states, len(self.actions)])
        self.current_state = env.startPos[0] * env.n + env.startPos[1]
    
    def chooseAction(self, state):
        state_actions = self.q_table[state][:]
        if np.random.uniform() > self.epsilon or state_actions.all() == False:
            return np.random.choice(self.actions)
        else:
            return state_actions.argmax()

class QLearningModel(RLModel):
    def __init__(self, env: Env):
        super(QLearningModel, self).__init__(env)

    def learn(self, state, action, reward, next_state, done):
        self.current_score = reward + self.current_score * self.gamma
        q_predict = self.q_table[state, action]
        
        if not done:
            q_target = reward + self.gamma * self.q_table[next_state, :].max()
        else:
            q_target = reward

            print("Epoch: %d / %d, used step: %d, score = %.2f" % (self.current_epoch, self.n_epoch, self.current_step, self.current_score))
            self.current_epoch += 1
            self.current_step = 0
            self.current_score = 0.0

        self.q_table[state, action] += self.learning_rate * (q_target - q_predict)
        self.current_state = next_state
        self.current_step += 1

class SarsaModel(RLModel):
    def __init__(self, env: Env):
        super(SarsaModel, self).__init__(env)
        self.next_action = None
    
    def learn(self, state, action, reward, next_state, next_action, done):
        self.current_score = reward + self.current_score * self.gamma
        q_predict = self.q_table[state, action]

        if not done:
            q_target = reward + self.gamma * self.q_table[next_state, next_action]
        else:
            q_target = reward

            print("Epoch: %d / %d, used step: %d, score = %.2f" % (self.current_epoch, self.n_epoch, self.current_step, self.current_score))
            self.current_epoch += 1
            self.current_step = 0
            self.current_score = 0.0

        self.q_table[state, action] += self.learning_rate * (q_target - q_predict)
        self.current_state = next_state
        self.current_step += 1