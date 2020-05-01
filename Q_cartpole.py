import gym
import numpy as np

class Q_Learning():
    def __init__(self,Environment,alpha,gamma,epsilon,epsilon_decay,min_epsilon = 0.1,MAX_ITERATIONS = 300,MAX_EPISODES = 100,MAX_STATES = 10):
        self.env = Environment
        self.gamma = gamma # Q learning rate
        self.alpha = alpha # TD error discount factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.MAX_EPISODES = MAX_EPISODES
        self.MAX_STATES = MAX_STATES
        self.bins = self.create_bins()
        self.Q = np.zeros([self.MAX_STATES,self.env.observation_space.shape[0],self.env.action_space.n]) # 10 x 4 x 2        
        # print(self.Q.shape)

    def create_bins(self):
        bins = np.zeros((4,self.MAX_STATES))
        bins[0] = np.linspace(-4.8,4.8,self.MAX_STATES)
        bins[1] = np.linspace(-5,5,self.MAX_STATES) # instead of going from -inf to inf we go from -5 to 5
        bins[2] = np.linspace(-41.8,41.8,self.MAX_STATES)
        bins[3] = np.linspace(-5,5,self.MAX_STATES)
        return bins

    def assign_bins(self,observation):
        state = np.zeros(4)
        for i in range(4):
            state[i]=np.digitize(observation[i],self.bins[i]) # state argument
        # print(state)
        return state.astype(int)        
    
    def run(self):
        for e in range(self.MAX_EPISODES):
            # Reset Environment
            obs = self.env.reset()

            #Decaying epsilon
            self.epsilon = min(self.min_epsilon,self.epsilon - self.epsilon_decay * self.epsilon)
            
            #discritizing current state
            curr_state = self.assign_bins(obs)

            for i in range(self.MAX_ITERATIONS):
                # Render Environment
                self.env.render()
                # getting random probability
                prob = np.random.rand()

                # implementing epsilon greedy policy
                if(prob < self.epsilon):
                    u = self.env.action_space.sample()
                else:
                    u = np.argmax(self.Q[curr_state])

                # Performing the selected action
                Obs, reward, done, info = self.env.step(u)

                # Discritizing new_state
                new_state = self.assign_bins(Obs)

                # Updating Q-Table
                TD_error = reward + self.alpha * (np.max(self.Q[new_state])-self.Q[curr_state][u])
                self.Q[curr_state] = self.Q[curr_state]+self.gamma * TD_error

                if done:
                    break

            print(f"Episode {e},{i},{self.epsilon}")
            

env = gym.make('CartPole-v0')
Q = Q_Learning(env,0.9,1,1,0.01,MAX_ITERATIONS = 300,MAX_EPISODES = 100,MAX_STATES = 10)
Q.run()
Q.env.close()
print(Q.Q)