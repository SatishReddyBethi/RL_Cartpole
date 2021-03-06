import gym
import numpy as np
import math
import matplotlib.pyplot as plt

class My_QL():
    # For stats
    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

    def __init__(self,Environment,alpha = 1.0,revolving_decay = True, min_gamma = 0.1,min_epsilon = 0.1,MAX_STEPS = 300,MAX_EPISODES = 100,MAX_STATES = 10,buckets=(1, 1, 6, 12,),monitor=False,SHOW_EVERY=1000,STATS_EVERY = 10):
        self.env = Environment
        # self.gamma = gamma # Q learning rate
        self.min_gamma = min_gamma # learning rate
        self.alpha = alpha # TD error discount factor
        # self.epsilon = epsilon
        # self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon # minimum exploration rate
        self.MAX_STEPS = MAX_STEPS
        self.MAX_EPISODES = MAX_EPISODES
        self.MAX_STATES = MAX_STATES
        # self.Q = np.zeros((MAX_STATES,MAX_STATES,MAX_STATES,MAX_STATES,self.env.action_space.n)) # 10 x 10 x 10 x 10 x 2   
        self.Q = np.zeros(buckets + (self.env.action_space.n,)) # 1 x 1 x 6 x 12 x 2   
        self.buckets = buckets
        # print(self.Q.shape)
        # self.gamma = 0.9
        self.ada_divisor = 25 # only for development purposes
        self.SHOW_EVERY = SHOW_EVERY #Render every 100 episodes
        self.revolving_decay = revolving_decay
        self.STATS_EVERY = STATS_EVERY
        if MAX_STEPS is not None: 
            self.env._max_episode_steps = MAX_STEPS
        if monitor: 
            self.env = gym.wrappers.Monitor(self.env, 'tmp/cartpole-1', force=True) # record results for upload

    def run(self):
        self.episode_reward = 0
        counter = 0
        ep_batch = 0
        for e in range(self.MAX_EPISODES):

            if(self.get_epsilon(ep_batch) == self.min_epsilon):
                ep_batch = 0
            ep_batch +=1

            if not self.revolving_decay:
                ep_batch = e
            # Reset Environment
            obs = self.env.reset()
            
            # Setting epsilon and gamma
            epsilon = self.get_epsilon(ep_batch)
            gamma = self.get_gamma(ep_batch)
            
            #discritizing current state
            curr_state = self.discretize(self.env.reset())

            # Training
            for i in range(self.MAX_STEPS):
                done,curr_state = self.step(e,epsilon,gamma,curr_state)
                if done:
                    break

            if(e % self.SHOW_EVERY == 0):
                #discritizing current state
                curr_state = self.discretize(self.env.reset())
                for i in range(self.MAX_STEPS):
                    done,curr_state = self.step(e,0,0,curr_state)
                    self.env.render()
                    if done:
                        break

            # print(f"Episode: {e}, Steps: {i}, Epsilon: {epsilon:.2f}, {curr_state}")

            # Saving cost funcitons for visualization
            self.ep_rewards.append(self.episode_reward)
            if not e % self.STATS_EVERY:
                self.average_reward = sum(self.ep_rewards[-self.STATS_EVERY:])/self.STATS_EVERY
                self.aggr_ep_rewards['ep'].append(e)
                self.aggr_ep_rewards['avg'].append(self.average_reward)
                self.aggr_ep_rewards['max'].append(max(self.ep_rewards[-self.STATS_EVERY:]))
                self.aggr_ep_rewards['min'].append(min(self.ep_rewards[-self.STATS_EVERY:]))
                print(f'Episode: {e:>5d}, average reward: {self.average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

            if i == self.MAX_STEPS -1:
                counter +=1
            else:
                counter = 0

            if counter > 10:
                print(f"System Solved at Episode{e}")
                break
        
        # Render the Result for n times the training steps
        n = 1
        curr_state = self.discretize(self.env.reset())
        for i in range(self.MAX_STEPS*n):
                # Render Environment
                self.env.render()
                u = np.argmax(self.Q[curr_state])

                # Performing the selected action
                Obs, reward, done, info = self.env.step(u)

                # Discritizing new_state
                curr_state = self.discretize(Obs)

                if(done):
                    break

    def step(self,e,epsilon,gamma,curr_state):
        
        # getting random probability
        prob = np.random.rand()

        # implementing epsilon greedy policy
        if(prob < epsilon):
            u = self.env.action_space.sample()
        else:
            u = np.argmax(self.Q[curr_state])

        # Performing the selected action
        Obs, reward, done, info = self.env.step(u)

        # Getting reward for one episode
        self.episode_reward += reward  

        # Discritizing new_state
        new_state = self.discretize(Obs)

        # Updating Q-Table
        TD_error = reward + self.alpha * (np.max(self.Q[new_state])-self.Q[curr_state][u])
        self.Q[curr_state][u] = self.Q[curr_state][u]+ gamma * TD_error

        curr_state = new_state

        return done,new_state
        
    # --------------------------------------------------------------------------------------------------------------------------------------

    def discretize(self,observation):
        upper_bound = np.array([self.env.observation_space.high[0],0.5,self.env.observation_space.high[2],math.radians(50)])
        lower_bound = np.array([self.env.observation_space.low[0],-0.5,self.env.observation_space.low[2],-math.radians(50)])
        Total_range = upper_bound-lower_bound
        obs = np.amin([np.amax([observation,lower_bound],axis=0),upper_bound],axis=0)
        percentage = (obs-lower_bound)/Total_range
        index = np.round(percentage * (np.array(self.Q.shape[0:4])-1))
        state = index.astype(int)
        # print(observation,obs)
        return tuple(state.tolist())     
    
    # --------------------------------------------------------------------------------------------------------------------------------------
    # Choosing action based on epsilon-greedy policy
    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    # Updating Q-value of state-action pair based on the update equation
    def update_q(self, state_old, action, reward, state_new, gamma):
        self.Q[state_old][action] += gamma * (reward + self.alpha * np.max(self.Q[state_new]) - self.Q[state_old][action])

    # Adaptive learning of Exploration Rate
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    # Adaptive learning of Learning Rate
    def get_gamma(self, t):
        return max(self.min_gamma, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))
    # --------------------------------------------------------------------------------------------------------------------------------------
    
class Q_Learning():
    def __init__(self,Environment,alpha = 1.0,min_gamma = 0.1,min_epsilon = 0.1,MAX_STEPS = 300,MAX_EPISODES = 100,MAX_STATES = 10,buckets=(1, 1, 6, 12,),monitor=False):
        self.env = Environment
        # self.gamma = gamma # Q learning rate
        self.min_gamma = min_gamma # learning rate
        self.alpha = alpha # TD error discount factor
        # self.epsilon = epsilon
        # self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon # minimum exploration rate
        self.MAX_STEPS = MAX_STEPS
        self.MAX_EPISODES = MAX_EPISODES
        self.MAX_STATES = MAX_STATES
        self.bins = self.create_bins()
        # self.Q = np.zeros([self.MAX_STATES,self.env.observation_space.shape[0],self.env.action_space.n]) # 10 x 4 x 2   
        self.buckets = buckets # down-scaling feature space to discrete range
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))     
        # print(self.Q.shape)

        self.ada_divisor = 25 # only for development purposes
        if MAX_STEPS is not None: 
            self.env._max_episode_steps = MAX_STEPS
        if monitor: 
            self.env = gym.wrappers.Monitor(self.env, 'tmp/cartpole-1', force=True) # record results for upload
    #region myRun
    # def run(self):
    #     for e in range(self.MAX_EPISODES):
    #         # Reset Environment
    #         obs = self.env.reset()

    #         #Decaying epsilon
    #         self.epsilon = max(self.min_epsilon,self.epsilon - self.epsilon_decay * self.epsilon)
            
    #         #discritizing current state
    #         curr_state = self.discretize(self.env.reset()) # self.assign_bins(obs)

    #         for i in range(self.MAX_STEPS):
    #             # Render Environment
    #             self.env.render()
    #             # getting random probability
    #             prob = np.random.rand()

    #             # implementing epsilon greedy policy
    #             if(prob < self.epsilon):
    #                 u = self.env.action_space.sample()
    #             else:
    #                 u = np.argmax(self.Q[curr_state])

    #             # Performing the selected action
    #             Obs, reward, done, info = self.env.step(u)

    #             # Discritizing new_state
    #             new_state = self.discretize(Obs) # self.assign_bins(Obs)

    #             # Updating Q-Table
    #             TD_error = reward + self.alpha * (np.max(self.Q[new_state])-self.Q[curr_state][u])
    #             self.Q[curr_state] = self.Q[curr_state]+self.gamma * TD_error

    #             if done:
    #                 break

    #             # if done and i< self.MAX_ITERATIONS-1:
    #             #     reward = -300
    #             #     TD_error = reward + self.alpha * (np.max(self.Q[new_state])-self.Q[curr_state][u])
    #             #     self.Q[curr_state] = self.Q[curr_state]+self.gamma * TD_error
    #             #     break
    #             # else:
    #             #     TD_error = reward + self.alpha * (np.max(self.Q[new_state])-self.Q[curr_state][u])
    #             #     self.Q[curr_state] = self.Q[curr_state]+self.gamma * TD_error

    #         print(f"Episode {e},{i},{self.epsilon}")
    #endregion
    #region Descritize 1
    # --------------------------------------------------------------------------------------------------------------------------------------
    
    def create_bins(self):
        bins = np.zeros((4,self.MAX_STATES))
        bins[0] = np.linspace(-4.8,4.8,self.MAX_STATES)
        bins[1] = np.linspace(-5,5,self.MAX_STATES) # instead of going from -inf to inf we go from -5 to 5
        bins[2] = np.linspace(-0.418,0.418,self.MAX_STATES)
        bins[3] = np.linspace(-5,5,self.MAX_STATES)
        return bins

    def assign_bins(self,observation):
        state = np.zeros(4)
        for i in range(4):
            state[i]=np.digitize(observation[i],self.bins[i]) # state argument
        # print(state)
        return state.astype(int)       
    
    # --------------------------------------------------------------------------------------------------------------------------------------
    #endregion
    
    def discretize(self,obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    # Choosing action based on epsilon-greedy policy
    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    # Updating Q-value of state-action pair based on the update equation
    def update_q(self, state_old, action, reward, state_new, gamma):
        self.Q[state_old][action] += gamma * (reward + self.alpha * np.max(self.Q[state_new]) - self.Q[state_old][action])

    # Adaptive learning of Exploration Rate
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    # Adaptive learning of Learning Rate
    def get_gamma(self, t):
        return max(self.min_gamma, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def run(self):

        for e in range(self.MAX_EPISODES):
            # As states are continuous, discretize them into buckets
            current_state = self.discretize(self.env.reset())
            # Get adaptive learning alpha and epsilon decayed over time
            gamma = self.get_gamma(e)
            epsilon = self.get_epsilon(e)
            done = False
            i = 0

            while not done:
                # Render environment
                self.env.render()

                # Choose action according to greedy policy and take it
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)

                # Update Q-Table
                self.update_q(current_state, action, reward, new_state, gamma)
                current_state = new_state
                i += 1
            
            print(f"Episode {e},{i},{epsilon},{current_state}")
    # --------------------------------------------------------------------------------------------------------------------------------------
    
            

env = gym.make('CartPole-v0')
Max_Episodes = 10_000
Q = My_QL(env,min_epsilon = 0.1,MAX_STEPS = 300,MAX_EPISODES = Max_Episodes,MAX_STATES = 5,SHOW_EVERY = 100,STATS_EVERY = 10,revolving_decay = True, buckets=(1,1,6,12))
Q.run()
Q.env.close()

plt.plot(Q.aggr_ep_rewards['ep'], Q.aggr_ep_rewards['avg'], label="average rewards")
plt.plot(Q.aggr_ep_rewards['ep'], Q.aggr_ep_rewards['max'], label="max rewards")
plt.plot(Q.aggr_ep_rewards['ep'], Q.aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.grid(True)
plt.savefig(f"Reward_vs_Episodes_{Max_Episodes}.png")
plt.show()