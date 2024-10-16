import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import csv
from gym.utils import seeding


class NetworkSlicingEnv(gym.Env):
    def __init__(self,
                 max_steps=1000,
                 total_bandwidth=200,
                 aver_req=[0.5, 1.0],
                 users=[10, 100],
                 requirement_interval=[40, 80],
                 lambda_throughput_embb=5.0,
                 lambda_delay_embb=-0.5,
                 lambda_violation_embb=-0.5,
                 lambda_throughput_urllc=0.5,
                 lambda_delay_urllc=-5.0,
                 lambda_violation_urllc=-0.5):
        
        super(NetworkSlicingEnv, self).__init__()

        # Number of slices: 2 URLLC and 1 eMBB
        self.slice_types = ['URLLC', 'URLLC', 'eMBB']  # Slice types
        self.num_slices = len(self.slice_types)
        self.discrete_action = [0.0, 0.25, 0.5, 0.75, 1.0]  # Allocation options
        self.n_discrete = len(self.discrete_action)
        
        # Fixed number of time steps per episode
        self.max_steps = max_steps
        self.current_step = 0

        # Action space: Allocate resources to 3 slices
        self.action_space = spaces.Discrete(self.n_discrete ** self.num_slices)

        # Users and requirements configuration
        self.users_min = users[0]
        self.users_max = users[1]
        self.aver_req_min = aver_req[0]
        self.aver_req_max = aver_req[1]
        self.total_bandwidth = total_bandwidth
        self.emmb_coe = 2.0
        self.is_recording_initialized = False

        # Observation space: Separate users, requirements, and metrics for each slice
        self.observation_space = spaces.Dict({
            'users_1': spaces.Box(low=self.users_min, high=self.users_max, shape=(), dtype=np.int64),
            'users_2': spaces.Box(low=self.users_min, high=self.users_max, shape=(), dtype=np.int64),
            'users_3': spaces.Box(low=self.users_min, high=self.users_max, shape=(), dtype=np.int64),
            'requirements_1': spaces.Box(low=0, high=self.aver_req_max * self.users_max, shape=(), dtype=np.float64),
            'requirements_2': spaces.Box(low=0, high=self.aver_req_max * self.users_max, shape=(), dtype=np.float64),
            'requirements_3': spaces.Box(low=0, high=self.aver_req_max * self.users_max * self.emmb_coe, shape=(), dtype=np.float64),
            'log_throughput_1': spaces.Box(low=0, high=self.total_bandwidth, shape=(), dtype=np.float64),
            'log_throughput_2': spaces.Box(low=0, high=self.total_bandwidth, shape=(), dtype=np.float64),
            'log_throughput_3': spaces.Box(low=0, high=self.total_bandwidth, shape=(), dtype=np.float64),
            'log_delay_1': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float64),
            'log_delay_2': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float64),
            'log_delay_3': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float64),
            'log_violation_1': spaces.Box(low=0, high=self.aver_req_max * self.users_max, shape=(), dtype=np.float64),
            'log_violation_2': spaces.Box(low=0, high=self.aver_req_max * self.users_max, shape=(), dtype=np.float64),
            'log_violation_3': spaces.Box(low=0, high=self.aver_req_max * self.users_max * self.emmb_coe, shape=(), dtype=np.float64)
        })
        
        # Reward weights for eMBB
        self.lambda_throughput_embb = lambda_throughput_embb
        self.lambda_delay_embb = lambda_delay_embb
        self.lambda_violation_embb = lambda_violation_embb

        # Reward weights for URLLC
        self.lambda_throughput_urllc = lambda_throughput_urllc
        self.lambda_delay_urllc = lambda_delay_urllc
        self.lambda_violation_urllc = lambda_violation_urllc

        # Requirement interval
        self.requirement_interval_min = requirement_interval[0]
        self.requirement_interval_max = requirement_interval[1]

        self.reset()

        self.users_history = []
        self.requirements_history = []

        self.np_random = None
        self.seed()

    def reset(self):
        # Reset step count
        self.current_step = 0
        self.interval_step = 1
        
        # Randomly initialize users for each slice
        users = self.np_random.integers(self.users_min, self.users_max, size=(self.num_slices,))

        # Randomly initialize total requirements for each slice based on the number of users
        self.state = {
            'users_1': users[0],
            'users_2': users[1],
            'users_3': users[2],
            'requirements_1': users[0] * self.np_random.uniform(self.aver_req_min, self.aver_req_max),
            'requirements_2': users[1] * self.np_random.uniform(self.aver_req_min, self.aver_req_max),
            'requirements_3': users[2] * self.np_random.uniform(self.aver_req_min * self.emmb_coe, self.aver_req_max * self.emmb_coe),
            'log_throughput_1': 0.0,
            'log_throughput_2': 0.0,
            'log_throughput_3': 0.0,
            'log_delay_1': 0.0,
            'log_delay_2': 0.0,
            'log_delay_3': 0.0,
            'log_violation_1': 0.0,
            'log_violation_2': 0.0,
            'log_violation_3': 0.0
        }
        self.requirement_interval = self.np_random.integers(self.requirement_interval_min, self.requirement_interval_max)

        return self.get_observation()

    def get_observation(self):
        # Returns the current observation, including users, requirements, and log metrics
        return {
            'users_1': self.state['users_1'],
            'users_2': self.state['users_2'],
            'users_3': self.state['users_3'],
            'requirements_1': self.state['requirements_1'],
            'requirements_2': self.state['requirements_2'],
            'requirements_3': self.state['requirements_3'],
            'log_throughput_1': self.state['log_throughput_1'],
            'log_throughput_2': self.state['log_throughput_2'],
            'log_throughput_3': self.state['log_throughput_3'],
            'log_delay_1': self.state['log_delay_1'],
            'log_delay_2': self.state['log_delay_2'],
            'log_delay_3': self.state['log_delay_3'],
            'log_violation_1': self.state['log_violation_1'],
            'log_violation_2': self.state['log_violation_2'],
            'log_violation_3': self.state['log_violation_3']
        }

    def step(self, action):
        if self.state['requirements_1'] > 100:
            print('Out of range!!!')

        action = self.onehot2action(action)
        # Normalize actions so they sum to the total available bandwidth
        allocated_resources = action / np.sum(action) * self.total_bandwidth

        # Record users and requirements at each step
        self.users_history.append([self.state['users_1'], self.state['users_2'], self.state['users_3']])
        self.requirements_history.append([self.state['requirements_1'], self.state['requirements_2'], self.state['requirements_3']])

        # Loop through each slice and calculate throughput, delay, violation
        for i in range(self.num_slices):
            required = self.state[f'requirements_{i+1}']
            allocated = allocated_resources[i]

            # Throughput
            throughput = min(allocated, required)
            self.state[f'log_throughput_{i+1}'] = throughput

            # Delay
            if allocated == 0.0:
                delay = 1000.0
            else:
                delay = (required - allocated) * 1000 / allocated if allocated < required else 0.0
            self.state[f'log_delay_{i+1}'] = delay

            # Violation
            violation = max(0.0, required - allocated)
            self.state[f'log_violation_{i+1}'] = violation

        # Update step count
        self.current_step += 1
        self.interval_step += 1
        
        # Check if the time to change demand has been reached
        if self.interval_step % self.requirement_interval == 0:
            # Change demand and time interval randomly
            users = self.np_random.integers(self.users_min, self.users_max, size=(self.num_slices,))
            self.state['users_1'] = users[0]
            self.state['users_2'] = users[1]
            self.state['users_3'] = users[2]
            self.state['requirements_1'] = users[0] * self.np_random.uniform(self.aver_req_min, self.aver_req_max)
            self.state['requirements_2'] = users[1] * self.np_random.uniform(self.aver_req_min, self.aver_req_max)
            self.state['requirements_3'] = users[2] * self.np_random.uniform(self.aver_req_min * self.emmb_coe, self.aver_req_max * self.emmb_coe)
            self.requirement_interval = self.np_random.integers(self.requirement_interval_min, self.requirement_interval_max)
            self.interval_step = 0

        # Check if episode is done
        done = self.current_step >= self.max_steps

        # The reward could be calculated based on the throughput, delay, and violation for each slice
        urllc_reward_1 = (self.lambda_throughput_urllc * self.state['log_throughput_1'] +
                          self.lambda_delay_urllc * self.state['log_delay_1'] +
                          self.lambda_violation_urllc * self.state['log_violation_1'])

        urllc_reward_2 = (self.lambda_throughput_urllc * self.state['log_throughput_2'] +
                          self.lambda_delay_urllc * self.state['log_delay_2'] +
                          self.lambda_violation_urllc * self.state['log_violation_2'])
        
        embb_reward = (self.lambda_throughput_embb * self.state['log_throughput_3'] +
                       self.lambda_delay_embb * self.state['log_delay_3'] +
                       self.lambda_violation_embb * self.state['log_violation_3'])

        # Total reward is the sum of eMBB and URLLC rewards
        total_reward = embb_reward + urllc_reward_1 + urllc_reward_2

        self.record() 

        return self.get_observation(), total_reward, done, {}

    def onehot2action(self, action):
        # Convert one-hot encoded action to resource allocation for each slice
        if action == 0: # Avoid all 0 allocation
            action = 31
        allocation = [0] * self.num_slices
        for i in range(self.num_slices):
            allocation[i] = self.discrete_action[action % self.n_discrete]
            action = action // self.n_discrete
        return allocation
    
    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)  # Will use system entropy if seed is None
        return [seed]
    
    def record(self, file_name='recorded metrics test.csv', close_file=False):
        # Initialize recording if not done already
        if not self.is_recording_initialized:
            self.csv_file = open(file_name, mode='w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
                'users_1', 'users_2', 'users_3',
                'requirements_1', 'requirements_2', 'requirements_3',
                'log_throughput_1', 'log_throughput_2', 'log_throughput_3',
                'log_delay_1', 'log_delay_2', 'log_delay_3',
                'log_violation_1', 'log_violation_2', 'log_violation_3',
            ])
            self.csv_writer.writeheader()
            self.is_recording_initialized = True

        # Record observation for the current step
        observation = self.get_observation()

        self.csv_writer.writerow(observation)

        # Close file if it's the end of the simulation
        if close_file and self.csv_file:
            self.csv_file.close()
            self.is_recording_initialized = False


    def render(self):
        print(f"Step: {self.current_step}")
        for i, slice_type in enumerate(self.slice_types):
            print(f"Slice {i+1} ({slice_type}): Users: {self.state[f'users_{i+1}']}, "
                  f"Requirement: {self.state[f'requirements_{i+1}']}, "
                  f"Throughput: {self.state[f'log_throughput_{i+1}']}, "
                  f"Delay: {self.state[f'log_delay_{i+1}']}, "
                  f"Violation: {self.state[f'log_violation_{i+1}']}")
        print("")

    def draw_figures(self):
        steps = range(len(self.users_history))  # Number of steps recorded

        # Create subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot user numbers over steps for each slice
        for i in range(self.num_slices):
            axs[0].plot(steps, [users[i] for users in self.users_history], label=f"Slice {i+1} users")
        axs[0].set_title('Number of Users per Slice over Time')
        axs[0].set_xlabel('Steps')
        axs[0].set_ylabel('Users')
        axs[0].legend(loc='upper right')

        # Plot requirements over steps for each slice
        for i in range(self.num_slices):
            axs[1].plot(steps, [reqs[i] for reqs in self.requirements_history], label=f"Slice {i+1} requirements")
        axs[1].set_title('Requirements per Slice over Time')
        axs[1].set_xlabel('Steps')
        axs[1].set_ylabel('Requirements')
        axs[1].legend(loc='upper right')

        # Show the plot
        plt.tight_layout()
        plt.savefig('network_env.png')
        plt.show()


if __name__ == "__main__":
    # Example usage:
    env = NetworkSlicingEnv()
    state = env.reset()

    for step in range(1000):
        action = env.action_space.sample()  # Sample a random action
        allocation = env.onehot2action(action)
        print(f'action: {action}, allocation:{allocation}')
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            env.draw_figures()
            break

