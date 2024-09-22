import gym
from gym import spaces
import numpy as np
import yaml
import matplotlib.pyplot as plt


class NetworkSlicingEnv(gym.Env):
    def __init__(self, config_file='config.yaml'):
        super(NetworkSlicingEnv, self).__init__()

        # Load parameters from YAML file
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
        self.params = self.config['parameters']

        # Number of slices: 2 URLLC and 1 eMBB
        self.num_slices = 3
        self.slice_types = ['URLLC', 'URLLC', 'eMBB']  # Slice types

        # Fixed number of time steps per episode from the YAML config
        self.max_steps = self.params['max_steps']
        self.current_step = 0

        # Action space: Allocate resources to 3 slices
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_slices,), dtype=np.float32)

        self.users_max = self.params['users_max']
        self.users_min = self.params['users_min']
        self.aver_req_max = self.params['aver_req_max']
        self.aver_req_min = self.params['aver_req_min']

        # Observation space: Random number of users and requirements for each slice
        self.observation_space = spaces.Dict({
            'users': spaces.Box(low=self.users_min, high=self.users_max, shape=(self.num_slices,), dtype=np.int32),
            'requirements': spaces.Box(low=0, high=self.aver_req_max * self.users_max, shape=(self.num_slices,), dtype=np.float32)
        })

        # Resource pool from YAML
        self.total_bandwidth = self.params['total_bandwidth']
        
        # Reward weights for eMBB and URLLC from YAML
        self.lambda_throughput_embb = self.params['lambda_throughput_embb']
        self.lambda_delay_embb = self.params['lambda_delay_embb']
        self.lambda_violation_embb = self.params['lambda_violation_embb']

        self.lambda_throughput_urllc = self.params['lambda_throughput_urllc']
        self.lambda_delay_urllc = self.params['lambda_delay_urllc']
        self.lambda_violation_urllc = self.params['lambda_violation_urllc']

        # Requirement interval from YAML
        self.requirement_interval_min = self.params['requirement_interval_min']
        self.requirement_interval_max = self.params['requirement_interval_max']

        self.reset()

        self.users_history = []
        self.requirements_history = []

    def reset(self):
        # Reset step count
        self.current_step = 0
        
        # Randomly initialize users for each slice
        users = np.random.randint(self.users_min, self.users_max, size=(self.num_slices,))

        # Randomly initialize total requirements for each slice based on the number of users
        self.state = {
            'users': users,
            'requirements': np.array([
                user_count * np.random.uniform(self.aver_req_min, self.aver_req_max)
                for user_count in users
            ])
        }
        self.requirement_interval = np.random.randint(self.requirement_interval_min, self.requirement_interval_max)
        
        return self.state

    def step(self, action):
        # Normalize actions so they sum to the total available bandwidth
        allocated_resources = action / np.sum(action) * self.total_bandwidth

        # Record users and requirements at each step
        self.users_history.append(self.state['users'].copy())
        self.requirements_history.append(self.state['requirements'].copy())

        # Initialize metrics
        embb_throughput, embb_delay, embb_violation = 0, 0, 0
        urllc_throughput, urllc_delay, urllc_violation = 0, 0, 0

        # Loop through each slice and calculate metrics
        for i in range(self.num_slices):
            required = self.state['requirements'][i]
            allocated = allocated_resources[i]

            if self.slice_types[i] == 'eMBB':
                # eMBB metrics
                embb_throughput += min(allocated, required)
                embb_delay += (required - allocated) / required if allocated < required else 0
                embb_violation += max(0, required - allocated)

            elif self.slice_types[i] == 'URLLC':
                # URLLC metrics
                urllc_throughput += min(allocated, required)
                urllc_delay += (required - allocated) / required if allocated < required else 0
                urllc_violation += max(0, required - allocated)

        # eMBB reward calculation
        embb_reward = (self.lambda_throughput_embb * embb_throughput +
                       self.lambda_delay_embb * embb_delay +
                       self.lambda_violation_embb * embb_violation)
        
        # URLLC reward calculation
        urllc_reward = (self.lambda_throughput_urllc * urllc_throughput +
                        self.lambda_delay_urllc * urllc_delay +
                        self.lambda_violation_urllc * urllc_violation)
        
        # Total reward is the sum of both slice rewards
        total_reward = embb_reward + urllc_reward

        # Update step count
        self.current_step += 1
        
        # Check if the time to change demand has been reached
        if self.current_step % self.requirement_interval == 0:
            # Change demand and time interval randomly
            users = np.random.randint(self.users_min, self.users_max, size=(self.num_slices,))
            self.state['users'] = users
            self.state['requirements'] = np.array([
                user_count * np.random.uniform(self.aver_req_min, self.aver_req_max)
                for user_count in users
            ])
            self.requirement_interval = np.random.randint(self.requirement_interval_min, self.requirement_interval_max)

        # Check if episode is done
        done = self.current_step >= self.max_steps

        # Return next state, reward, done, and additional info
        return self.state, total_reward, done, {}

    def render(self):
        print(f"Step: {self.current_step}")
        for i, slice_type in enumerate(self.slice_types):
            print(f"Slice {i+1} ({slice_type}): Users: {self.state['users'][i]}, "
                  f"Requirement: {self.state['requirements'][i]}")
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
        plt.savefig('F:\sliceWM\zzzzzzz.png')
        plt.show()



# Example usage:
env = NetworkSlicingEnv(config_file='F:\sliceWM\config.yaml')
state = env.reset()

for step in range(1000):
    action = env.action_space.sample()  # Sample a random action
    env.render()
    next_state, reward, done, info = env.step(action)
    if done:
        env.draw_figures()
        break

