import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import csv
from gym.utils import seeding


class NetworkSlicingEnv(gym.Env):
    def __init__(self, max_steps=1000,
                 total_bandwidth=200,
                 user_range=[1, 20],  # Range of random users in each slice
                 lambda_throughput_urllc=0.5, lambda_throughput_embb=5.0, lambda_latency_urllc=1.0, lambda_latency_embb=1.0):
        
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

        # Users configuration
        self.users_min = user_range[0]
        self.users_max = user_range[1]
        self.total_bandwidth = total_bandwidth
        self.is_recording_initialized = False

        # Observation space: Queue for each user in each slice
        self.observation_space = spaces.Dict({
            'queues': spaces.Box(low=0, high=np.inf, shape=(self.num_slices, self.users_max), dtype=np.float64),
            'throughput_metrics': spaces.Box(low=0, high=np.inf, shape=(self.num_slices,), dtype=np.float64),
            'latency_metrics': spaces.Box(low=0, high=np.inf, shape=(self.num_slices,), dtype=np.float64)
        })
        
        # Reward weights for URLLC and eMBB
        self.lambda_throughput_urllc = lambda_throughput_urllc
        self.lambda_throughput_embb = lambda_throughput_embb
        self.lambda_latency_urllc = lambda_latency_urllc
        self.lambda_latency_embb = lambda_latency_embb

        self.np_random = None
        self.seed()
        self.reset()

        self.arrival_rate = 10  # Average arrival rate for new data  
        self.T = 5  # Window length for average
        self.queues = np.zeros((self.num_slices, self.users_max))
        self.queue_history = [[[] for _ in range(self.users_max)] for _ in range(self.num_slices)]
        self.sent_data_history = [[[] for _ in range(self.users_max)] for _ in range(self.num_slices)]
        self.latency_history = []

    def reset(self):
        # Reset step count
        self.current_step = 0
        
        # Randomly initialize users for each slice
        self.num_users = self.np_random.integers(self.users_min, self.users_max + 1, size=(self.num_slices,))
        
        # Reset the queues for each user in each slice
        self.queues = np.zeros((self.num_slices, self.users_max))
        self.queue_history = [[[] for _ in range(self.users_max)] for _ in range(self.num_slices)]
        self.sent_data_history = [[[] for _ in range(self.users_max)] for _ in range(self.num_slices)]
      
        
        return self.get_observation()

    def get_observation(self):
        # Returns the current observation, including queues for each user in each slice
        return {
            'queues': self.queues,
            'throughput_metrics': self.throughput_metrics,
            'latency_metrics': self.latency_metrics
        }

    def step(self, action):
        action = self.onehot2action(action)
        allocated_resources = action / np.sum(action) * self.total_bandwidth

        # Update queues for each user in each slice
        for s in range(self.num_slices):
            for u in range(self.num_users[s]):
                # Arrival data size follows a uniform distribution
                arrival_data = self.np_random.uniform(0, self.arrival_rate)
                
                # Update queue with arrival data
                self.queues[s][u] += arrival_data

                # Allocate resources and update the queue
                sent_data = allocated_resources[s] * (self.queues[s][u] / np.sum(self.queues[s][:self.num_users[s]])) if np.sum(self.queues[s][:self.num_users[s]]) > 0 else 0
                sent_data = min(sent_data, self.queues[s][u])
                self.queues[s][u] -= sent_data
                
                # Record data
                self.sent_data_history[s][u].append(sent_data)
                self.queue_history[s][u].append(self.queues[s][u])


        # Calculate metrics for reward
        self.throughput_metrics = self.calculate_throughput()
        self.latency_metrics = self.calculate_latency()
        throughput_metrics = self.calculate_throughput()
        latency_metrics = self.calculate_latency()

        # Calculate rewards for each slice
        urllc_reward = sum(self.lambda_throughput_urllc * throughput_metrics[:2]) - sum(self.lambda_latency_urllc * latency_metrics[:2])
        embb_reward = self.lambda_throughput_embb * throughput_metrics[2] - self.lambda_latency_embb * latency_metrics[2]

        total_reward = urllc_reward + embb_reward

        # Update step count
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Record metrics
        self.record_metrics(self.throughput_metrics, self.latency_metrics)

        return self.get_observation(), total_reward, done, {}

    def calculate_throughput(self):
        throughput_metrics = []
        for s in range(self.num_slices):
            slice_throughput = 0.0
            for u in range(self.num_users[s]):
                # Calculate user's average throughput by averaging the data sent by the user over the past T timesteps
                sent_data_history = self.sent_data_history[s][u][-self.T:] if len(self.sent_data_history[s][u]) >= self.T else self.sent_data_history[s][u][:]
                avg_throughput = np.mean(sent_data_history) if sent_data_history else 0
                slice_throughput += avg_throughput
            throughput_metrics.append(slice_throughput)
        return throughput_metrics

    def calculate_latency(self):
        latency_metrics = []
        for s in range(self.num_slices):
            slice_latency = 0.0
            for u in range(self.num_users[s]):
                # Calculate user's average queue size over the past T timesteps
                queue_history = self.queue_history[s][u][-self.T:] if len(self.queue_history[s][u]) >= self.T else self.queue_history[s][u][:]
                avg_queue = np.mean(queue_history) if queue_history else 0
                
                # Calculate user's average throughput over the past T timesteps
                sent_data_history = self.sent_data_history[s][u][-self.T:] if len(self.sent_data_history[s][u]) >= self.T else self.sent_data_history[s][u][:]
                avg_throughput = np.mean(sent_data_history) if sent_data_history else 0
                
                # Calculate average latency for the user
                avg_latency = avg_queue / avg_throughput if avg_throughput > 0 else 0
                slice_latency += avg_latency
            # Average latency for the slice
            latency_metrics.append(slice_latency / self.num_users[s] if self.num_users[s] > 0 else 0)
        return latency_metrics

    def onehot2action(self, action):
        if action == 0:  # Avoid all 0 allocation
            action = 31
        allocation = [0] * self.num_slices
        for i in range(self.num_slices):
            allocation[i] = self.discrete_action[action % self.n_discrete]
            action = action // self.n_discrete
        return allocation

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)  # Will use system entropy if seed is None
        return [seed]

    def record_metrics(self, throughput_metrics, latency_metrics):
        self.throughput_history.append(throughput_metrics)
        self.latency_history.append(latency_metrics)

    def render(self):
        print(f"Step: {self.current_step}")
        for i, slice_type in enumerate(self.slice_types):
            print(f"Slice {i + 1} ({slice_type}): Users: {self.num_users[i]}, "
                  f"Throughput: {self.throughput_history[-1][i]}, "
                  f"Latency: {self.latency_history[-1][i]}")
        print("")

    def draw_figures(self):
        steps = range(len(self.throughput_history))  # Number of steps recorded

        # Create subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot throughput over steps for each slice
        for i in range(self.num_slices):
            axs[0].plot(steps, [throughput[i] for throughput in self.throughput_history], label=f"Slice {i + 1} throughput")
        axs[0].set_title('Throughput per Slice over Time')
        axs[0].set_xlabel('Steps')
        axs[0].set_ylabel('Throughput')
        axs[0].legend(loc='upper right')

        # Plot latency over steps for each slice
        for i in range(self.num_slices):
            axs[1].plot(steps, [latency[i] for latency in self.latency_history], label=f"Slice {i + 1} latency")
        axs[1].set_title('Latency per Slice over Time')
        axs[1].set_xlabel('Steps')
        axs[1].set_ylabel('Latency')
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
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            env.draw_figures()
            break
