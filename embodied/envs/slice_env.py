import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from gym.utils import seeding


class NetworkSlicingEnv(gym.Env):
    def __init__(self, max_steps=1000,
                 total_bandwidth=150,
                 user_range=[8, 16],  # Range of random users in each slice
                 lambda_throughput_urllc=0.5, lambda_throughput_embb=5.0, lambda_latency_urllc=-2.0, lambda_latency_embb=-0.1):
        
        super(NetworkSlicingEnv, self).__init__()

        # Number of slices: 2 URLLC and 1 eMBB
        self.slice_types = ['URLLC', 'URLLC', 'eMBB']  # Slice types
        self.num_slices = len(self.slice_types)
        self.discrete_action = [0.0, 0.25, 0.5, 0.75, 1.0]  # Allocation options
        self.n_discrete = len(self.discrete_action)
        
        # Fixed number of time steps per episode
        self.max_steps = max_steps
        self.current_step = 0
        self.max_queue = 200
        self.user_coe = 2.0

        # Action space: Allocate resources to 3 slices
        self.action_space = spaces.Discrete(self.n_discrete ** self.num_slices)

        # Users configuration
        self.users_min = user_range[0]
        self.users_max = user_range[1]
        self.total_bandwidth = total_bandwidth
        self.is_recording_initialized = False

        # Observation space: Queue for each user in each slice
        self.observation_space = spaces.Dict({
            'aver_throughput': spaces.Box(low=0, high=np.inf, shape=(self.num_slices,), dtype=np.float64),
            'aver_latency': spaces.Box(low=0, high=np.inf, shape=(self.num_slices,), dtype=np.float64),
            'total_queue': spaces.Box(low=0, high=np.inf, shape=(self.num_slices,), dtype=np.float64)
        })
        
        # Reward weights for URLLC and eMBB
        self.lambda_throughput_urllc = lambda_throughput_urllc
        self.lambda_throughput_embb = lambda_throughput_embb
        self.lambda_latency_urllc = lambda_latency_urllc
        self.lambda_latency_embb = lambda_latency_embb

        self.np_random = None
        self.seed()
        self.reset()

        self.arrival_rate = 15  # Average arrival rate for new data  
        self.T = 3  # Window length for average
        self.queues = np.zeros((self.num_slices, self.users_max))
        self.queue_history = [[[] for _ in range(self.users_max)] for _ in range(self.num_slices)]
        self.sent_data_history = [[[] for _ in range(self.users_max)] for _ in range(self.num_slices)]
        self.throughput_history = []
        self.latency_history = []

    def reset(self):
        # Reset step count
        self.current_step = 0
        
        # Randomly initialize users for each slice
        self.num_users = self.np_random.integers(self.users_min, self.users_max + 1, size=(self.num_slices,))
        self.num_users[0] = self.np_random.integers(
        int(self.users_min / self.user_coe), 
        int((self.users_max + 1) / self.user_coe)
        )
        self.num_users[1] = self.np_random.integers(
            int(self.users_min / self.user_coe), 
            int((self.users_max + 1) / self.user_coe)
        )
        
        # Reset the queues for each user in each slice
        self.queues = np.zeros((self.num_slices, self.users_max))
        self.queue_history = [[[] for _ in range(self.users_max)] for _ in range(self.num_slices)]
        self.arrival_data_history = [[[] for _ in range(self.users_max)] for _ in range(self.num_slices)]
        self.sent_data_history = [[[] for _ in range(self.users_max)] for _ in range(self.num_slices)]

        self.slice_total_queue = [0.0] * self.num_slices
        self.slice_aver_throughput = [0.0] * self.num_slices
        self.slice_aver_latency = [0.0] * self.num_slices
        
        return self.get_observation()

    def get_observation(self):
        # Returns the current observation, including queues for each user in each slice
        return {
            'total_queue': np.array(self.slice_total_queue) * (1 + np.random.uniform(-0.08, 0.08)),
            'aver_throughput': self.slice_aver_throughput,
            'aver_latency': self.slice_aver_latency
        }

    def step(self, action):
        action = self.onehot2action(action)
        allocated_resources = action / np.sum(action) * self.total_bandwidth

        # Update queues for each user in each slice
        for s in range(self.num_slices):
            for u in range(self.num_users[s]):
                arrival_data = self.np_random.uniform(0, self.arrival_rate)
                self.arrival_data_history[s][u].append(arrival_data)
                self.queues[s][u] += arrival_data
                self.queues[s][u] = min(self.queues[s][u], self.max_queue)

                # Allocate resources and update the queue
                if len(self.queue_history[s][u]) == 0 or sum(self.queue_history[s][user][self.current_step-1] for user in range(self.num_users[s])) == 0:
                    # When the first step and when all queue equal to 0, use equal allocation
                    sent_data = allocated_resources[s] / self.num_users[s]
                else:
                    # Proportional allocation
                    last_queue = self.queue_history[s][u][self.current_step-1]
                    sent_data = allocated_resources[s] * last_queue / sum(self.queue_history[s][user][self.current_step-1] for user in range(self.num_users[s]))
                sent_data = min(sent_data, self.queues[s][u])
                self.queues[s][u] -= sent_data
                
                self.sent_data_history[s][u].append(sent_data)
                self.queue_history[s][u].append(self.queues[s][u])

        # TODO: the unit of latency

        # Calculate metrics for reward
        self.slice_aver_throughput = self.calculate_throughput()
        self.slice_aver_latency = self.calculate_latency()
        self.slice_total_queue = self.calculate_queue()

        # Calculate rewards for each slice
        urllc_reward = np.dot([self.lambda_throughput_urllc] * 2, self.slice_aver_throughput[:2]) + np.dot([self.lambda_latency_urllc] * 2, self.slice_aver_latency[:2])
        embb_reward = (self.lambda_throughput_embb * self.slice_aver_throughput[2]) + (self.lambda_latency_embb * self.slice_aver_latency[2])

        total_reward = urllc_reward + embb_reward

        # Update step count
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Record metrics
        self.record_metrics(self.slice_aver_throughput, self.slice_aver_latency)

        return self.get_observation(), total_reward, done, {}
    
    def calculate_queue(self):
        slice_total_queue = []
        for s in range(self.num_slices):
            slice_queue = 0.0
            for u in range(self.num_users[s]):
                # Calculate user's average queue size over the past T timesteps
                queue_history = self.queue_history[s][u][-self.T:] if len(self.queue_history[s][u]) >= self.T else self.queue_history[s][u][:]
                avg_queue = np.mean(queue_history) if queue_history else 0
                slice_queue += avg_queue
            # Average queue for the slice
            slice_total_queue.append(slice_queue)
        return slice_total_queue

    def calculate_throughput(self):
        slice_aver_throughput = []
        for s in range(self.num_slices):
            slice_throughput = 0.0
            for u in range(self.num_users[s]):
                # Calculate user's average throughput by averaging the data sent by the user over the past T timesteps
                sent_data_history = self.sent_data_history[s][u][-self.T:] if len(self.sent_data_history[s][u]) >= self.T else self.sent_data_history[s][u][:]
                avg_throughput = np.mean(sent_data_history) if sent_data_history else 0
                slice_throughput += avg_throughput
            slice_aver_throughput.append(slice_throughput)
        return slice_aver_throughput

    def calculate_latency(self):
        slice_aver_latency = []
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
                avg_latency = avg_queue / avg_throughput if avg_throughput > 0 else 500
                slice_latency += avg_latency
            # Average latency for the slice
            slice_aver_latency.append(slice_latency / self.num_users[s] if self.num_users[s] > 0 else 0)
        return slice_aver_latency

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

        record = {
            "throughput_1": throughput_metrics[0],
            "throughput_2": throughput_metrics[1],
            "throughput_3": throughput_metrics[2],
            "latency_1": latency_metrics[0],
            "latency_2": latency_metrics[1],
            "latency_3": latency_metrics[2]
        }
        
        csv_file = "recorded_metrics.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    def render(self):
        print(f"Step: {self.current_step}")
        for i, slice_type in enumerate(self.slice_types):
            print(f"Slice {i + 1} ({slice_type}): Users: {self.num_users[i]}")
            # for u in range(self.num_users[i]):
            #     print(f"  User {u + 1} Arrival: {self.arrival_data_history[i][u][self.current_step-1]:.2f}")
            #     print(f"  User {u + 1} Queue: {self.queues[i][u]:.2f}")
            print(f"  Total Queue: {self.slice_total_queue[i]:.2f}")
            print(f"  Throughput: {self.slice_aver_throughput[i]:.2f}")
            print(f"  Latency: {self.slice_aver_latency[i]:.2f}")
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
        # env.render()
        if done:
            # env.draw_figures()
            break
