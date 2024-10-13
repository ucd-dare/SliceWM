import numpy as np
import random
from embodied.envs.slice_env import NetworkSlicingEnv

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def test_environment_with_policies(env, policy='equal', steps=3000):
    state = env.reset()
    step = 0
    
    while step < steps:
        if policy == 'equal':
            action = env.action_space.sample()
        elif policy == 'uniform':
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        
        # allocation = env.onehot2action(action)
        # print(f'Step: {step}, Policy: {policy}, Action: {action}, Allocation: {allocation}')
        
        obs, reward, done, info = env.step(action)
        step += 1

        if done:
            env.reset()

env = NetworkSlicingEnv()

print("Testing with Equal Policy")
test_environment_with_policies(env, policy='equal')

# print("Testing with Uniform Policy")
# test_environment_with_policies(env, policy='uniform')
