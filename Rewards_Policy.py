import numpy as np
from States_Actions import *

def reward_function(state: ACCState, action: ACCAction, next_state: ACCState) -> float:
    # Define weights for different factors
    w_safety = 0.5
    w_efficiency = 0.3
    w_comfort = 0.2

    # Safety: Penalize being too close to the preceding vehicle
    safety_distance = next_state.vehicle_speed * 2  # 2 seconds rule
    safety_reward = -np.exp(safety_distance - next_state.distance_ahead)

    # Efficiency: Reward for maintaining a steady speed close to the desired speed
    desired_speed = 30.0  # m/s (108 km/h)
    efficiency_reward = -abs(next_state.vehicle_speed - desired_speed) / desired_speed

    # Comfort: Penalize large accelerations
    comfort_reward = -abs(action.acceleration) / 5.0

    # Combine rewards
    total_reward = w_safety * safety_reward + w_efficiency * efficiency_reward + w_comfort * comfort_reward

    return total_reward

def epsilon_greedy_policy(q_table, state, epsilon=0.1):
    # Map the state to a unique index within bounds
    index = int(np.dot(state, np.arange(1, len(state) + 1))) % q_table.shape[0]
    if np.random.random() < epsilon:
        return np.random.choice(len(q_table[0]))  # Random action
    else:
        return np.argmax(q_table[index])  # Greedy action