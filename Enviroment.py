import numpy as np
from States_Actions import ACCState, ACCAction
from Rewards_Policy import reward_function
class ACCEnvironment:
    def __init__(self):
        self.state = ACCState()
        self.action_space = np.linspace(-5, 5, 21)  # 21 discrete acceleration values from -5 to 5 m/s^2

    def reset(self):
        # Initialize the state with some random values
        self.state.vehicle_speed = np.random.uniform(0, 30)  # m/s
        self.state.preceding_vehicle_speed = np.random.uniform(0, 30)  # m/s
        self.state.distance_ahead = np.random.uniform(10, 100)  # meters
        self.state.vehicle_acceleration = 0  # m/s^2
        self.state.traffic_density = np.random.randint(0, 200)  # vehicles per km
        return self.state.get_state_vector()

    def step(self, action_index):
        action = ACCAction()
        action.acceleration = self.action_space[action_index]
        
        # Update state based on action
        dt = 1.0  # time step of 1 second
        self.state.vehicle_speed += action.acceleration * dt
        self.state.vehicle_speed = max(0, self.state.vehicle_speed)  # Ensure non-negative speed
        self.state.distance_ahead += (self.state.preceding_vehicle_speed - self.state.vehicle_speed) * dt
        self.state.distance_ahead = max(0, self.state.distance_ahead)  # Ensure non-negative distance
        self.state.vehicle_acceleration = action.acceleration

        # Simplified updates for preceding vehicle and traffic
        self.state.preceding_vehicle_speed += np.random.normal(0, 1)  # Random speed changes
        self.state.preceding_vehicle_speed = max(0, self.state.preceding_vehicle_speed)
        self.state.traffic_density += np.random.randint(-10, 11)  # Random traffic density changes
        self.state.traffic_density = np.clip(self.state.traffic_density, 0, 200)

        # Calculate reward
        reward = reward_function(self.state, action, self.state)

        # Check if episode is done (you might want to define your own conditions)
        done = False

        return self.state.get_state_vector(), reward, done, {}
