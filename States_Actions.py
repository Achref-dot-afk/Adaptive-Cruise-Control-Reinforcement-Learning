import numpy as np

class ACCState:
    def __init__(self):
        self.vehicle_speed = 0.0  # m/s
        self.preceding_vehicle_speed = 0.0  # m/s
        self.distance_ahead = 0.0  # meters
        self.vehicle_acceleration = 0.0  # m/s^2
        self.traffic_density = 0  # vehicles per km

    def normalize(self):
        # Normalize values to a [0, 1] range
        # These max values are examples and should be adjusted based on your specific scenario
        max_speed = 50.0  # m/s (180 km/h)
        max_distance = 100.0  # meters
        max_acceleration = 5.0  # m/s^2
        max_density = 200  # vehicles per km

        return np.array([
            self.vehicle_speed / max_speed,
            self.preceding_vehicle_speed / max_speed,
            self.distance_ahead / max_distance,
            (self.vehicle_acceleration + max_acceleration) / (2 * max_acceleration),  # Shift to [0, 1]
            self.traffic_density / max_density
        ])

    def get_state_vector(self):
        return self.normalize()

class ACCAction:
    def __init__(self):
        self.acceleration = 0.0  # m/s^2

    def get_action_vector(self):
        # Normalize acceleration to [-1, 1] range
        return np.array([self.acceleration / 5.0])  # Assuming max acceleration is 5 m/s^2