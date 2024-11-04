import numpy as np
from typing import Tuple, List
import numpy.typing as npt

class ACCQLearning:
    def __init__(self, env, learning_rate=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # State discretization parameters
        self.speed_bins = np.linspace(0, 40, 20)  # Speed bins from 0 to 40 m/s
        self.distance_bins = np.linspace(0, 150, 30)  # Distance bins from 0 to 150m
        self.rel_speed_bins = np.linspace(-20, 20, 20)  # Relative speed bins
        
        # Initialize Q-table with state discretization
        self.q_table = {}
        
    def discretize_state(self, state: npt.NDArray) -> Tuple:
        """Discretize continuous state space into bins"""
        vehicle_speed = np.digitize(state[0], self.speed_bins)
        distance = np.digitize(state[2], self.distance_bins)
        rel_speed = np.digitize(state[1] - state[0], self.rel_speed_bins)
        
        return (vehicle_speed, distance, rel_speed)
    
    def choose_action(self, state: npt.NDArray) -> int:
        """Choose action using epsilon-greedy policy with decay"""
        discrete_state = self.discretize_state(state)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.env.action_space.shape[0])
        
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.env.action_space.shape[0])
        else:
            return np.argmax(self.q_table[discrete_state])
    
    def update(self, state: npt.NDArray, action: int, reward: float, next_state: npt.NDArray):
        """Update Q-table using Double Q-learning approach"""
        current_state = self.discretize_state(state)
        next_state_discrete = self.discretize_state(next_state)
        
        # Initialize Q-values if state not seen before
        if current_state not in self.q_table:
            self.q_table[current_state] = np.zeros(self.env.action_space.shape[0])
        if next_state_discrete not in self.q_table:
            self.q_table[next_state_discrete] = np.zeros(self.env.action_space.shape[0])
        
        # Get current Q value
        current_q = self.q_table[current_state][action]
        
        # Get next action using current policy
        next_action = np.argmax(self.q_table[next_state_discrete])
        
        # Update using Q-learning
        target = reward + self.gamma * self.q_table[next_state_discrete][next_action]
        self.q_table[current_state][action] += self.learning_rate * (target - current_q)
    
    def train(self, num_episodes: int, max_steps_per_episode: int = 1000) -> List[float]:
        """Train the agent with improved logging and early stopping"""
        episode_rewards = []
        best_avg_reward = float('-inf')
        patience = 50  # Episodes to wait before early stopping
        episodes_without_improvement = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            
            for step in range(max_steps_per_episode):
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Modify reward for better learning
                modified_reward = self._calculate_modified_reward(state, action, reward, next_state)
                
                self.update(state, action, modified_reward, next_state)
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Store episode reward
            episode_rewards.append(total_reward)
            
            # Calculate average reward over last 100 episodes
            if len(episode_rewards) >= 100:
                avg_reward = np.mean(episode_rewards[-100:])
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += 1
                
                # Early stopping
                if episodes_without_improvement >= patience:
                    print(f"Early stopping at episode {episode}")
                    break
            
            if episode % 100 == 0:
                print(f"Episode {episode}/{num_episodes}, Avg Reward: {np.mean(episode_rewards[-100:]):.2f}, Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards
    
    def _calculate_modified_reward(self, state: npt.NDArray, action: int, reward: float, next_state: npt.NDArray) -> float:
        """Calculate modified reward with additional safety and comfort factors"""
        # Extract relevant state information
        current_speed = state[0]
        next_speed = next_state[0]
        distance = next_state[2]
        rel_speed = next_state[1] - next_state[0]
        acceleration = self.env.action_space[action]
        
        # Safety penalty (exponential penalty for close distances)
        safe_distance = current_speed * 2  # 2-second rule
        safety_penalty = -np.exp(max(0, safe_distance - distance) / 10)
        
        # Comfort penalty (quadratic penalty for high accelerations)
        comfort_penalty = -(acceleration ** 2) / 25
        
        # Efficiency reward (reward for maintaining desired speed)
        desired_speed = 30.0  # m/s
        speed_reward = -abs(next_speed - desired_speed) / desired_speed
        
        # Combine rewards with weights
        modified_reward = (
            0.4 * reward +  # Original reward
            0.3 * safety_penalty +  # Safety component
            0.2 * comfort_penalty +  # Comfort component
            0.1 * speed_reward  # Efficiency component
        )
        
        return modified_reward

def run_simulation(q_learner: ACCQLearning, env, num_steps: int = 1000) -> Tuple[List[float], List[float], List[float]]:
    """Run simulation with the improved agent"""
    state = env.reset()
    speeds = []
    distances = []
    accelerations = []
    
    for _ in range(num_steps):
        action = q_learner.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        
        speeds.append(next_state[0])  # Vehicle speed
        distances.append(next_state[2])  # Distance to preceding vehicle
        accelerations.append(env.action_space[action])
        
        state = next_state
        if done:
            break
    
    return speeds, distances, accelerations