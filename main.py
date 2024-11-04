from Enviroment import ACCEnvironment
from ACCQLearning import ACCQLearning, run_simulation
import matplotlib.pyplot as plt

# Create environment and improved Q-Learning agent
env = ACCEnvironment()
acc_q_learning = ACCQLearning(env)

# Train the agent
episode_rewards = acc_q_learning.train(num_episodes=10000)

# Run simulation with improved agent
speeds, distances, accelerations = run_simulation(acc_q_learning, env)

# Plot results including training progress
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(episode_rewards)
plt.title('Training Progress')
plt.ylabel('Episode Reward')
plt.xlabel('Episode')

plt.subplot(2, 2, 2)
plt.plot(speeds)
plt.title('Vehicle Speed')
plt.ylabel('Speed (m/s)')

plt.subplot(2, 2, 3)
plt.plot(distances)
plt.title('Distance to Preceding Vehicle')
plt.ylabel('Distance (m)')

plt.subplot(2, 2, 4)
plt.plot(accelerations)
plt.title('Vehicle Acceleration')
plt.ylabel('Acceleration (m/s^2)')

plt.tight_layout()
plt.show()