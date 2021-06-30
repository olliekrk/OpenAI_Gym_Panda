import matplotlib.pyplot as plt

from panda_agents import panda_ppo


agent, runner = panda_ppo.get_agent_and_runner()
runner.run(num_episodes=panda_ppo.EPISODES)

plt.figure(0)
plt.plot(runner.episode_returns)
plt.title('PPO rewards')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('ppo-rewards.png', format='png')

plt.figure(1)
plt.plot(runner.episode_timesteps, 'C1')
plt.title('PPO timesteps per episode')
plt.xlabel('Episode')
plt.ylabel('Timesteps')
plt.savefig('ppo-timesteps.png', format='png')