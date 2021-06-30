import matplotlib.pyplot as plt

from panda_agents import panda_ppo


agent, runner = panda_ppo.get_agent_and_runner()
runner.run(num_episodes=panda_ppo.EPISODES)

plt.plot(runner.episode_returns)
plt.title('PPO rewards')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('ppo-rewards.png', format='png')