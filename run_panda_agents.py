from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import matplotlib.pyplot as plt
import argparse

from panda_agents import panda_ppo, panda_ddpg


policy_help = """
learning policy. Possible choices:
ppo - Proximal Policy Optimization,
ddpg - Deep Deterministic Policy Gradient,
actor - Advantage Actor-Critic
"""

parser = argparse.ArgumentParser()
parser.add_argument('policy', choices=[
    'ppo', 'ddpg', 'actor'
], help=policy_help, metavar='policy')
parser.add_argument('--max_timesteps', type=int, required=False,
help='maximum number of timesteps per episode', metavar='')
parser.add_argument('--episodes', type=int, required=False,
help='number of episodes when learning', metavar='')
args = parser.parse_args()

POLICY_CHOICES = {
    'ppo': panda_ppo,
    'ddpg': panda_ddpg
}

rs = RandomState(MT19937(SeedSequence(52136)))

policy = args.policy
max_episode_timesteps = args.max_timesteps
chosen_policy = POLICY_CHOICES[policy]
episodes = args.episodes if args.episodes is not None else chosen_policy.EPISODES

agent, runner = chosen_policy.get_agent_and_runner(max_episode_timesteps)
runner.run(num_episodes=episodes)

plt.figure(0)
plt.plot(runner.episode_returns)
plt.title(f'{policy.upper()} rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig(f'{policy}-rewards.png', format='png')

plt.figure(1)
plt.plot(runner.episode_timesteps, 'C1')
plt.title(f'{policy.upper()} timesteps per episode')
plt.xlabel('Episode')
plt.ylabel('Timesteps')
plt.savefig(f'{policy}-timesteps.png', format='png')