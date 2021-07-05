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
args = parser.parse_args()

POLICY_CHOICES = {
    'ppo': panda_ppo.get_agent_and_runner,
    'ddpg': panda_ddpg.get_agent_and_runner
}

rs = RandomState(MT19937(SeedSequence(52136)))

policy = args.policy
agent, runner = POLICY_CHOICES[policy]()
runner.run(num_episodes=panda_ppo.EPISODES)

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