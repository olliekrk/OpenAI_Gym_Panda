import gym
from tensorforce import Agent, Runner
from tensorforce.environments import OpenAIGym, Environment
from gym.wrappers import Monitor
from matplotlib import animation
import matplotlib.pyplot as plt
import random
import time

random.seed(time.time_ns())

EPISODES = 1

environment = OpenAIGym(
    level='CartPole-v1',
    visualize=True,
    visualize_directory=f'visualization_{random.randint(0, 1000)}',
)

agent = Agent.load(directory='model', environment=environment)

# runner = Runner(agent=agent, environment=environment, max_episode_timesteps=500)
# runner.run(num_episodes=EPISODES, evaluation=True)
# runner.close()

sum_rewards = 0.0
for _ in range(EPISODES):
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(
            states=states, internals=internals, independent=True, deterministic=True
        )
        states, terminal, reward = environment.execute(actions=actions)
        sum_rewards += reward

print('Mean evaluation return:', sum_rewards / EPISODES)

# Close agent and environment
agent.close()
environment.close()

