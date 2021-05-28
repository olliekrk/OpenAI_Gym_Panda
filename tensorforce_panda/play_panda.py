import gym
import panda_gym
from tensorforce import Agent, Runner
from tensorforce.environments import OpenAIGym, Environment
from gym.wrappers import Monitor
from matplotlib import animation
import matplotlib.pyplot as plt
import random
import time

random.seed(time.time_ns())

LEVEL = 'PandaReach-v1'
EPISODES = 1
EPISODE_MAX_LENGTH = 500

MODEL_DICT = f'{LEVEL}/model'
VISUALIZE_DICT = f'{LEVEL}/visualize/{random.randint(0, 1000)}'

gym_environment = gym.make(LEVEL)
environment = Environment.create(
    environment=gym_environment, 
    max_episode_timesteps=EPISODE_MAX_LENGTH,
    visualize=True,
    visualize_directory=VISUALIZE_DICT,
)

agent = Agent.load(directory=MODEL_DICT, environment=environment)

runner = Runner(agent=agent, environment=environment, max_episode_timesteps=EPISODE_MAX_LENGTH)
runner.run(num_episodes=EPISODES, evaluation=True)
runner.close()

# sum_rewards = 0.0
# for _ in range(EPISODES):
#     states = environment.reset()
#     internals = agent.initial_internals()
#     terminal = False
#     while not terminal:
#         actions, internals = agent.act(
#             states=states, internals=internals, independent=True, deterministic=True
#         )
#         states, terminal, reward = environment.execute(actions=actions)
#         sum_rewards += reward

# print('Mean evaluation return:', sum_rewards / EPISODES)

# # Close agent and environment
# agent.close()
# environment.close()

