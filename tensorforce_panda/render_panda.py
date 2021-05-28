import gym
import panda_gym
from tensorforce import Agent
from tensorforce.environments import Environment
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder 
import random
import time

random.seed(time.time_ns())

LEVEL = 'PandaReach-v1'
EPISODES = 1
EPISODE_MAX_LENGTH = 1000

MODEL_DICT = f'{LEVEL}/model'
VISUALIZE_DICT = f'{LEVEL}/visualize/{random.randint(0, 1000)}'

gym_environment = gym.make(LEVEL)
gym_environment = Monitor(gym_environment, VISUALIZE_DICT, force=True)

environment = Environment.create(
    environment=gym_environment, 
    max_episode_timesteps=EPISODE_MAX_LENGTH,
)

agent = Agent.load(directory=MODEL_DICT, environment=environment)

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
        gym_environment.render()
        sum_rewards += reward

print('Mean evaluation return:', sum_rewards / EPISODES)

# Close agent and environment
agent.close()
environment.close()

