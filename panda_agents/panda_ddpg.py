import gym
import panda_gym
from tensorforce import Runner, Environment
from tensorforce.agents.agent import Agent
from gym.wrappers.time_limit import TimeLimit
import numpy as np


LEVEL = 'PandaReachDense-v1'
EPISODES = 500  # 5000
EPISODE_MAX_LENGTH = 150
PARALLEL = 10

MODEL_DICT = f'{LEVEL}/model'
SUMMARY_DICT = f'{LEVEL}/summary'
RECORD_DICT = f'{LEVEL}/record'


def get_agent_and_runner():
    # OpenAI-Gym environment specification
    gym_environment = gym.make(LEVEL, render=True)
    gym_environment = TimeLimit(gym_environment.unwrapped, max_episode_steps=EPISODE_MAX_LENGTH)
    # gym_environment = Monitor(gym_environment, RECORD_DICT, force=True)


    environment = Environment.create(
        environment=gym_environment, 
        max_episode_timesteps=gym_environment.spec.max_episode_steps,
    )

    # PPO agent specification
    agent = Agent.create(
        agent='ddpg',
        environment=environment,
        # parallel_interactions=PARALLEL,
        # Automatically configured network
        # network='auto',
        network=[
            dict(type='dense', size=32, activation='tanh'),
            dict(type='dense', size=32, activation='tanh'),
            dict(type='dense', size=64, activation='tanh'),
            dict(type='dense', size=64, activation='tanh'),
            dict(type='dropout', rate=0.6),
            dict(type='dense', size=16, activation='tanh')
        ],
        # PPO optimization parameters
        batch_size=15, update_frequency=2, learning_rate=4.5e-5,
        memory=10000,
        # Reward estimation
        discount=0.995, predict_terminal_values=False,
        # Regularization
        l2_regularization=0.2, entropy_regularization=0.0,
        # Preprocessing
        state_preprocessing='linear_normalization', reward_preprocessing=None,
        # Exploration
        exploration=0.15, variable_noise=0.0,
        # Default additional config values
        config=None,
        # Save agent every 10 updates and keep the 5 most recent checkpoints
        saver=dict(directory=MODEL_DICT, frequency=10, max_checkpoints=5),
        # Log all available Tensorboard summaries
        summarizer=dict(directory=SUMMARY_DICT, summaries='all'),
        # Do not record agent-environment interaction trace
        recorder=None  # RECORD_DICT
    )

    # Initialize the runner
    runner = Runner(
        agent=agent, 
        environment=environment, 
        max_episode_timesteps=gym_environment.spec.max_episode_steps,
        # num_parallel=PARALLEL,
        # remote="multiprocessing"
    )
    
    return agent, runner
