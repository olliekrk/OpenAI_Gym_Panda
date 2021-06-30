import gym
import panda_gym
from tensorforce import Runner, Environment
from tensorforce.agents.agent import Agent
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers import Monitor


LEVEL = 'PandaReachDense-v1'
EPISODES = 50  # 5000
EPISODE_MAX_LENGTH = 250
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
        agent='ppo',
        environment=environment,
        # parallel_interactions=PARALLEL,
        # Automatically configured network
        # network='auto',
        network=[
            dict(type='dense', size=32, activation='relu'),
            dict(type='dense', size=32, activation='relu'),
            dict(type='dense', size=64, activation='relu'),
            dict(type='dense', size=64, activation='relu'),
            dict(type='dense', size=16, activation='tanh')
        ],
        # PPO optimization parameters
        batch_size=50, update_frequency=2, learning_rate=4.5e-1, multi_step=10,
        subsampling_fraction=0.33,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.995, predict_terminal_values=False,
        # Baseline network and optimizer
        baseline=dict(type='auto', size=32, depth=1),
        baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
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
