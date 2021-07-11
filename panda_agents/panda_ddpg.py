import gym
import panda_gym
from tensorforce import Runner, Environment
from tensorforce.agents.agent import Agent
from gym.wrappers.time_limit import TimeLimit


LEVEL = 'PandaReachDense-v1'
EPISODES = 15000  # 5000
EPISODE_MAX_LENGTH = 50
PARALLEL = 10

MODEL_DICT = f'{LEVEL}/model'
SUMMARY_DICT = f'{LEVEL}/summary'
RECORD_DICT = f'{LEVEL}/record'


def get_agent_and_runner(max_timesteps=EPISODE_MAX_LENGTH):
    max_timesteps = EPISODE_MAX_LENGTH if max_timesteps is None else max_timesteps
    # OpenAI-Gym environment specification
    gym_environment = gym.make(LEVEL, render=True)
    gym_environment = TimeLimit(gym_environment.unwrapped, max_episode_steps=max_timesteps)
    # gym_environment = Monitor(gym_environment, RECORD_DICT, force=True)


    environment = Environment.create(
        environment=gym_environment, 
        max_episode_timesteps=gym_environment.spec.max_episode_steps,
    )

    agent = Agent.create(
        agent='ddpg',
        environment=environment,
        memory=10000,
#         parallel_interactions=PARALLEL,
        # Automatically configured network
        network=[
            dict(type='dense', size=256, activation='tanh'),
            dict(type='dense', size=256, activation='tanh'),
            dict(type='dense', size=256, activation='tanh'),
        ],
#         network='auto',
        batch_size=256, update_frequency=2, learning_rate=0.001,
        # Reward estimation
        discount=0.995, predict_terminal_values=False,
        # Regularization
        l2_regularization=1.0, entropy_regularization=0.0,
        # Preprocessing
        state_preprocessing='linear_normalization', reward_preprocessing=None,
        # Exploration
        exploration=0.3, variable_noise=0.2,
        # Default additional config values
        config=None,
        # Save agent every 10 updates and keep the 5 most recent checkpoints
        saver=dict(directory=MODEL_DICT, frequency=10, max_checkpoints=5),
        # Log all available Tensorboard summaries
        summarizer=dict(directory=SUMMARY_DICT, summaries='all'),
        # Do not record agent-environment interaction trace
        recorder=None # RECORD_DICT
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
