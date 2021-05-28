import gym
import panda_gym
from tensorforce import Runner, Environment
from tensorforce.agents.agent import Agent

LEVEL = 'PandaReach-v1'
EPISODES = 50000
EPISODE_MAX_LENGTH = 500
PARALLEL = 10

MODEL_DICT = f'{LEVEL}/model'
SUMMARY_DICT = f'{LEVEL}/summary'
RECORD_DICT = f'{LEVEL}/record'

def main():
    # OpenAI-Gym environment specification
    gym_environment = gym.make(LEVEL)

    environment = Environment.create(
        environment=gym_environment, 
        max_episode_timesteps=EPISODE_MAX_LENGTH,
    )

    # PPO agent specification
    agent = Agent.create(
        agent='ppo',
        environment=environment,
        parallel_interactions=PARALLEL,
        # Automatically configured network
        network='auto',
        # PPO optimization parameters
        batch_size=10, update_frequency=2, learning_rate=3e-4, multi_step=10,
        subsampling_fraction=0.33,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.99, predict_terminal_values=False,
        # Baseline network and optimizer
        baseline=dict(type='auto', size=32, depth=1),
        baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # Preprocessing
        state_preprocessing='linear_normalization', reward_preprocessing=None,
        # Exploration
        exploration=0.0, variable_noise=0.0,
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
        max_episode_timesteps=EPISODE_MAX_LENGTH,
        # num_parallel=PARALLEL,
        # remote="multiprocessing"
    )

    # Train
    runner.run(
        num_episodes=EPISODES,
        # batch_agent_calls=True,
    )
    
    runner.close()

    # plus agent.close() and environment.close() if created separately


if __name__ == '__main__':
    main()