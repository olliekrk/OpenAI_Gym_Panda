from spinup import ppo_tf1 as ppo
import tensorflow as tf
import gym
from gym.wrappers import FlattenObservation
import panda_gym

EPOCHS = 100
STEPS_PER_EPOCH = 4000
ENV = 'PandaPush-v1'

def env_fn():
    env = gym.make(ENV)
    print(env.observation_space)
    env = FlattenObservation(env)
    return env

ac_kwargs = dict(
    hidden_sizes=[64,64], 
    activation=tf.nn.sigmoid,
)

logger_kwargs = dict(output_dir=f'logs/sigmoid_{ENV}', exp_name=f'exp_sigmoid_{ENV}')

ppo(
    env_fn=env_fn,
    ac_kwargs=ac_kwargs,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    logger_kwargs=logger_kwargs
)