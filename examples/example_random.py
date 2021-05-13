import time
import gym
import panda_gym

env = gym.make('PandaReach-v1', render=True)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample() # random action
    obs, reward, done, info = env.step(action)
    time.sleep(.01)

print(f'DONE?: {done}')
env.close()