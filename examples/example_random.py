import time
import gym
import panda_gym

STEPS = 500

env = gym.make('PandaPush-v1', render=True)

env.reset()
done = False
for i in range(STEPS):
    action = env.action_space.sample() # random action
    obs, reward, done, info = env.step(action)
    print(obs, reward, info)
    time.sleep(.01)
    if done:
        env.reset()

print(f'DONE?: {done}')
env.close()