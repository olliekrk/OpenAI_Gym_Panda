# Using Panda-Agents

The main script is located at `../run_panda_agents.py`

To get help:
```
$ python run_panda_agents.py -h
```

An example output:
```
usage: run_panda_agents.py [-h] [--max_timesteps] [--episodes] policy

positional arguments:
  policy            learning policy. Possible choices:
                ppo - Proximal Policy Optimization,
                ddpg - Deep Deterministic Policy Gradient,
                actor - Advantage Actor-Critic

optional arguments:
  -h, --help        show this help message and exit
  --max_timesteps   maximum number of timesteps per episode
  --episodes        number of episodes when learning
```

An example of usage:
```
$ python run_panda_agents.py ddpg --episodes=10000
```