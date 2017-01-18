import logging
import time
import os, sys
import pdb
import numpy as np

import gym

# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def demo(self, observation, reward, done):
        angle = input("input angle in degrees \n")
        vel = input("input velocity\n")
        return np.array([angle*np.pi/180., vel])

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make('Golf-v0')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env.monitor.start(outdir, force=True, seed=0)

    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    agent = RandomAgent(env.action_space)

    episode_count = 100
    max_steps = 200
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        stopped = True

        for j in range(max_steps):
            time.sleep(0.05)
            env._render()
            if stopped:
                action = agent.demo(ob, reward, done)

            ob, reward, done, info = env.step(action)
            stopped = info['stopped']

            if done:
                break

    # Dump result info to disk
    env.monitor.close()

