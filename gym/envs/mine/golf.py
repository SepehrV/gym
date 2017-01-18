import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from os import path

import pdb

class GolfEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 1
    }

    def __init__(self):
        self.max_speed=100
        self.max_angle = 2*np.pi
        self.field_length = 100
        self.window_size = 500
        self.scale = self.window_size/self.field_length
        self.proximity_thresh = 1
        self.viewer = None

        self.action_space = spaces.Box(np.array([0,0]),
                                       np.array([self.max_angle,self.max_speed]))
        self.observation_space = spaces.Box(
                np.array([0,0,0,0]),
                np.array([self.field_length]*4))
        self.stopped = True
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,u):
        costs = 0
        done = False
        if self.stopped == True:
            u = np.clip(u, [0, 0],
                           [self.max_angle, self.max_speed])
            self.u = u
            self.stopped = False

        bx, by, gx, gy= self.state # th := theta
        angle, vel = self.u

        if vel > 0:
            print (self.u)
            bx = bx + math.cos(angle)
            by = by + math.sin(angle)
            if bx > self.field_length or bx < 0 or by > self.field_length or by < 0:
                done = True
                costs = 100
            else:
                self.state[0] = bx
                self.state[1] = by
                self.u[1] = self.u[1] - 1
        else:
            self.stopped = True
            costs = np.linalg.norm([bx-gx, by-gy])
            if costs < self.proximity_thresh:
                done = True

        return self._get_obs(), -costs, done, {'stopped':self.stopped}

    def _reset(self):
        self.state = self.np_random.uniform(np.array([0,0,0,0]),
                                            np.array([self.field_length]*4))
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.window_size,self.window_size)
            #self.viewer.set_bounds(0,self.field_length,0,self.field_length)
            self.ball_transform = rendering.Transform()
            self.target_transform = rendering.Transform()
            target = rendering.make_circle(self.scale)
            target.set_color(0,1,0)
            target.add_attr(self.target_transform)
            ball = rendering.make_circle(self.scale)
            ball.set_color(1,0,0)
            ball.add_attr(self.ball_transform)
            self.viewer.add_geom(ball)
            self.viewer.add_geom(target)
            #fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            #self.img = rendering.Image(fname, 1., 1.)
            #self.imgtrans = rendering.Transform()
            #self.img.add_attr(self.imgtrans)

        #self.viewer.add_onetime(self.img)
        self.target_transform.set_translation(
                int(self.scale*self.state[2]),
                int(self.scale*self.state[3]))
        self.ball_transform.set_translation(
                int(self.scale*self.state[0]),
                int(self.scale*self.state[1]))
        #if self.last_u:
        #    self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
