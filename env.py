import gym
from Quadcopter import Quadcopter
from gym import spaces
import numpy as np


class quadEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, position=(0, 0)):
        super(quadEnv, self).__init__()

        assert(position.size == 3)
        self.quad = Quadcopter()
        self.quad.state[0] = position[0]
        self.quad.state[2] = position[2]

        self.current = position
        self.goal = None

    def reset(self, position=(0, 0)):
        self.goal = np.random.randn(1, 1)
        self.distance = np.linalg.norm(self.goal-self.current)

        self.quad = Quadcopter()
        self.quad.state[0] = position[0]
        self.quad.state[2] = position[2]

        self.current = position
        return self.distance

    def step(self, action):
