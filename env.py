import gym
from matplotlib import pyplot as plt
from quadcopter import Quadcopter
from gym import spaces
import numpy as np


class quadEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, position=(0, 0), stepsize=0.01, timesteps=1000, sse=1e-2, render=False):
        super(quadEnv, self).__init__()

        assert(len(position) == 2)
        self.quad = Quadcopter()
        self.quad.state[0] = position[0]
        self.quad.state[2] = position[1]

        self.current = position
        self.goal = None
        self.timesteps = timesteps
        self.stepsize = stepsize
        self.currentstep = 0
        self.sse = sse
        self.render = render
        self.initdistance = 0
        self.ylims = (-2, 10)
        self.xlims = (-2, 2)

    def reset(self, position=(0, 0)):

        x = np.random.rand(1)*(self.xlims[1]-self.xlims[0]) + self.xlims[0]
        y = np.random.rand(1)*(self.ylims[1])

        self.goal = np.array([x, y]).squeeze()
        self.distance = np.linalg.norm(self.goal-self.current)
        self.initdistance = self.distance

        self.quad = Quadcopter()
        self.quad.state[0] = position[0]
        self.quad.state[2] = position[1]

        self.current = position
        self.currentstep = 0

        if self.render:
            plt.close()
            plt.ion()
            self.xval = [position[0], ]
            self.yval = [position[1], ]
            theta = self.quad.state[7]
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.ax.set_xlim(self.xlims)
            self.ax.set_ylim(self.ylims)

            # Goal
            plt.scatter(self.goal[0], self.goal[1], label='Goal',  c='g', s=10)

            # Trajectory
            self.traj_plt, = plt.plot(self.xval,
                                      self.yval,
                                      label='X',  c='r')

            # Arm
            self.arm_plt, = plt.plot([-self.quad.L*np.cos(theta) + position[0], self.quad.L*np.cos(theta) + position[0]],
                                     [position[1] + self.quad.L *
                                      np.sin(theta), position[1] - self.quad.L*np.sin(theta)],
                                     c='b', linewidth=2.5)

            plt.show()

        return self.distance

    def getobs(self, state):
        x, y = state[0], state[2]
        self.current = np.array([x, y])
        distance = np.linalg.norm(self.goal-self.current)
        self.distance = distance
        return distance

    def cal_reward(self):
        reward = -self.distance/self.initdistance
        return reward

    def is_done(self):
        inbounds = self.xlims[0] <= self.current[0] <= self.xlims[1] and self.ylims[0] <= self.current[1] <= self.ylims[1]
        return self.distance <= self.sse or self.currentstep >= self.timesteps or not inbounds

    def render_screen(self):
        theta = self.quad.state[7]
        x, y = self.current
        self.xval.append(x)
        self.yval.append(y)
        self.traj_plt.set_xdata(self.xval)
        self.traj_plt.set_ydata(self.yval)
        self.arm_plt.set_xdata(
            [-self.quad.L*np.cos(theta) + x, self.quad.L*np.cos(theta) + x])
        self.arm_plt.set_ydata([y + self.quad.L *
                                np.sin(theta), y - self.quad.L*np.sin(theta)])
        self.ax.set_title(f"Error:{self.distance:2f}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.pause(0.001)

    def step(self, action):
        Thrust = action[0]
        Tau = action[1]

        self.quad.thrust = Thrust
        self.quad.tau = Tau

        state = self.quad.step(self.stepsize, self.currentstep)
        self.currentstep += 1

        obs = self.getobs(state)
        reward = self.cal_reward()
        done = self.is_done()
        info = {
            "states": self.quad.state,
            "distance": self.distance
        }

        if self.render:
            self.render_screen()

        return obs, reward, done, info


if __name__ == '__main__':
    env = quadEnv(render=True)
    print("Current Error:", env.reset())
    while(True):
        env.reset()
        for i in range(100):
            _, _, done, _ = env.step([10, 0.001])
            if done:
                break
