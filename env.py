import gym
from matplotlib import pyplot as plt
from quadcopter import Quadcopter
from gym import spaces
import numpy as np
from stable_baselines.common.env_checker import check_env
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
import msvcrt


class quadEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, position=(0, 0), stepsize=0.01, timesteps=1000, sse=1, vsse=1, render=False):
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
        self.vsse = vsse
        self.render = render
        self.initdistance = 0
        self.ylims = (-2, 10)
        self.xlims = (-2, 2)

        self.observation_space = spaces.Box(-200, 200, shape=(5,))
        self.action_space = spaces.Box(0, 1, shape=(2,))

    def reset(self):
        position = (0, 0)
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

        return self.getobs(self.quad.state)

    def getobs(self, state):
        x, y = state[0], state[2]
        self.current = np.array([x, y])
        distance = np.linalg.norm(self.goal-self.current)
        self.distance = distance
        return np.concatenate([[self.distance], self.current, self.goal])

    def cal_reward(self):
        reward = 2*np.exp(-self.distance/self.initdistance)
        return reward

    def is_done(self):
        state = self.quad.state
        pos_control = self.distance <= self.sse
        speed_control = np.linalg.norm(state[3:6]) <= self.vsse
        angle_control = -np.pi/2 <= state[7] <= np.pi/2
        inbounds = self.xlims[0] <= self.current[0] <= self.xlims[1] and self.ylims[
            0] <= self.current[1] <= self.ylims[1] and angle_control

        return (pos_control and speed_control) or self.currentstep >= self.timesteps or not inbounds, inbounds, pos_control

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
        self.ax.set_title(
            f"Error:{self.distance:2f},T:{self.quad.state[6:9]}]")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.pause(0.001)

    def step(self, action):
        Thrust = action[0]
        Tau = action[1]

        # MIN_thrust, MAX_thrust = 0, 20
        # MIN_tau, MAX_tau = -0.002, 0.002

        # Thrust = MAX_thrust*Thrust + MIN_thrust
        # Tau = MAX_tau*Tau + MIN_tau

        self.quad.thrust = Thrust
        self.quad.tau = Tau

        state = self.quad.step(self.stepsize, self.currentstep)
        self.currentstep += 1

        obs = self.getobs(state)
        reward = self.cal_reward()
        done, inbounds, pos = self.is_done()
        if not inbounds:
            reward = -50
        elif done:
            reward = 100
        elif pos:
            reward = 5
        else:
            reward = -1
        info = {
            "states": self.quad.state,
            "distance": self.distance
        }

        if self.render:
            self.render_screen()

        return obs, reward, done, info


if __name__ == '__main__':
    env = quadEnv(render=True)
    # print("Current Error:", env.reset())
    train = False
    check = True

    if train:
        n_cpu = 8
        env = make_vec_env(quadEnv, n_envs=n_cpu)
        model = PPO("MlpPolicy", env, verbose=2,
                    learning_rate=1e-3, n_steps=int(4096/n_cpu))
        # model = PPO.load("hello.zip", env=env, learning_rate=2e-4)
        model.learn(total_timesteps=1e5)
        model.save("hello.zip")
    else:
        model = PPO.load("hello.zip")

        while(True):
            obs = env.reset()
            done = False
            actions = []
            action = np.array([2, 0.0])
            while True:
                if not check:
                    action, _ = model.predict(obs)
                else:

                    if msvcrt.kbhit():
                        t = msvcrt.getch()
                        if t == b'a':
                            action[1] -= 0.001
                        if t == b'd':
                            action[1] += 0.001
                        if t == b'w':
                            action[0] += 1
                        if t == b's':
                            action[0] -= 1
                        print(t, action)
                    # continue

                actions.append(action)
                # print(action)
                obs, _, done, _ = env.step(action)
                if done:
                    break
