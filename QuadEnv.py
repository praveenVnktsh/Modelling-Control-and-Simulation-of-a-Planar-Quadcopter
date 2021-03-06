import gym
from matplotlib import pyplot as plt
from numpy.core.numeric import False_
from quadcopter import Quadcopter, State
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
import msvcrt


class quadEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, position=(0, 0), stepsize=0.01, timesteps=300, sse=0.25, vsse=0.25, render=False):
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

        self.observation_space = spaces.Box(-200, 200, shape=(8,))
        self.action_space = spaces.Box(-1, 1, shape=(2,))

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

        return self.getobs(State(self.quad.state))

    def getobs(self, state):
        self.current = state.getpos()
        distance = np.linalg.norm(self.goal-self.current)
        self.distance = distance
        return np.concatenate([self.goal, state.getstate()])

    def cal_reward(self, state):

        pos_control = self.distance <= self.sse
        speed_control = np.linalg.norm([state.xdot, state.ydot]) <= self.vsse
        angle_control = -np.pi/3 <= state.theta <= np.pi/3
        angle_speed_control = -2 < self.quad.state[10] < 2

        inbounds = self.xlims[0] <= self.current[0] <= self.xlims[1] and self.ylims[
            0] <= self.current[1] <= self.ylims[1]

        done = (
            pos_control and speed_control) or self.currentstep >= self.timesteps or not inbounds
        done = self.currentstep >= self.timesteps

        reward = -2
        if not inbounds:
            reward += -10

        if not angle_speed_control:
            reward += -0.5*angle_speed_control

        if pos_control:
            reward += 10
            if speed_control:
                reward += 10
        reward += 12*(np.exp(-self.distance/self.initdistance) -
                      np.exp(-1))/(1-np.exp(-1))

        return reward, done

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

    def step(self, action):
        Thrust = action[0]
        Tau = action[1]

        MIN_thrust, MAX_thrust = 0, 20
        MIN_tau, MAX_tau = -0.5, 0.5

        Thrust = (Thrust + 1)*(MAX_thrust-MIN_thrust)/2 + MIN_thrust
        Tau = (Tau + 1)*(MAX_tau-MIN_tau)/2 + MIN_tau

        self.quad.thrust = Thrust
        self.quad.tau = Tau

        state = self.quad.step(self.stepsize, self.currentstep)
        self.currentstep += 1

        obs = self.getobs(state)
        reward, done = self.cal_reward(state)

        info = {
            "states": self.quad.state,
            "distance": self.distance,
            "reward": reward
        }

        if self.render:
            self.render_screen()

        return obs, reward, done, info


if __name__ == '__main__':
    env = quadEnv(render=True)
    train = 0
    check = False

    if train:
        n_cpu = 8
        env = make_vec_env(quadEnv, n_envs=n_cpu)
        model = PPO("MlpPolicy", env, verbose=2,
                    learning_rate=1e-4, n_steps=int(4096/n_cpu))
        model.learn(total_timesteps=1e6)
        model.save("model.zip")
    else:
        model = PPO.load(r"goodmodels\bestilnow.zip")
        # model = PPO.load(r"model.zip")

        while(True):
            obs = env.reset()
            done = False
            actions = []
            action = np.array([2, 0.0])
            while True:
                if not check:
                    action, _ = model.predict(obs)

                    print(action, env.quad.state[10])
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
                        if t == b'q':
                            break

                        print(t, action)

                actions.append(action)
                obs, _, done, _ = env.step(action)
                if msvcrt.kbhit():
                    t = msvcrt.getch()
                    if t == b'q':
                        done = True
                if done:
                    break
