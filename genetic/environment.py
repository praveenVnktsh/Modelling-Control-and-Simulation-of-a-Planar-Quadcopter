from matplotlib import pyplot as plt
from quadcopter import Quadcopter, State
import numpy as np


class quadEnv():

    def __init__(self, position=(0, 0), stepsize=0.005,render=False):

        
        self.quad = Quadcopter()
        self.quad.state[0] = position[0]
        self.quad.state[2] = position[1]

        self.current = position
        self.setpoint = None
        self.stepsize = stepsize
        self.currentstep = 0
        self.render = render

        self.initdistance = 0
        self.ylims = (-2, 10)
        self.xlims = (-2, 2)


    def reset(self, setpoint, startTheta):
        position = (0, 0)
        self.fitness = 0
        self.setpoint =  setpoint
        self.distance = np.linalg.norm(np.array(self.setpoint)-np.array(self.current))
        self.initdistance = self.distance

        self.quad = Quadcopter()
        self.quad.state[0] = position[0]
        self.quad.state[2] = position[1]
        self.quad.state[7] = startTheta
        self.preverror = np.array([abs(setpoint[0]), abs(setpoint[1])])
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

            # setpoint
            plt.scatter(self.setpoint[0], self.setpoint[1], label='setpoint',  c='g', s=10)

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

    
    def render_screen(self, state):
        theta = state.theta
        x, y = state.x, state.y
        self.xval.append(x)
        self.yval.append(y)
        self.traj_plt.set_xdata(self.xval)
        self.traj_plt.set_ydata(self.yval)
        self.arm_plt.set_xdata(
            [-self.quad.L*np.cos(theta) + x, self.quad.L*np.cos(theta) + x])
        self.arm_plt.set_ydata([y + self.quad.L *
                                np.sin(theta), y - self.quad.L*np.sin(theta)])
        self.ax.set_title(
            f"Error:{self.preverror},Reward : {self.fitness}]")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def getReward(self, state : State, action):
        setpoint = self.setpoint
        error = np.array([abs(setpoint[0] - state.x), abs(setpoint[1] - state.y)])
        # reward = np.sum(np.array((-error + self.preverror)))
        if np.sum(error) > 0.3:
            reward = np.sum(np.array((-error + self.preverror)))
        else:
            reward = 2*np.exp(-(abs(state.xdot)/12.0 + abs(state.ydot)/12.0))

        self.preverror = error

        if abs(state.theta) > 1.57:
            reward -= 100
        if abs(state.y) > 10 or abs(state.x) > 2:
            reward -= 100

        return reward

    def isDone(self, state : State, reward):
        done = abs(state.theta) > 1 or reward < -5
        done = done or abs(state.x) > 2 or abs(state.y) > 10
        return done

    def step(self, action):

        self.quad.thrust = action[0]*10 + self.quad.M * 9.81
        self.quad.tau = action[1]

        state = self.quad.step(self.stepsize, self.currentstep)
        self.currentstep += 1

        reward = self.getReward(state, action)
        done = self.isDone(state, reward)
        self.fitness += reward

        info = {
            "states": self.quad.state,
            "distance": self.distance
        }

        if self.render:
            self.render_screen(state)

        return state, reward, done, info


if __name__ == '__main__':
    env = quadEnv(render=True)
    # print("Current Error:", env.reset())
    train = False
    check = True
