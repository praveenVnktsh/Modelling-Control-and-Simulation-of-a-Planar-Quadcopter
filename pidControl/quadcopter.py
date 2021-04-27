import math
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import cv2
g = 9.81

class State():

    def __init__(self, state = np.zeros(6)):
        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]
        self.xdot = state[3]
        self.ydot = state[4]
        self.thetadot = state[5]

    def wrap_angle(self, val):
        return ((val + np.pi) % (2 * np.pi) - np.pi)

    def setState(self, arr):
        self.x = arr[0]
        self.y = arr[1]
        self.theta = self.wrap_angle(arr[2])
        self.xdot = arr[3]
        self.ydot = arr[4]
        self.thetadot = arr[5]
    def __str__(self):
        return str(self.getArray())
    def getArray(self):
        array = [
            self.x,
            self.y,
            self.theta,
            self.xdot,
            self.ydot,
            self.thetadot
        ]
        return np.array(array)

class Quadcopter():

    def __init__(self, state=State()):
        self.ode = scipy.integrate.ode(self.state_dot).set_integrator('vode', nsteps=500, method='bdf')

        self.state : State = state

        self.I = 2.5e-4
        self.M = 0.18
        self.L = 0.086
        
        self.thrust = 0
        self.tau = 0

        self.img = np.ones((512, 512, 3), dtype= np.uint8) * 255

    def state_dot(self):

        state_dot = np.zeros(6)

        state = self.state 

        x_dotdot = -self.thrust * math.sin(state.theta)/self.M
        y_dotdot = -g + (self.thrust * math.cos(state.theta) / self.M)
        theta_dotdot = self.tau/self.I


        state_dot[0] = state.xdot
        state_dot[1] = state.ydot
        state_dot[2] = state.thetadot

        state_dot[3] = x_dotdot
        state_dot[4] = y_dotdot
        state_dot[5] = theta_dotdot

        return state_dot

    def render(self, info = {}):

        info['x     '] = self.state.x
        info['y     '] = self.state.y
        info['theta '] = self.state.theta
        info['Thrust'] = self.thrust
        info['Tau   '] = self.tau

        scale = 100

        
        theta = self.state.theta
        x = -self.state.x * scale + 250
        y = -self.state.y * scale + 450

        cv2.circle(self.img, (int(x), int(y)), 1, (0, 0, 255), thickness = - 1)
        

        temp = self.img.copy()

        left =  (int(x - self.L*scale*np.cos(theta)) , int(y - self.L*scale*np.sin(theta)))
        right = (int(x + self.L*scale*np.cos(theta)) , int(y + self.L*scale*np.sin(theta)))

        for i, (k, v) in enumerate(info.items()):
            cv2.putText(temp, k + ' : ' + str(round(v, 4)), (10, 20 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)



        cv2.line(temp, left, right, color = (255, 0, 0), thickness= 3, lineType = cv2.LINE_AA)
        cv2.imshow('render', temp)
        if cv2.waitKey(1) == ord('q'):
            exit()

        

    def step(self, dt):

        self.ode.set_initial_value(self.state.getArray(), 0)
        
        self.state.setState(self.ode.integrate(self.ode.t + dt))

        return self.state


if __name__ == "__main__":

    quad = Quadcopter()
    
    quad.thrust = 1 + quad.M * g
    quad.tau = 0.001

    for i in range(1000):
        state = quad.step(0.001)
        quad.render()
   