import math
from pid import PID
from quadcopter import Quadcopter
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":
    g = 9.81
    timesteps = 1000
    stepsize = 0.001
    plot = False
    initState = np.array([
        0, 
        0, 
        -1, 
        0, 
        0, 
        0,
        0,
        0.1,
        0,
        0,
        0,
        0
    ], dtype = float)
    quad = Quadcopter(state = initState)

    pid = PID(dt = stepsize, Kp = [1.0, 10.0], Kd = [0.1, 2.30], Ki = [0.0, 0.0])
    setpoint = np.array([0.0, 0.0])
    pid.setSetpoint(setpoint)
    
    plt.ion()
    
    xval = []
    yval = []
    
    # quad.thrust =  1 + quad.M * g
    # quad.tau = 0.001

    
    ylims = (-10, 10)
    xlims = (-2, 2)
    if plot:
        plt.figure(1)
        plt.clf()
        plt.xlim(xlims[0], xlims[1])
        plt.ylim(ylims[0], ylims[1])
        plt.show()
        plt.grid()
    thetaval = []
    for i in range(timesteps):
        state = quad.step(stepsize, i, True)
        
        # if i > 30:
        #     quad.tau = 0

        pid.step(quad, state)

        x = state[0]
        y = state[2]

        theta = state[7]

        xval.append(x)
        yval.append(y)
        thetaval.append(theta)

        if plot and i % 2  == 0: 
            plt.clf()
            plt.xlim(xlims[0], xlims[1])
            plt.ylim(ylims[0], ylims[1])
            plt.plot(xval, yval, label = 'X',  c = 'r')
            plt.plot([-quad.L*np.cos(theta) + x, quad.L*np.cos(theta) + x], [y + quad.L*np.sin(theta), y - quad.L*np.sin(theta)], c= 'b', linewidth = 2.5)
            plt.pause(0.001)

        


    plt.figure(2)
    time = np.linspace(0, timesteps*stepsize, num = timesteps)
    plt.plot(time, yval, label = 'X',  c = 'r')
    plt.plot(time, thetaval, label = 'Theta', c = 'b')
    plt.grid()
    plt.legend()
    # plt.show()
    plt.show(block = True)