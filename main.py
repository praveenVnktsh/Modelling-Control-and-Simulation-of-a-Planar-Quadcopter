import math
from pid import PID
from quadcopter import Quadcopter, State
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from tqdm import tqdm



def plotSim():
    global state
    plt.figure(1)
    plt.clf()
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    plt.plot(xval, yval, label='X',  c='r')
    theta = state.theta
    x = state.x
    y = state.y
    plt.plot([-quad.L*np.cos(theta) + x, quad.L*np.cos(theta) + x], [y +
                                                                        quad.L*np.sin(theta), y - quad.L*np.sin(theta)], c='b', linewidth=2.5)


def plotGraph(i):
    time = np.linspace(0, len(yval)*i, num=len(yval))


    plt.figure(2)
    plt.clf()

    plt.plot(time, yval, label='Y',  c='r')
    plt.plot(time, xval, label='X', c='b')
    plt.plot(time, thetaval, label='Theta', c='g')
    plt.grid()
    plt.legend()


if __name__ == "__main__":
    g = 9.81
    stepsize = 0.005
    plot = True
    initState = np.array([
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ], dtype=float)
    quad = Quadcopter(state=initState)

    # height, angle track, y track

    pid = PID(dt=stepsize, Kp=[2.0, 0.5, 0.1], Kd=[1.5, 0.3, 0.04])
    
    

    plt.ion()

    xval = []
    yval = []


    ylims = (-10, 10)
    xlims = (-2, 2)
    state = State()
    if plot:
        plotSim()
    thetaval = []
    prevx = 0
    prevy = 0
    complete = False
    i = 0
    setpoints = [
        np.array([1, 2.5]),
        np.array([0, 0.5]),

    ]
    for setpoint in setpoints:
        complete = False
        pid.setSetpoint(setpoint)
        while not complete:
            i += 1
            state = quad.step(stepsize, i, False)

            phi, complete = pid.step(quad, state)

            xval.append(state.x)
            yval.append(state.y)
            thetaval.append(state.theta)
            
            

            dist = np.sqrt(abs(prevx - state.x)**2 + abs(prevy - state.y)**2) > 0.1 
            if i % 1000 == 0:
                print(i)
            if plot and (dist or i % 500 == 0):
                print('Updating at', i)


                plt.pause(0.001)
                plotSim()
                plotGraph(i)

                prevx = state.x
                prevy = state.y


    plotGraph(i)
    print('Simulation complete')
    plt.show(block=True)
