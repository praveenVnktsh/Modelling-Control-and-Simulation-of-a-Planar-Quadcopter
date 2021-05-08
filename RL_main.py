import math
from pidControl.pid import PID
from quadcopter import Quadcopter, State
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from tqdm import tqdm


def plotStep(obj):
    theta = state.theta
    x = state.x
    y = state.y

    # Fig 1
    obj["traj"].set_data(xval, yval)
    obj["arm"].set_xdata(
        [-quad.L*np.cos(theta) + x, quad.L*np.cos(theta) + x])
    obj["arm"].set_ydata(
        [y + quad.L * np.sin(theta), y - quad.L*np.sin(theta)])

    # Fig 2
    time = np.linspace(0, len(yval)*stepsize, num=len(yval))
    obj["ax"][1].set_xlim([0, np.max(time)])
    obj["ax"][1].set_ylim([np.min([xval, yval, thetaval]),
                           np.max([xval, yval, thetaval])])

    obj["x"].set_data(time, xval)
    obj["y"].set_data(time, yval)
    obj["t"].set_data(time, thetaval)

    # Flushing Graphs
    obj["fig"].canvas.draw()
    obj["fig"].canvas.flush_events()
    return obj


def initGraph():
    theta = state.theta
    x = state.x
    y = state.y
    plt.ion()
    fig, ax = plt.subplots(2, figsize=(10, 8))

    # For Fig 1
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[0].grid()
    ax[0].set_title("Simulating Drone")
    # Trajectory
    traj_plt, = ax[0].plot(xval, yval, label='Trajectory',  c='r')
    # Arm
    arm_plt, = ax[0].plot([quad.L*np.cos(theta) + x, quad.L*np.cos(theta) + x], [y + quad.L * np.sin(theta), y - quad.L*np.sin(theta)],
                          c='b', linewidth=2.5)

    # for point in setpoints:
    ax[0].scatter(setpoints[:, 0], setpoints[:, 1], c="g", label="goal")
    ax[0].legend()

    # For fig 2
    ax[1].grid()
    ax[1].set_title("Drone Trajectory")
    trajy = ax[1].plot(yval, label='Y',  c='r')[0]
    trajx = ax[1].plot(xval, label='X', c='b')[0]
    trajt = ax[1].plot(thetaval, label='Theta', c='g')[0]
    ax[1].legend()

    obj = {"arm": arm_plt,
           "fig": fig,
           "ax": ax,
           "traj": traj_plt,
           "x": trajx,
           "y": trajy,
           "t": trajt}

    return obj


if __name__ == "__main__":
    g = 9.81
    stepsize = 0.005
    plot = True
    initState = np.zeros(12, dtype=float)
    quad = Quadcopter()

    # height, angle track, y track

    pid = PID(dt=stepsize, Kp=[2.0, 0.5, 0.1], Kd=[1.5, 0.3, 0.04])

    plt.ion()

    xval = []
    yval = []
    thetaval = []

    ylims = (-2, 10)
    xlims = (-4, 4)
    setpoints = np.array([
        [1, 1],
        [-1, 2.5],
        [2, 5],
    ])

    state = State()
    if plot:
        graphobj = initGraph()
    prevx = 0
    prevy = 0
    complete = False
    i = 0
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

            dist = np.sqrt(abs(prevx - state.x)**2 +
                           abs(prevy - state.y)**2) > 0.1
            if i % 1000 == 0:
                print(i)
            if plot and (dist or i % 500 == 0):
                print('Updating at', i, "Angular Speed:", quad.state[10])
                graphobj = plotStep(graphobj)

                prevx = state.x
                prevy = state.y

    print('Simulation complete')
    plt.show(block=True)
