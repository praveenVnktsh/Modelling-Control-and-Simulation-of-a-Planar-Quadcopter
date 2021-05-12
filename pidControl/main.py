import math
from pid import PID
from quadcopter import Quadcopter, State
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import cv2
import json

def execute(pid):
    g = 9.81
    stepsize = 0.001
    plot = True
    render = False
    state = State()
    state.theta = 0
    quad = Quadcopter(state = state)
    complete = False
    xval = []
    thetaval = []
    yval = []
    setpoints = [
        np.array([1.0, 1.0]),
    ]

    totaltime = 0.0
    ymax = 0.0
    xmax = 0.0
    # print(10//stepsize)
    # time = np.linspace(0., 10., num = int(10/stepsize))
    # thrusts = np.sin(50*time) + quad.M*g
    cnt = 0
    startTimex = 0
    endtimex = 0
    startTimey = 0
    endtimey = 0
    for setpoint in setpoints:
        complete = False
        pid.setSetpoint(setpoint)
        
        
        fail = False
        while totaltime < 10:
            # quad.thrust = thrusts[cnt]
            # quad.tau = 1

            cnt += 1

            state, fail = quad.step(stepsize)

            totaltime += stepsize
            phi, complete = pid.step(quad, state, stepsize)
            ymax = max(ymax, state.y)
            xmax = max(xmax, state.x)
            # if totaltime > 5:
            #     fail = True
            #     print('Failed')

            if round(state.y, 1) == 0.1:
                startTimey = totaltime
            elif round(state.y, 1) == 0.9:
                endtimey =  totaltime

            if round(state.x, 1) == 0.1:
                startTimex = totaltime
            elif round(state.x, 1) == 0.9:
                endtimex = totaltime
            
            xval.append(state.x)
            yval.append(state.y)
            thetaval.append(state.theta)
            
            info = {
                'x_des' : setpoint[0], 
                'y_des' : setpoint[1]
            }
            if render:
                quad.render(info)
    
    info = {}
    
    if not fail:
        print('Rise time y', endtimey - startTimey)
        print('Overshoot y %  = ', (ymax - setpoints[0][1])*100/ setpoints[0][1])
        print()
        print('Rise time x', endtimex - startTimex)
        print('Overshoot x %  = ', (xmax - setpoints[0][0])*100/ setpoints[0][0])

        if plot:
            time = np.linspace(0, totaltime, num = len(xval))
            # plt.plot(time, [setpoints[0][0] for i in range(len(xval))],  label = 'X_des')
            # plt.plot(time, [setpoints[0][1]  for i in range(len(xval))],  label = 'Y_des')
            plt.plot(time, xval, label = 'Y_out')
            plt.plot(time, yval, label = 'X_out')

            with open('non.txt', 'w') as f:
                json.dump(xval, f)

            plt.grid()
            plt.title("Closed Loop Step response Nonlinear system (x_des, y_des)")
            plt.xlabel('Time (s)')
            plt.ylabel('Output')
            plt.legend()

            plt.show()
        # info = {
        #     'risetimex' : timediffx if timediffx is not None else 100,
        #     'risetimey' : timediffy if timediffy is not None else 100,
        #     'OSy' : (ymax - setpoints[0][1])*100/ setpoints[0][1] if timediffy is not None else 100,
        #     'Osx' :  (xmax - setpoints[0][0])*100/ setpoints[0][0] if timediffy is not None else 100,
        #     'pid' : str(pid)
        # }
    info = None
    if render:
        cv2.waitKey(0)
    return info, fail






# y track, x track, phi track
# y - Rise time -0.36000000000000026 Overshoot % =  5.50947454441173 for Kp = 6.0, Kd = 1.42

# y, x, phi


pid = PID( 
        Kp = [7.0, 2.5, 0.04], 
        Kd = [1.42,   0.56, 0.008]
)
info, fail = execute(pid)
# for j in np.linspace(0.0, 3.0, 300):
#     # for i in np.linspace(0.01, 2, 20):
#     pid = PID( 
#         Kp = [9.0, 3.0, 0.7], 
#         Kd = [1.42,   j, 0.1]
#     )
#     try:
#         info, fail = execute(pid)
#     except Exception as e:
#         pass
#     if not fail:
#         print('---------------')
#         print(info)

