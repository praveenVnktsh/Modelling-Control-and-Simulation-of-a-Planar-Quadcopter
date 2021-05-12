import math
from pid import PID
from quadcopter import Quadcopter, State
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import json



# y track, x track, phi track
# y - Rise time -0.36000000000000026 Overshoot % =  5.50947454441173 for Kp = 6.0, Kd = 1.42

# y, x, phi



def execute(pid):
    global values, bestvalues
    complete = False
    xval = []
    yval = []

    setpoints = [
        np.array([1.0, 1.0]),
    ]

    totaltime = 0.0
    ymax = 0.0
    xmax = 0.0

    state = State()
    quad = Quadcopter(state = state)
    g = 9.81
    stepsize = 0.005
    plot = False
    render = False

    for setpoint in setpoints:
        complete = False
        pid.setSetpoint(setpoint)
        fail = False
        while not complete or not fail:

            state, fail = quad.step(stepsize)
            totaltime += stepsize
            phi, complete = pid.step(quad, state, stepsize)
            fail = fail or (totaltime > 0.8)
            ymax = max(ymax, state.y)
            xmax = max(xmax, state.x)
            # print(fail)
            if not fail:
                if round(state.y, 1) == 0.1:
                    startTimey = totaltime
                elif round(state.y, 1) == 0.9:
                    timediffy = startTimey - totaltime

                if round(state.x, 1) == 0.1:
                    startTimex = totaltime
                elif round(state.x, 1) == 0.9:
                    timediffx = startTimex - totaltime
            
            xval.append(state.x)
            yval.append(state.y)
            
            
            if render:
                info = {
                'x_des' : setpoint[0], 
                'y_des' : setpoint[1]
                }
                quad.render(info)

        time = np.linspace(0, totaltime, num = len(xval))


    if not fail:
        info = {
            'risetimex' : timediffx if timediffx is not None else 100,
            'risetimey' : timediffy if timediffy is not None else 100,
            'OSy' : (ymax - setpoints[0][1])*100/ setpoints[0][1] if timediffy is not None else 100,
            'Osx' :  (xmax - setpoints[0][0])*100/ setpoints[0][0] if timediffy is not None else 100,
            'pid' : str(PID)
        }
        values.append(info)
        if info['risetimex'] < 0.36 and info['risetimey'] < 0.36 and info['OSy'] < 16 and info['OSy'] < 16:
            bestvalues.append(info)
            print(info)
            if len(bestvalues) % 10 == 0:
                json.dump(bestvalues, open('bestvalues.txt' , 'w'))
            

        

    # print('Rise time y', timediffy)
    # print('Overshoot y %  = ', (ymax - setpoints[0][1])*100/ setpoints[0][1])
    # print()
    # print('Rise time x', timediffx)
    # print('Overshoot x %  = ', (xmax - setpoints[0][0])*100/ setpoints[0][0])



pid = PID( 
    Kp = [0.0, 0.0, 0.0], 
    Kd = [0.0, 0.0, 0.0]
)



values = []
bestvalues = []
pbar = tqdm(total = 50*20*50*20)
# for Kpx in np.linspace(0, 10, 50):
Kpy = 6.0
Kdy = 1.42
for Kpx in np.linspace(0.1, 10, 50):
    for Kpz in np.linspace(0.1, 2, 20):
        for Kdx in np.linspace(0.1, 5, 50):
            for Kdz in np.linspace(0.1, 2, 20):
                pid = PID( 
                    Kp = [Kpy, Kpx, Kpz], 
                    Kd = [Kdy, Kdx, Kdz]
                )
                execute(pid)
                pbar.update(1)

json.dump(values, open('values.txt' , 'w'))