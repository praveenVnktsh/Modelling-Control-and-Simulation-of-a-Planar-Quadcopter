import math
from pid import PID
from quadcopter import Quadcopter, State
import numpy as np
import matplotlib.pyplot as plt


g = 9.81
stepsize = 0.005
plot = True
state = State()
quad = Quadcopter(state = state)

# y track, x track, phi track
# y - Rise time -0.36000000000000026 Overshoot % =  5.50947454441173 for Kp = 6.0, Kd = 1.42
# 
# 
pid = PID( 
    Kp = [6.00, 0.60, 1.5], 
    Kd = [1.5, 0.25, 0.1]
)





prevx = 0
prevy = 0
complete = False
i = 0
xval = []
yval = []
setpoints = [
    np.array([1.0, 1.0]),
]

totaltime = 0.0
ymax = 0.0
xmax = 0.0

plot = False


for setpoint in setpoints:
    complete = False
    pid.setSetpoint(setpoint)
    
    

    while not complete:

        state = quad.step(stepsize)
        totaltime += stepsize
        phi, complete = pid.step(quad, state, stepsize)
        ymax = max(ymax, state.y)
        xmax = max(xmax, state.x)

        if round(state.y, 1) == 0.10:
            startTimey = totaltime
        elif round(state.y, 1) == 0.90:
            timediffy = startTimey - totaltime

        if round(state.x, 1) == 0.10:
            startTimex = totaltime
        elif round(state.x, 1) == 0.90:
            timediffx = startTimex - totaltime
        
        xval.append(state.x)
        yval.append(state.y)
        
        info = {
            'x_des' : setpoint[0], 
            'y_des' : setpoint[1]
        }
        quad.render(info)

    time = np.linspace(0, totaltime, num = len(xval))

print('Rise time y', timediffy)
print('Overshoot y %  = ', (ymax - setpoints[0][1])*100/ setpoints[0][1])
print()
print('Rise time x', timediffx)
print('Overshoot x %  = ', (xmax - setpoints[0][0])*100/ setpoints[0][0])

if plot:
    plt.plot(time, [setpoints[0][0] for i in range(len(xval))],  label = 'X_des')
    plt.plot(time, [setpoints[0][1]  for i in range(len(xval))],  label = 'Y_des')
    plt.plot(time, xval, label = 'Y_out')
    plt.plot(time, yval, label = 'X_out')

    plt.grid()
    plt.title("Step Response, Closed Loop")
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.legend()

    plt.show()

