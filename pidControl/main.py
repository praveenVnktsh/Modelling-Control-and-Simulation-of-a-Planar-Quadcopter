import math
from pid import PID
from quadcopter import Quadcopter, State
import numpy as np
import scipy.integrate
from tqdm import tqdm
import cv2


g = 9.81
stepsize = 0.005
plot = True
state = State()
quad = Quadcopter(state = state)

# z track, x track, phi track
pid = PID( 
    Kp = [2.0, 0.3, 1.0], 
    Kd = [1.0, 0.25, 0.1]
)




xval = []
yval = []




thetaval = []
prevx = 0
prevy = 0
complete = False
i = 0
setpoints = [
    np.array([1.0, 1.0]),
    np.array([0, 0.5]),
    np.array([-1.0, 1.5]),
    np.array([0, 0.0]),
]
for setpoint in setpoints:
    complete = False
    pid.setSetpoint(setpoint)
    while not complete:

        state = quad.step(stepsize)

        phi, complete = pid.step(quad, state, stepsize)
        
        info = {
            'x_des' : setpoint[0], 
            'y_des' : setpoint[1]
        }
        quad.render(info)

print('Simulation complete')
