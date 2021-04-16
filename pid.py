from quadcopter import Quadcopter
import numpy as np
g = 9.81
class PID():

    def __init__(self, dt, Kp= 0.0, Ki= 0.0, Kd = 0.0):
        self.Kp = Kp
        self.dt = dt
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prevErrors = np.array([0, 0], dtype= float)

    def setSetpoint(self, setpoint):
        self.setpoint = setpoint

    def step(self, quad : Quadcopter, state):

        errors = self.setpoint - np.array([state[7], state[2]])
        self.integral += errors


        quad.thrust = quad.M * g + self.Kp[1] * errors[1] + self.Kd[1] * (errors[1] - self.prevErrors[1])/self.dt  + self.Ki[1] * (self.integral[1])
        quad.tau = self.Kp[0] * errors[0] + self.Kd[0] * (errors[0] - self.prevErrors[0])/self.dt  + self.Ki[0] * (self.integral[0])

        # quad.thrust = quad.M * g + self.Kp[1] * errors[1] + self.Kd[1] * (errors[1] - self.prevErrors[1])/self.dt  + self.Ki[1] * (self.integral[1])
        # quad.tau = self.Kp[0] * errors[0] + self.Kd[0] * (errors[0] - self.prevErrors[0])/self.dt  + self.Ki[0] * (self.integral[0])
        

        # f1 = 0.5*(quad.thrust + quad.tau/quad.L)
        # f2 = 0.5*(quad.thrust - quad.tau/quad.L)

        self.prevErrors = errors