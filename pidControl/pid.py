from quadcopter import Quadcopter, State
import numpy as np
g = 9.81
class PID():

    def __init__(self, Kp= [0.0, 0.0, 0.0], Kd = [0.0, 0.0, 0.0], istate : State = State()):
        self.Kp = Kp
        self.Kd = Kd
        self.preverror = np.array([0, 0], dtype= float)
        self.prevphi = istate.theta

    def setSetpoint(self, setpoint):
        self.setpoint = setpoint

    def step(self, quad : Quadcopter, state : State, dt):

        error = self.setpoint - np.array([state.x, state.y])
        yerror = error[1]

        quad.thrust = quad.M * g + self.Kp[0] * yerror - self.Kd[0] * state.ydot

        xerror = error[0]
        phi = -self.Kp[1] * xerror + self.Kd[1] * state.xdot
        
        phiError = (phi - state.theta)

        phiErrorDot = (phi - self.prevphi)/dt - state.thetadot

        quad.tau = self.Kp[2] * phiError + self.Kd[2] * phiErrorDot


        self.prevphi = phi


        complete = False

        if abs(yerror) + abs(xerror) < 0.05:
            self.completeindex += 1
        else:
            self.completeindex = 0

        if self.completeindex == 100:
            complete = True

        return phi, complete