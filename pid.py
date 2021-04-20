from quadcopter import Quadcopter, State
import numpy as np
g = 9.81
class PID():

    def __init__(self, dt, Kp= [0.0, 0.0, 0.0], Kd = [0.0, 0.0, 0.0], istate = State()):
        self.Kp = Kp
        self.dt = dt
        self.Kd = Kd
        self.preverror = np.array([0, 0], dtype= float)
        self.prevphi = istate.theta
        self.prevstate = istate

    def setSetpoint(self, setpoint):
        self.setpoint = setpoint

    def step(self, quad : Quadcopter, state : State):

        error = self.setpoint - np.array([state.x, state.y])


        yerror = error[1]
        yerrordot = (state.y - self.prevstate.y)/self.dt
        quad.thrust = quad.M * g + self.Kp[0] * yerror - self.Kd[0] * yerrordot



        xerror = error[0]
        xerrordot = (state.x - self.prevstate.x)/self.dt


        phi = -(-self.Kp[1] * xerror + self.Kd[1] * xerrordot)


        phiError = (phi - state.theta)

        phiErrorDot = (phi - self.prevphi)/self.dt - (state.theta - self.prevstate.theta)/self.dt

        quad.tau = self.Kp[2] * phiError + self.Kd[2] * phiErrorDot


        self.prevphi = phi

        self.prevstate = state

        complete = False
        if abs(yerror) + abs(xerror) < 0.02:
            complete = True

        return phi, complete
        # print(quad.thrust, quad.tau, phi, phiError, error, errordot)