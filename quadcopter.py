import math
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

g = 9.81


class Quadcopter():

    def __init__(self, state=np.zeros(12)):
        I = 2.5e-4

        self.ode = scipy.integrate.ode(self.state_dot).set_integrator(
            'vode', nsteps=500, method='bdf')
        self.I = np.array(
            [
                [I, 0, 0],
                [0, I, 0],
                [0, 0, I]
            ])
        self.invI = np.linalg.inv(self.I)
        self.M = 0.18
        self.L = 0.086
        self.state = state
        self.thrust = 0
        self.tau = 0

    def rotation_matrix(self, angles):

        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_z = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def state_dot(self):

        state_dot = np.zeros(12)

        # The velocities(t+1 x_dots equal the t x_dots)
        state_dot[0] = self.state[3]
        state_dot[1] = self.state[4]
        state_dot[2] = self.state[5]

        # The acceleration
        x_dotdot = np.array([0, 0, -g]) + np.dot(self.rotation_matrix(
            self.state[6:9]), np.array([0, 0, self.thrust]))/self.M
        state_dot[3] = x_dotdot[0]
        state_dot[4] = x_dotdot[1]
        state_dot[5] = x_dotdot[2]

        # The angular rates(t+1 theta_dots equal the t theta_dots)
        state_dot[6] = 0*self.state[9]
        state_dot[7] = self.state[10]
        state_dot[8] = 0*self.state[11]

        # The angular accelerations
        omega = self.state[9:12]
        tau = np.array([0, self.tau, 0])

        omega_dot = np.dot(
            self.invI, (tau - np.cross(omega, np.dot(self.I, omega))))
        state_dot[9] = 0*omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = 0*omega_dot[2]
        return state_dot

    def wrap_angle(self, val):
        return ((val + np.pi) % (2 * np.pi) - np.pi)

    def step(self, dt, i, print_position=False):

        # Ode step:
        # [Waiting for Praveeen]
        self.ode.set_initial_value(self.state, 0)
        self.state = self.ode.integrate(self.ode.t + dt)
        self.state[6:9] = self.wrap_angle(self.state[6:9])

        # position
        if print_position and (i % 10 == 0):
            print(f"Current Px:{self.state[0]:2f} Py:{self.state[1]:2f} Pz:{self.state[2]:2f}",
                  f"Current Vx:{self.state[3]:2f} Vy:{self.state[4]:2f} Vz:{self.state[5]:2f}",
                  f"Angle:{self.state[6]:2f},{self.state[7]:2f},{self.state[8]:2f}")
        return self.state


if __name__ == "__main__":

    quad = Quadcopter()
    plt.figure(1)
    plt.ion()

    xval = []
    yval = []

    quad.thrust = 1 + quad.M * g
    quad.tau = 0.01
    plt.clf()
    # plt.ylim(-5, 100)
    # plt.xlim(-5, 5)

    for i in range(1000):
        state = quad.step(0.001)

        # if i > 10:
        #     quad.tau = 0

        xval.append(state[0])
        yval.append(state[2])

        # plt.pause(0.001)
    plt.plot(xval, label='X',  c='r')
    plt.plot(yval, label='Y', c='b')
    plt.grid()
    plt.legend()
    # plt.show()
    plt.show(block=True)
