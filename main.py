import math
from quadcopter import Quadcopter
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":
    g = 9.81
    quad = Quadcopter()
    
    plt.ion()
    
    xval = []
    yval = []
    
    quad.thrust =  1 + quad.M * g
    quad.tau = 0.001

    timesteps = 1000


    plt.figure(1)
    plt.clf()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()
    plt.grid()

    for i in tqdm(range(timesteps)):
        state = quad.step(0.001, False)

        # if i > 30:
        #     quad.tau = 0
        x = state[0]
        y = state[2]

        theta = state[7]

        xval.append(x)
        yval.append(y)

        if i % (timesteps //100)  == 0: 
            plt.clf()
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.plot(xval, yval, label = 'X',  c = 'r')
            plt.plot([-quad.L*np.cos(theta) + x, quad.L*np.cos(theta) + x], [y + quad.L*np.sin(theta), y - quad.L*np.sin(theta)], c= 'b', linewidth = 2.5)
            plt.pause(0.001)

        


    plt.figure(2)
    plt.plot(xval, label = 'X',  c = 'r')
    plt.plot(yval, label = 'Y', c = 'b')
    plt.grid()
    plt.legend()
    # plt.show()
    plt.show(block = True)