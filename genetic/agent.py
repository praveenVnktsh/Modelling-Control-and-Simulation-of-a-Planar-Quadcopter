from quadcopter import State
from config import Args
from net import Net
import torch
import numpy as np

class Agent():
    def __init__(self, args:Args, device, stateDict = None):

        self.args = args
        self.net = Net().double().to(device)
        self.device = device
        if stateDict != None:
            self.net.load_state_dict(stateDict)

    def chooseAction(self, state : State):


        statearr = np.array([
            (state.setpoint[0] - state.x)/2.0,
            (state.setpoint[1] - state.y)/5.0,
            state.theta,
            state.xdot / 12.0,
            state.ydot / 12.0 ,
            state.thetadot,
        ])

        state = torch.from_numpy(statearr).double().to(self.device).unsqueeze(0)
    



        with torch.no_grad():
            thrust, tau = self.net(state)

        return (thrust.squeeze().item(), tau.squeeze().item(), )
    
    def getParams(self):
        return self.net.state_dict()