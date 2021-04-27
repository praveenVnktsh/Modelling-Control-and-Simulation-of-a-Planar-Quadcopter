  
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.thrust = nn.Linear(6, 1)
        self.tau = nn.Sequential(
            nn.Linear(6, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            )

    def forward(self, input):
        
        return self.thrust(input), self.tau(input)