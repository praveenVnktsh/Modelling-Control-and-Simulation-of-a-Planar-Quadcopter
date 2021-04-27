import torch
import os
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Args():

    def __init__(self):
        # Logistical Parameters
        self.checkpoint = 24
        self.trial = 9
        self.test = False
        
        # evolution parameters
        
        self.mutationPower = 0.0001
        self.nAvg = 1

        
        # Settings
        self.nSurvivors = 50
        self.nAgents = 500
        if self.test:
            self.nAgents = 50
            self.nSurvivors = self.nAgents
            
        
        self.render = True
        self.deathThreshold = 1500



        #logistical parameters
        saveloc = 'model/train_' + str(self.trial) + '/'
        self.saveLocation = saveloc

        os.makedirs(self.saveLocation, exist_ok = True)
        f = open(saveloc + 'params.json','w')
        json.dump(self.getParamsDict(), f, indent=4, sort_keys=True, cls=NumpyEncoder)
        f.close()

    def getParamsDict(self):
        ret = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        return ret
    
        

def configure():
    
    args = Args()
    useCuda = torch.cuda.is_available()
    device = torch.device("cuda" if useCuda else "cpu")
    return args, useCuda, device