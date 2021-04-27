from numpy import random
from quadcopter import State
from comet_ml import Experiment
import numpy as np
from agent import Agent
from environment import quadEnv
from config import Args, configure
import torch
import os
import glob
import cv2
from tqdm import tqdm
import time
from tensorboardX import SummaryWriter


configs, use_cuda,  device = configure()

writer = SummaryWriter(log_dir= 'logs/run' +str(configs.trial))
    

def mutateWeightsAndBiases(agents, configs:Args):
    nextAgents = []

    if configs.test == True:
        for i in range(configs.nAgents):
            pair = agents[i]
            agentNet = Agent(configs, device, stateDict = pair[0].getParams())
            nextAgents.append(agentNet)
    else:
        for i in range(configs.nAgents):
            pair = agents[i % len(agents)]
            agentNet = Agent(configs, device, stateDict = pair[0].getParams())
            for param in agentNet.net.parameters():
                param.data += configs.mutationPower * torch.randn_like(param)
            nextAgents.append(agentNet)

    return nextAgents


def saveWeightsAndBiases(agentDicts, generation, configs:Args):
    loc = configs.saveLocation +'generation_'+str(generation) +  '/' 
    os.makedirs(loc, exist_ok = True)
    for i in range(len(agentDicts)):
        torch.save(agentDicts[i], loc + str(i) +  '-AGENT.pkl')



if __name__ == "__main__":
    print('-------------BEGINNING EXPERIMENT--------------')
    
    currentAgents = []
    
    if configs.checkpoint != 0:
        for location in sorted(glob.glob(configs.saveLocation +'generation_'+str(configs.checkpoint) +  '/*')):
            print('LOADING FROM',location)
            statedict = torch.load(location)
            currentAgents.append(statedict)
        
        currentAgents = mutateWeightsAndBiases(currentAgents, configs)
        print('-> Loaded agents from checkpoint', configs.checkpoint)
    else:
        for spawnIndex in range(configs.nAgents):
            agent = Agent(configs, device)
            currentAgents.append(agent)


    env = quadEnv()

    
    for generationIndex in range(configs.checkpoint, 100000):
        nextAgents = []
        rewards = []
        startTime = time.time()
        
        if configs.test:
            env.render = True
        
        
        setpoint = (0.0, 1.0)
        for agentIndex in tqdm(range(len(currentAgents))):
            # setpoint = (np.random.randint(-20,21) / 10.0, np.random.randint(20,51) / 10.0)
            
            state = State(setpoint = setpoint)
            fitness = 0.0

            if not configs.test:
                if (generationIndex) % 100 == 0 and agentIndex % 10 == 0:
                    env.render = True
                else:
                    env.render = False
            else:
                env.render = True
            state.theta = random.randint(-100, 100) / 100.0
            env.reset(setpoint, state.theta)

            for timestep in range(configs.deathThreshold):
                action = currentAgents[agentIndex].chooseAction(state)
                state, reward, dead, _ = env.step(action)
                fitness += reward
                # print(reward, fitness, dead)
                if dead:
                    break
            rewards.append(fitness)

            # print('Fitness of agent', agentIndex, '=', fitness)      


        
        logData = {
            'Generation':generationIndex, 
            'Timestep':timestep, 
            'Alive':int(configs.nAgents - np.sum(dead))  ,
            'Total agents' : configs.nAgents,
            'Fitness': np.round(np.mean(rewards), 3),
        }

        avgScore = np.mean(rewards)
        writer.add_scalar('fitness', np.mean(avgScore), global_step  = generationIndex)

        print('Generation', generationIndex,'Complete in ',time.time() - startTime , 'seconds')
        print('FITNESS = ', avgScore)
        
        
        if not configs.test:
            temp = [[currentAgents[agentIndex], rewards[agentIndex]] for agentIndex in range(len(currentAgents)) ]
            currentAgents = sorted(temp, key = lambda ag: ag[1], reverse = True)
            rewards = sorted(rewards, reverse = True)
            nextAgents = currentAgents[:configs.nSurvivors]
            print('Fitness of survivors = ', np.average([rewards[i] for i in range(len(nextAgents))]))
            currentAgents = mutateWeightsAndBiases(nextAgents, configs)
            if (generationIndex + 1) % 25 == 0:
                saveWeightsAndBiases(nextAgents, generationIndex, configs)
        

        print('---------------')