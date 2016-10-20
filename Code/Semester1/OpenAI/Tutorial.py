#!/usr/bin/env ipython

import gym
import time
import numpy as np
import GPy

env = gym.make('Pendulum-v0')
env.reset()
x = env.action_space
action = gym.spaces.Box(-2, 2, shape=(1,))

observationsNumpy = np.zeros((1,3))
inputsNumpy = np.zeros((1,1))

for _ in  range(10):
    env.render()
    actionToTake = action.sample()
    bufferAction = np.array([actionToTake])
    inputsNumpy = np.append(inputsNumpy, bufferAction, axis = 0)

    observations, rewards, done, info = env.step(actionToTake)

    bufferObservations = np.asarray([observations])
    observationsNumpy = np.append(observationsNumpy, bufferObservations, axis = 0)

print inputsNumpy.shape
print observationsNumpy.shape

k = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(inputsNumpy,observationsNumpy,k)

print(m)
    # time.sleep(0.1)
    # if done == True:
    #     env.reset()
