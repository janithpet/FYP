#!/usr/bin/env ipython

import gym
import time
import numpy as np
import GPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import pylab

env = gym.make('Pendulum-v0')
observations = env.reset()
x = env.action_space
action = gym.spaces.Box(-2, 2, shape=(1,))

attemptsTotal = 500
dirNumber = 0
looper = True


reset = True

# inputsNumpy = np.zeros((1,4))
# observationsNumpy = np.zeros((1,3))

absoluteError = np.zeros((attemptsTotal,3))
absoluteErrorMean = np.zeros((attemptsTotal,3))
attemptNumpy = np.zeros([attemptsTotal,1])
predictions = np.zeros([1,3])

k = GPy.kern.RBF(input_dim=4, variance=1., lengthscale=1.)

for attempt in  range(attemptsTotal+1):

    if reset == True:
        observations = env.reset()

    env.render()
    if attempt == 0:
        actionToTake = action.sample()
        bufferAction = np.array([actionToTake])
        bufferStateCurrent = np.asarray([observations])
        bufferInput = np.append(bufferAction,bufferStateCurrent, axis = 1)

        inputsNumpy = bufferInput

        observations, rewards, done, info = env.step(actionToTake)

        bufferObservations = np.asarray([observations])
        observationsNumpy = bufferObservations

        m = GPy.models.GPRegression(inputsNumpy,observationsNumpy,k)
        m.optimize()

    if attempt > 0:
        actionToTake = action.sample()
        bufferAction = np.array([actionToTake])
        bufferStateCurrent = np.asarray([observations])
        bufferInput = np.append(bufferAction,bufferStateCurrent, axis = 1)
        # print(absoluteError.shape, attemptNumpy.shape)


        inputsNumpy = np.append(inputsNumpy, bufferInput, axis = 0)

        predictionPDF = m.predict(bufferInput)[0]
        (predictionRows, predictionColumns) = predictionPDF.shape

        observations, rewards, done, info = env.step(actionToTake)

        bufferObservations = np.asarray([observations])
        observationsNumpy = np.append(observationsNumpy, bufferObservations, axis = 0)

        m = GPy.models.GPRegression(inputsNumpy, observationsNumpy, k)
        m.optimize()

        for i in range(predictionColumns):
            predictions[0,i] = np.random.normal(predictionPDF[0,i], np.sqrt(predictionPDF[1,i]))

        absoluteError[attempt-1,:] = np.abs(predictions - bufferObservations)
        absoluteErrorMean[attempt-1,:] = np.abs(predictionPDF[0,:] - bufferObservations)
        attemptNumpy[attempt-1,0] = attempt

    print(attempt)

# print(absoluteError.shape, attemptNumpy.shape)
# plt.plot(attemptNumpy[:,0],absoluteError[:,0])

while looper == True:
    if reset == True:
        if os.path.exists('./Learning_1_Figs/ResetAttempt%d' %dirNumber) == False:
            os.makedirs('./Learning_1_Figs/ResetAttempt%d' %dirNumber)
            looper = False
            os.chdir('./Learning_1_Figs/ResetAttempt%d' %dirNumber)
        else:
            dirNumber = dirNumber + 1
    if reset == False:
        if os.path.exists('./Learning_1_Figs/ContAttempt%d' %dirNumber) == False:
            os.makedirs('./Learning_1_Figs/ContAttempt%d' %dirNumber)
            looper = False
            os.chdir('./Learning_1_Figs/ContAttempt%d' %dirNumber)
        else:
            dirNumber = dirNumber + 1

plt.plot(attemptNumpy[:,0],absoluteErrorMean[:,0])
plt.plot(attemptNumpy[:,0],absoluteErrorMean[:,1])
plt.plot(attemptNumpy[:,0],absoluteErrorMean[:,2])
plt.legend(['State 1', 'State 2', 'State 3'], loc='upper left')
pylab.savefig('./AbsoluteErrors')

fig1 = plt.figure()
ax = fig1.gca(projection='3d')
ax.scatter(attemptNumpy[:,0], inputsNumpy[1:,1], absoluteErrorMean[:,0], cmap=cm.coolwarm)
pylab.savefig('./Scatter')

fig2 = plt.figure()
plt.hist(inputsNumpy[:,0], bins = 100)
# plt.legend('Control Input', loc='upper left')
pylab.savefig('./ControlInputHist')

fig3 = plt.figure()
plt.hist(inputsNumpy[:,1],bins = 100)
# plt.legend('State 1', loc='upper left')
pylab.savefig('./State_1Hist')

fig4 = plt.figure()
plt.hist(inputsNumpy[:,2],bins = 100)
# plt.legend('State 2', loc='upper left')
pylab.savefig('./State_2Hist')

fig5 = plt.figure()
plt.hist(inputsNumpy[:,3], bins = 100)
# plt.legend('State 3', loc='upper left')
pylab.savefig('./State_3Hist')
