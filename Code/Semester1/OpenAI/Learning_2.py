#!/usr/bin/env ipython

import gym
import numpy as np
import GPy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pylab
import os

#define functions
def generateInputs(inputsNumpy, action, observations, attempt):
    actionToTake = action.sample()
    bufferAction = np.array([actionToTake])
    bufferStateCurrent = np.asarray([observations])
    bufferInput = np.append(bufferAction, bufferStateCurrent, axis = 1)

    if attempt == 0:
        inputsNumpy = bufferInput
    elif attempt > 0:
        inputsNumpy = np.append(inputsNumpy, bufferInput, axis = 0)

    return inputsNumpy, bufferInput, actionToTake

def generateObservations(observationsNumpy, observations, attempt):
    bufferObservations = np.asarray([observations])

    if attempt == 0:
        observationsNumpy = bufferObservations
    elif attempt > 0:
        observationsNumpy = np.append(observationsNumpy, bufferObservations, axis = 0)

    return observationsNumpy, bufferObservations

def generateModel(inputsNumpy, observationsNumpy, k):
    m = GPy.models.GPRegression(inputsNumpy, observationsNumpy, k)
    m.optimize()

    return m

def generatePrediction(m, bufferInput, attempt):
    predictionPDF = m.predict(bufferInput)[0]

    return predictionPDF

def generateAbsoluteError(absoluteErrorMean, attemptNumpy, predictionPDF, bufferObservations, attempt, attemptsTotal):
    if attempt > 0:
        absoluteErrorMean[attempt-1,:] = np.abs(predictionPDF[0,:] - bufferObservations)
        attemptNumpy[attempt-1,0] = attempt
    else:
        absoluteErrorMean = np.zeros((attemptsTotal,3))
        attemptNumpy = np.zeros((attemptsTotal,1))

    return absoluteErrorMean, attemptNumpy

def generateGraphs(looper, reset, dirNumber, attemptNumpy, absoluteErrorMean, inputsNumpy):
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

#############################################################################################################################

#Set up standard parameters
standardEnvironment = "Pendulum-v0"
standardAttempts = "50"
standardReset = "False"

#Set up environment
environmentSetUp = raw_input('Please enter the environment: ')
if environmentSetUp == "":
    environmentSetUp = standardEnvironment
env = gym.make(environmentSetUp)
observations = env.reset()
action = gym.spaces.Box(-2,2, shape=(1,))

#Set up program parameters
attemptsTotal = raw_input('Please enter the number of tests: ')
if attemptsTotal == "":
    attemptsTotal = standardAttempts
attemptsTotal = int(attemptsTotal)
reset = raw_input('Reset every interation?:  ')
if reset == "":
    reset = standardReset
reset = reset in ['True']

looper = True
dirNumber = 0

#set up initial variables

absoluteError = np.zeros((attemptsTotal,3))
absoluteErrorMean = np.zeros((attemptsTotal,3))
attemptNumpy = np.zeros([attemptsTotal,1])
predictions = np.zeros([1,3])
inputsNumpy = np.zeros((1,4))
observationsNumpy = np.zeros((1,3))
predictionPDF = np.zeros((1,3))

k = GPy.kern.RBF(input_dim=4, variance=1., lengthscale=1.)


for attempt in range(attemptsTotal+1):
    print attempt

    if reset == True:
        observations = env.reset()
    env.render()

    #generate input array
    (inputsNumpy, bufferInput, actionToTake) = generateInputs(inputsNumpy, action, observations, attempt)

    #generate predictions if present attempt isn't the first
    if attempt > 0:
        predictionPDF = generatePrediction(m, bufferInput, attempt)

    #carry out simulation
    (observations, rewards, done, info) = env.step(actionToTake)

    #generate output array
    (observationsNumpy, bufferObservations) = generateObservations(observationsNumpy, observations, attempt)

    #generate GP model
    m = generateModel(inputsNumpy, observationsNumpy, k)

    #generate performance arrays
    (absoluteErrorMean, attemptNumpy) = generateAbsoluteError(absoluteErrorMean, attemptNumpy, predictionPDF, bufferObservations, attempt, attemptsTotal)

generateGraphs(looper, reset, dirNumber, attemptNumpy, absoluteErrorMean, inputsNumpy)
