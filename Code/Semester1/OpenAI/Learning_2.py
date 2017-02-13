#!/usr/bin/env ipython

import gym
import numpy as np
import GPy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pylab
import os

plt.rc('font',family='Times New Roman')
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

def generateAbsoluteError(absoluteError_1,absoluteError_2,absoluteError_3, attemptNumpy, predictionPDF, bufferObservations, attempt, attemptsTotal, i):
    if attempt > 0:
        absoluteError_1[attempt-1,i] =  100 * (np.abs(predictionPDF[0,0] - bufferObservations[0,0])) / bufferObservations[0,0]
        # print absoluteError_1[attempt-1,i]
        absoluteError_2[attempt-1,i] =  100 * (np.abs(predictionPDF[0,1] - bufferObservations[0,1])) / bufferObservations[0,1]
        absoluteError_3[attempt-1,i] =  100 * (np.abs(predictionPDF[0,2] - bufferObservations[0,2])) / bufferObservations[0,2]
        attemptNumpy[attempt-1,0] = attempt
    else:
        # absoluteError_1 = np.zeros((attemptsTotal,10))
        # absoluteError_2 = np.zeros((attemptsTotal,10))
        # absoluteError_3 = np.zeros((attemptsTotal,10))
        attemptNumpy = np.zeros((attemptsTotal,1))

    return absoluteError_1,absoluteError_2,absoluteError_3, attemptNumpy

def generateGraphs(looper, reset, dirNumber, attemptNumpy, absoluteErrorMean, inputsNumpy):
    while looper == True:
        if reset == True:
            if os.path.exists('./Learning_2_Figs/ResetAttempt%d' %dirNumber) == False:
                os.makedirs('./Learning_2_Figs/ResetAttempt%d' %dirNumber)
                looper = False
                os.chdir('./Learning_2_Figs/ResetAttempt%d' %dirNumber)
            else:
                dirNumber = dirNumber + 1
        if reset == False:
            if os.path.exists('./Learning_2_Figs/ContAttempt%d' %dirNumber) == False:
                os.makedirs('./Learning_2_Figs/ContAttempt%d' %dirNumber)
                looper = False
                os.chdir('./Learning_2_Figs/ContAttempt%d' %dirNumber)
            else:
                dirNumber = dirNumber + 1
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(attemptNumpy[:,0],absoluteErrorMean[:,0])
    plt.plot(attemptNumpy[:,0],absoluteErrorMean[:,1])
    plt.plot(attemptNumpy[:,0],absoluteErrorMean[:,2])
    ax.fill_between(attemptNumpy[:,0], absoluteErrorMean[:,0] + 2*errorVariance[:,0], absoluteErrorMean[:,0] - 2*errorVariance[:,0], color='#b9cfe7', edgecolor='#86B5F4',alpha=0.5)
    ax.fill_between(attemptNumpy[:,0], absoluteErrorMean[:,1] + 2*errorVariance[:,1], absoluteErrorMean[:,1] - 2*errorVariance[:,1], color='#77c171', edgecolor='#A6F486',alpha=0.5)
    ax.fill_between(attemptNumpy[:,0], absoluteErrorMean[:,2] + 2*errorVariance[:,2], absoluteErrorMean[:,2] - 2*errorVariance[:,2], color='#F37064', edgecolor='#777E77', alpha=0.5)
    plt.legend(['State 1', 'State 2', 'State 3'], loc='upper left')
    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Percentage Error / %', fontsize=16)
    pylab.savefig('./AbsoluteErrors.eps', format='eps')

    # fig1 = plt.figure()
    # ax = fig1.gca(projection='3d')
    # ax.scatter(attemptNumpy[:,0], inputsNumpy[1:,1], absoluteErrorMean[:,0], cmap=cm.coolwarm)
    # pylab.savefig('./Scatter.eps', format='eps')
    #
    # fig2 = plt.figure()
    # plt.hist(inputsNumpy[:,0], bins = 100)
    # # plt.legend('Control Input', loc='upper left')
    # pylab.savefig('./ControlInputHist.eps', format='eps')
    #
    # fig3 = plt.figure()
    # plt.hist(inputsNumpy[:,1],bins = 100)
    # # plt.legend('State 1', loc='upper left')
    # pylab.savefig('./State_1Hist.eps', format='eps')
    #
    # fig4 = plt.figure()
    # plt.hist(inputsNumpy[:,2],bins = 100)
    # # plt.legend('State 2', loc='upper left')
    # pylab.savefig('./State_2Hist.eps', format='eps')
    #
    # fig5 = plt.figure()
    # plt.hist(inputsNumpy[:,3], bins = 100)
    # # plt.legend('State 3', loc='upper left')
    # pylab.savefig('./State_3Hist.eps', format='eps')

#############################################################################################################################

#Set up standard parameters
standardEnvironment = "Pendulum-v0"
standardAttempts = "30"
standardReset = "True"

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
totalIterations = 10

absoluteError = np.zeros((attemptsTotal,3))
absoluteError_1 = np.zeros((attemptsTotal,totalIterations))
absoluteError_2 = np.zeros((attemptsTotal,totalIterations))
absoluteError_3 = np.zeros((attemptsTotal,totalIterations))
absoluteErrorMean = np.zeros((attemptsTotal,3))
errorVariance = np.zeros((attemptsTotal,3))
attemptNumpy = np.zeros([attemptsTotal,1])
predictions = np.zeros([1,3])
inputsNumpy = np.zeros((1,4))
observationsNumpy = np.zeros((1,3))
predictionPDF = np.zeros((1,3))

k = GPy.kern.RBF(input_dim=4, variance=1., lengthscale=1.)

for i in range(totalIterations):

    predictions = np.zeros([1, 3])
    inputsNumpy = np.zeros((1, 4))
    observationsNumpy = np.zeros((1, 3))
    predictionPDF = np.zeros((1, 3))
    k = GPy.kern.RBF(input_dim=4, variance=1., lengthscale=1.)

    print i
    for attempt in range(attemptsTotal+1):
        print i, 'attempt:', attempt

        if reset == True:
            observations = env.reset()

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
        (absoluteError_1,absoluteError_2,absoluteError_3, attemptNumpy) = generateAbsoluteError(absoluteError_1,absoluteError_2,absoluteError_3, attemptNumpy, predictionPDF, bufferObservations, attempt, attemptsTotal,i)

    # print absoluteError_1
    # plt.figure()
    # plt.plot(attemptNumpy[:,i],absoluteError_1[:,i])

# print absoluteError_1
for i in range(attemptsTotal):
    absoluteErrorMean[i,0] = np.average(absoluteError_1[i,:])
    errorVariance[i,0] = np.std(absoluteError_1[i,:])

    absoluteErrorMean[i,1] = np.average(absoluteError_2[i,:])
    errorVariance[i,1] = np.std(absoluteError_2[i,:])

    absoluteErrorMean[i,2] = np.average(absoluteError_3[i,:])
    errorVariance[i,2] = np.std(absoluteError_3[i,:])


generateGraphs(looper, reset, dirNumber, attemptNumpy, absoluteErrorMean, inputsNumpy)
