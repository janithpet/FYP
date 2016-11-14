import gym
import numpy as np
import GPy
import pylab
import os

def setupEnvironment(**optionalParameters):
    if 'defaultEnvironment' in optionalParameters:
        defaultEnvironment = optionalParameters['defaultEnvironment']
    else:
        defaultEnvironment = 'Pendulum-v0'

    if 'setInputSpaceFeatures'in optionalParameters:
        setInputSpaceFeatures = optionalParameters['setInputSpaceFeatures']
        setInputSpaceFeatures = 'True' in setInputSpaceFeatures
    elif 'inputSpaceValues' in optionalParameters:
        inputSpaceValues = optionalParameters['inputSpaceValues']
        setInputSpaceFeatures = True
    else:
        setInputSpaceFeatures = False

    environmentToUse = raw_input('Please enter the environment: ')
    if environmentToUse == '':
        environmentToUse = defaultEnvironment

    env = gym.make(environmentToUse)
    observations = env.reset()
    action = env.action_space

    check = True

    if setInputSpaceFeatures == True:
        while check == True:
            if 'inputSpaceFeatures' in locals() == False:
                print 'The current input space is:', action
                print 'It has the following features:', vars(action)
                inputSpaceValues = input('Please enter new values for the features as a tuple of numpy arrays: ')
                if len(inputSpaceValues) != len(vars(action)):
                    print 'Your answer is missing a few variables:',  len(vars(action)) - len(inputSpaceValues)
                else:
                    check = False
        counter = 0
        for key, value in vars(action).iteritems():
            dummyFeatureValue = getattr(action, key)
            for i in range(len(dummyFeatureValue.shape)):
                dummyNewValue = inputSpaceValues[counter][i]
                setattr(action, key, dummyNewValue)

    dummySpaceDiscrete= gym.spaces.Discrete(2)
    dummySpaceBox = gym.spaces.Box(2,-2, shape=(1,))

    observationsNumpy = np.zeros((1, observations.shape[0]))
    predictionPDF = np.zeros((1, observations.shape[0]))

    if type(dummySpaceBox) == type(action):
        inputsNumpy = np.zeros((1, observations.shape[0] + action.shape[0]))
    elif type(dummySpaceDiscrete) == type(action):
        inputsNumpy = np.zeros((1,observations.shape[0] + 1))

    testSample = action.sample()
    [testObservation, testRewards, done, info] = env.step(testSample)
    rewardsNumpy = np.zeros((1,testRewards.size))

    return env, observations, inputsNumpy, observationsNumpy, rewardsNumpy, action, predictionPDF

def appendInputArray(inputsNumpy, action, observations, attempt):
    actionToTake = action.sample()

    if type(actionToTake) == int:
        bufferActionToTake = np.array([actionToTake])
    else:
        bufferActionToTake = actionToTake
    bufferAction = np.array([bufferActionToTake])
    bufferStateCurrent = np.asarray([observations])
    
    bufferInput = np.append(bufferAction, bufferStateCurrent, axis = 1)

    if attempt == 0:
        inputsNumpy = bufferInput
    elif attempt > 0:
        inputsNumpy = np.append(inputsNumpy, bufferInput, axis = 0)

    return inputsNumpy, bufferInput, actionToTake

def appendRewardsArray(rewardsNumpy, rewards, attempt):
    bufferRewards = np.asarray([rewards])
    bufferRewards = np.reshape(bufferRewards, ([1,1]))

    if attempt == 0:
        rewardsNumpy = bufferRewards
    elif attempt > 0:
        rewardsNumpy = np.append(rewardsNumpy, bufferRewards, axis = 0)

    return rewardsNumpy, bufferRewards

def appendObservationsArray(observationsNumpy, observations, attempt):
    bufferObservations = np.asarray([observations])

    if attempt == 0:
        observationsNumpy = bufferObservations
    elif attempt > 0:
        observationsNumpy = np.append(observationsNumpy, bufferObservations, axis = 0)

    return observationsNumpy, bufferObservations

def appendPolicyParameters(policyParameters, policyParametersNumpy, attempt_policy):
    if attempt_policy == 0:
        policyParametersNumpy = policyParameters

    policyParametersNumpy = np.append(policyParametersNumpy, policyParameters, axis = 0)

def appendTotalRewards(totalRewards, totalRewardsNumpy, attempt_policy):
    if attempt_policy == 0:
        totalRewardsNumpy = totalRewards

    totalRewardsNumpy = np.append(totalRewardsNumpy, totalRewards, axis = 0)

def generateModel(inputsNumpy, observationsNumpy, k):
    m = GPy.models.GPRegression(inputsNumpy, observationsNumpy, k)
    m.optimize()

    return m

def generatePrediction(m, bufferInput, predictionPDF, attempt):
    if attempt > 0:
        predictionPDF = m.predict(bufferInput)[0]
    else:
        predictionPDF = predictionPDF
    return predictionPDF

def generateAbsoluteErrorMean(absoluteErrorMean, attemptNumpy, predictionPDF, bufferObservation, attempt):
    if attempt > 0:
        absoluteErrorMean[attempt-1, :] = np.abs(predictionPDF[0,:] - bufferObservations)
        attemptNumpy[attempt-1, :] = attempt
    else:
        absoluteErrorMean = absoluteErrorMean
        attemptNumpy = attemptNumpy

    return absoluteErrorMean, attemptNumpy

def createNewFolder(analysisType, **optionalParameters):
    if analysisType == 'Batch':
        if 'batchSampleSize' in optionalParameters:
            batchSampleSize = int(optionalParameters['batchSampleSize'])
        else:
            print 'Please enter batchSampleSize for Batch analysis.'

        if 'batchResetNumber' in optionalParameters:
            batchResetNumber = int(optionalParameters['batchResetNumber'])
        else: print 'Please enter batchResetNumber for Batch analysis.'

    if 'saveDirectoryRoot' in optionalParameters:
        saveDirectoryRoot = 'True' in optionalParameters['saveDirectoryRoot']
        saveDirectory = './'

    if 'saveDirectory' in optionalParameters:
        saveDirectory = optionalParameters['saveDirectory']
    elif saveDirectoryRoot == False:
        saveDirectory = raw_input('Please enter the directory to create folder in: ')
    else:
        saveDirectory = './'
    if saveDirectory != '':
        saveDirectory = '%s/' %saveDirectory

    looper == True
    dirNumber = 0

    while looper == True:
        if analysisType == 'Continuous':
            if os.path.exists('%sContAttempt%d' %(saveDirectory, dirNumber)) == False:
                os.makedirs('%sContAttempt%d,' %(saveDirectory, dirNumber))
                directory = ('./%sContAttempt%d/,' %(saveDirectory, dirNumber))
                return directory
                looper = False
            else:
                dirNumber = dirNumber + 1
        if analysisType == 'Reset':
            if os.path.exists('%sResetAttempt%d' %(saveDirectory, dirNumber)) == False:
                os.makedirs('%sResetAttempt%d,' %(saveDirectory, dirNumber))
                directory = ('%sResetAttempt%d/,' %(saveDirectory, dirNumber))
                return directory
                looper = False
            else:
                dirNumber = dirNumber + 1
        if analysisType == 'Batch':
            if os.path.exists('%sBatchAttempt%d_%d-%d' %(saveDirectory, dirNumber, batchResetNumber, batchSampleSize)) == False:
                os.makedirs('%sBatchAttempt%d_%d-%d' %(saveDirectory, dirNumber, batchResetNumber, batchSampleSize))
                directory = ('%sBatchAttempt%d_%d-%d/' %(saveDirectory, dirNumber, batchResetNumber, batchSampleSize))
                looper = False
                return directory
            else:
                dirNumber = dirNumber + 1

def saveGraph(figure, fileName, **optionalParameters):
    if 'saveFigureRoot' in optionalParameters:
        saveFigureRoot = ['True'] in optionalParameters['saveFigureRoot']
        directory = './'

    if 'directory' in optionalParameters:
        directory = optionalParameters['directory']
    elif saveFigureRoot == False:
        directory = raw_input('Please enter where to save your figure')
    else:
        directory = './'
    if directory != '':
        directory = '%s/' %directory

    pylab.savefig('%s%s.svg' %(directory, fileName), format = 'svg')
    plyab.savefig('%s%s.pdf' %(directory, fileName), format = 'pdf')
