
#!/usr/bin/env ipython

import numpy as np
import matplotlib.pyplot as plt
import pylab
import time

plt.rc('font',family='Times New Roman')

import GPy

from math import *

def f(action, current_state):
    current_state = 10*(sin(action)) + current_state

    return current_state
total_it = 500
current_state = np.random.uniform(-pi,pi,1)
state_evolution = np.zeros((total_it,1))#, dtype='int64')
# input_states = np.zeros((total_it,2))#, dtype='int64')
# output_states = np.zeros((total_it,1))#, dtype='int64')
prediction_evolution = np.zeros((total_it,1))#, dtype='int64')
k = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)

for i in range(total_it):

    if i%50 == 0:
        print i
    action = np.random.uniform(-pi, pi, 1)

    # state_evolution[i] = current_state
    bufferInput = np.array([action,current_state])
    bufferInput = np.reshape(bufferInput, ([1, 2]))

    if i > 0:
        input_states = np.append(input_states, bufferInput, axis = 0)
    else:
        input_states = bufferInput

    current_state = f(action, current_state)
    bufferState = np.reshape(current_state, ([1,1]))
    if i > 0:
        output_states = np.append(output_states, bufferState, axis=0)
    else:
        output_states = bufferState

    if i > 0:
        # time.sleep(0.2)
        prediction = m.predict(bufferInput)[0]
        # print prediction
        prediction_evolution[i,0] = prediction

    m = GPy.models.GPRegression(input_states, output_states, k)
    m.optimize()

    action = np.random.uniform(-pi, pi, 1)


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(output_states)
ax1.plot(prediction_evolution)
ax2.plot((np.abs(prediction_evolution-output_states))/output_states*100,  'r',)

ax1.legend(['Real', 'Predicted'], loc='upper right')
ax1.set_xlabel('Time evolution / steps', fontsize=16)
ax1.set_ylabel('State', fontsize=16)
ax2.set_ylabel('Percentage Error / %', fontsize=16)
pylab.savefig('./stateEvolution.eps', format='eps')
plt.show()