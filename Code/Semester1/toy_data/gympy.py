import numpy as np
import matplotlib.pyplot as plt
import pylab

plt.rc('font', family='Times New Roman')

import GPy

class rl_components:

    def __init__(self, dimensions_input, dimensions_output, model_simulator=None, action_from_policy=None, reward_function=None, termination_function = None, goal_states=None, initial_states=None):
        self.d_in = dimensions_input
        self.d_out = dimensions_output
        self.sim = model_simulator
        self.action_from_policy = action_from_policy
        self.reward_function = reward_function
        self.goal_states = goal_states
        self.termination_function = termination_function

        self.model= None
        self.train_inputs = None
        self.train_outputs = None

        self.action = None
        if initial_states == None:
            self.initial_states= np.zeros((1,d_out))
        else:
            self.initial_states = initial_states
        self.current_states = self.initial_states
        self.state_evolution = self.initial_states
        self.variance_evolution = self.initial_states - self.initial_states
        self.current_variance = self.initial_states - self.initial_states
        self.current_inputs = None

        self.policy_parameters = None
        self.goal_counter = 0
        self.total_counter = 0
        self.total_reward = 0

        self.current_reward = 0
        self.progress_counter = 0
        self.progress_print = False

    def generate_input(self):
        if self.action == None:
            return None

        else:
            current_inputs = np.append(self.action, self.current_states, axis = 1)
            current_inputs = np.reshape(current_inputs, ([1, self.d_in]))

            return current_inputs

    def update_inputs(self):
        if self.train_inputs == None:
            train_inputs = self.current_inputs

        else:
            train_inputs = np.append(self.train_inputs, self.current_inputs, axis = 0)

        return train_inputs

    def update_outputs(self):
        if self.train_outputs == None:
            train_outputs = self.current_states

        else:
            train_outputs = np.append(self.train_outputs, self.current_states, axis = 0)

        return train_outputs

    def update_evolution(self, reset=None):
        if reset == None or reset == False:
            state_evolution = np.append(self.state_evolution, self.current_states, axis = 0)
            variance_evolution = np.append(self.variance_evolution, self.current_variance, axis = 0)

            [r,c] = state_evolution.shape
            [r1,c1] = variance_evolution.shape
            state_evolution = np.reshape(state_evolution, ([r, self.d_out]))
            variance_evolution = np.reshape(variance_evolution, ([r, self.d_out]))
        else:
            state_evolution = self.initial_states
            variance_evolution = self.initial_states - self.initial_states

            self.current_states = self.initial_states
            self.current_variance = self.current_variance

        return state_evolution, variance_evolution

    def generate_training_data_random(self, iterations, action_min, action_max, action_dim):

        for i in range(iterations):
            self.action = np.random.uniform(action_min, action_max, action_dim)
            self.action = np.reshape(self.action, ([1,action_dim]))

            self.current_inputs = self.generate_input()
            self.train_inputs = self.update_inputs()

            self.current_states = self.sim(self.action, self.current_states)
            self.train_outputs = self.update_outputs()

        self.current_inputs = None
        self.current_states = self.initial_states
        print '%d training data gathered' %iterations

    def generate_training_data_continuous(self, iterations):
        self.current_states = self.initial_states

        for i in range(iterations):
            [r,c] = self.action.shape
            self.action = self.action_from_policy(self.policy_parameters, self.current_states)
            self.action = np.reshape(self.action, ([1,c]))

            self.current_inputs = self.generate_input()
            self.train_inputs = self.update_inputs()

            self.current_states = self.model.predict(self.current_inputs)[0]
            self.current_states = np.reshape(self.current_states, ([1, self.d_out]))
            self.train_outputs = self.update_outputs()

        self.current_inputs = None
        self.current_states = self.initial_states
        print '%d new data added' %iterations

    def objective_function(self, policy_parameters):
        terminate = False

        self.state_evolution, self.variance_evolution = self.update_evolution(reset=True)
        self.current_states = self.initial_states
        self.goal_counter = 0
        self.total_counter = 0
        self.current_reward = 0
        self.total_reward = 0

        self.action = self.action_from_policy(policy_parameters, self.current_states)
        self.current_inputs = self.generate_input()

        while terminate == False:
            self.current_states = self.model.predict(self.current_inputs)[0]
            self.current_states = np.reshape(self.current_states, ([1,1]))

            self.state_evolution, self.variance_evolution = self.update_evolution(reset=False)

            self.current_reward = self.reward_function(self.current_states, self.goal_states)
            self.total_reward += self.current_reward

            self.current_termination = self.termination_function(self)
            terminate = self.current_termination.check()

            self.action = self.action_from_policy(policy_parameters, self.current_states)
            self.current_inputs = self.generate_input()

        self.progress_counter += 1
        if self.progress_print == True:
            print self.goal_counter, '\t', self.total_counter, '\t', self.progress_counter ,'\t', self.current_reward
        return self.total_reward


class plot:
    def __init__(self, rl_components):
        self.inheritence = rl_components
        self._plot_counter = None

    def predicted_path_GP(self, k, to_print=False):
        [r,c] = self.inheritence.train_inputs.shape
        self.inheritence.state_evolution, self.inheritence.variance_evolution = self.inheritence.update_evolution(reset=True)

        for i in range(r-1):
            self.inheritence.model = GPy.models.GPRegression(self.inheritence.train_inputs[:i+1], self.inheritence.train_outputs[:i+1], k)
            if k.name != 'linear':
                self.inheritence.model.optimize()
            self.inheritence.current_states = self.inheritence.model.predict(np.reshape(self.inheritence.train_inputs[i+1],([1,2])))[0]
            self.inheritence.current_variance = self.inheritence.model.predict(np.reshape(self.inheritence.train_inputs[i+1],([1,2])))[1]
            if to_print == True:
                print self.inheritence.current_variance
            self.inheritence.state_evolution, self.inheritence.variance_evolution = self.inheritence.update_evolution(reset=False)
            if to_print == True:
                print 'Completed %d iterations' %i

        absoluteErrorMean = ((np.abs(self.inheritence.state_evolution[1:]-self.inheritence.train_outputs[1:]))*(np.abs(self.inheritence.state_evolution[1:]-self.inheritence.train_outputs[1:])))

        plt.close()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(self.inheritence.train_outputs[0:])
        ax1.plot(self.inheritence.state_evolution[0:])
        ax2.plot(absoluteErrorMean,  'r')


        ax1.fill_between(range(self.inheritence.state_evolution.shape[0]), self.inheritence.state_evolution[:,0] + 2*np.sqrt(np.abs(self.inheritence.variance_evolution[:,0])), self.inheritence.state_evolution[:,0] - 2*np.sqrt(np.abs(self.inheritence.variance_evolution[:,0])), color='g', edgecolor='#86B5F4',alpha=0.5)

        ax1.legend(['Real', 'Predicted'], loc='upper right')
        ax1.set_xlabel('Time / units', fontsize=16)
        ax1.set_ylabel('State', fontsize=16)
        ax2.set_ylabel('Square Absolute Error', fontsize=16)
        pylab.savefig('./Plots/allplot_%d.svg' %self.plot_counter, format='svg')
        plt.show()



    def run_simulation_true(self, iterations):
        self.inheritence.state_evolution, self.inheritence.variance_evolution = self.inheritence.update_evolution(reset = True)
        print self.inheritence.policy_parameters
        for i in range(iterations):
            self.inheritence.action = self.inheritence.action_from_policy(self.inheritence.policy_parameters, self.inheritence.current_states)
            self.inheritence.current_states = self.inheritence.sim(self.inheritence.action, self.inheritence.current_states)
            self.inheritence.state_evolution, self.inheritence.variance_evolution = self.inheritence.update_evolution(reset = False)

        plt.close()
        plt.figure()

        plt.plot(self.inheritence.state_evolution)
        plt.xlabel('Time / units', fontsize=16)
        plt.ylabel('State / unit', fontsize=16)

        pylab.savefig('./Plots/final_solution_%d.svg' %self.plot_counter, format='svg')
