import numpy as np
import matplotlib.pyplot as plt
import pylab

#This sets the font of any plots from pylab to be 'Times New Roman' for consistency.
plt.rc('font', family='Times New Roman')

import GPy

#In order to start the RL, we must be able to set up the appropriate components. The following class allows for seemless transition between each.
#It takes as inputs:
#   1)dimensions_input:     The dimensionality of the set labelled as 'inputs'. As as example, consider the state + the action
#   2)dimensions_output:    The dimensionality of the set labelled as 'outputs'. For the above case, it would be the output state.
#   3)model_simulator:      A function that represents the real system. This could be a simple formula, or something that grabs an input from sensors. It should be able to take in vectors of dimension 1), and output a vector of dimension 2).
#   4)action_from_policy:   A function that defines the policy. This should take in a vector of dimension 2) and a vector representing the policy parameters, and produce an output. For now this is to be one dimensional.
#   5)reward_function:      A function that defines the reward/cost function. This should take in the current state, and goal state (dimension 2)) and output a number that is the reward/consistency
#   6)termination_function: A class that defines the termination condition. This uses a class because it allows this condition to be defined in terms of any of the variables stored within the class.
#   7)goal_states:          The goal states, as a vector of size 2)
#   8)initial_states:       The initial states, as a vector of size 2)
#   9)error_limit:          This is a number that sets the error limit.

#To run the combination of model free and model based RL, use objective_function_foo. The present version does not evaluate the long term variance. If only wanting model based, run objective_function.

class rl_components:

    def __init__(self, dimensions_input, dimensions_output, model_simulator=None, action_from_policy=None, reward_function=None, termination_function = None, goal_states=None, initial_states=None, error_limit=None):
        self.d_in = dimensions_input
        self.d_out = dimensions_output
        self.sim = model_simulator
        self.action_from_policy = action_from_policy
        self.reward_function = reward_function
        self.goal_states = goal_states
        self.termination_function = termination_function
        self.error_limit = error_limit

        self.GP_kernel = None
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

        if self.error_limit == None:
            self.error_limit = 10

        self.action_evolution = None
        self.inputs_evolution = None
        self.predicition_evolution = None
        self.use_model = False

        self.total_reward_evolution = np.zeros([1,1])
        self.total_reward_evolution = np.reshape(self.total_reward_evolution, ([1,1]))

        self.previous_states = self.current_states
        self.true_outputs = None

        self.use_differences = False


    def generate_input(self):
        if self.action == None:
            return None

        else:
            current_inputs = np.append(self.action, self.current_states, axis = 1)
            current_inputs = np.reshape(current_inputs, ([1, self.d_in]))

            return current_inputs

    def generate_input_evolution(self):
        if self.action_evolution == None:
            return None

        else:
            inputs_evolution = np.append(self.action_evolution, self.state_evolution, axis = 1)
            inputs_evolution = np.reshape(inputs_evolution, (self.action_evolution.size, self.d_in))

            return inputs_evolution

    def update_inputs(self):
        if self.train_inputs == None:
            train_inputs = self.current_inputs

        else:
            train_inputs = np.append(self.train_inputs, self.current_inputs, axis = 0)

        return train_inputs

    def update_outputs(self):
        if self.train_outputs == None:
            print self.previous_states, self.current_states
            train_outputs = self.differences_for_gp()

        else:
            train_outputs = np.append(self.train_outputs, self.differences_for_gp(), axis = 0)

        if self.true_outputs == None:
            true_outputs = self.current_states
        else:
            true_outputs = np.append(self.true_outputs, self.current_states, axis = 0)

        return train_outputs, true_outputs

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

            self.previous_states = self.current_states

            self.current_states = self.initial_states
            self.current_variance = self.current_variance

        return state_evolution, variance_evolution

    def generate_training_data_random(self, iterations, action_min, action_max, action_dim, policy_parameters=None):
        action_set = np.linspace(action_min,action_max,iterations)
        action_set = np.random.uniform(action_min, action_max, iterations)
        for i in range(iterations):
            if policy_parameters == None:
                action = action_set[i]
            else:
                action = self.action_from_policy(policy_parameters, self.current_states)

            self.action = np.array(action)
            self.action = np.reshape(self.action, ([1,action_dim]))

            self.current_inputs = self.generate_input()
            self.train_inputs = self.update_inputs()
            #print self.current_inputs
            self.previous_states = self.current_states
            self.current_states = self.sim(self.action, self.current_states)
            self.train_outputs, self.true_outputs = self.update_outputs()

        self.current_inputs = None
        self.current_states = self.initial_states
        print '%d training data gathered' %iterations

    def generate_training_data_continuous(self, iterations):
        self.previous_states = self.current_states
        self.current_states = self.initial_states

        for i in range(iterations):
            [r,c] = self.action.shape
            self.action = self.action_from_policy(self.policy_parameters, self.current_states)
            self.action = np.reshape(self.action, ([1,c]))

            self.current_inputs = self.generate_input()
            self.train_inputs = self.update_inputs()

            self.previous_states = self.current_states
            self.current_states = self.current_states_from_gp(self.current_inputs)
            self.current_states = np.reshape(self.current_states, ([1, self.d_out]))
            self.train_outputs, self.true_outputs = self.update_outputs()

        self.current_inputs = None
        self.current_states = self.initial_states
        print '%d new data added' %iterations

    def objective_function(self, policy_parameters):
        terminate = False

        self.state_evolution, self.variance_evolution = self.update_evolution(reset=True)
        self.previous_states = self.current_states
        self.current_states = self.initial_states
        self.goal_counter = 0
        self.total_counter = 0
        self.current_reward = 0
        self.total_reward = 0

        self.action = self.action_from_policy(policy_parameters, self.current_states)
        self.current_inputs = self.generate_input()

        while terminate == False:
            self.previous_states = self.current_states
            self.current_states = self.current_states_from_gp(self.current_inputs)
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

    def current_states_from_gp(self, current_inputs):
        difference = self.model.predict(current_inputs)[0]
        if self.use_differences == True:
            self.current_states = difference + self.current_states
        else:
            self.current_states = difference
        return self.current_states

    def differences_for_gp(self):
        if self.use_differences == True:
            difference = self.current_states - self.previous_states
        else:
            difference = self.current_states
        return difference

    def objective_function_foo(self, policy_parameters):
        terminate = False
        self.counter = 1
        self.avg_variance = 0

        self.state_evolution, self.variance_evolution = self.update_evolution(reset=True)
        self.previous_states = self.current_states
        self.current_states = self.initial_states
        self.goal_counter = 0
        self.total_counter = 0
        self.current_reward = 0
        self.total_reward = 0
        self.action_evolution = np.zeros([1,1])

        self.action = self.action_from_policy(policy_parameters, self.current_states)
        self.action_evolution = self.action
        self.current_inputs = self.generate_input()

        if self.use_model == False:
            while terminate == False:
                self.previous_states = self.current_states
                self.current_states = self.sim(self.current_inputs[0,0],self.current_inputs[0,1:])
                self.current_states = np.reshape(self.current_states, ([1,1]))

                self.state_evolution, self.variance_evolution = self.update_evolution(reset=False)

                self.current_reward = self.reward_function(self.current_states, self.goal_states)
                self.total_reward += self.current_reward

                self.current_termination = self.termination_function(self)
                terminate = self.current_termination.check()

                self.action = self.action_from_policy(policy_parameters, self.current_states)
                self.action_evolution = np.append(self.action_evolution, self.action, axis = 0)
                self.current_inputs = self.generate_input()

            self.progress_counter += 1

            self.inputs_evolution = self.generate_input_evolution()
            self.predicition_evolution = self.current_states_from_gp(self.inputs_evolution)
            error = np.average(np.absolute(self.predicition_evolution - self.state_evolution))
            if error < self.error_limit and self.progress_counter > 5:
                self.use_model = True
            self.train_inputs = np.append(self.train_inputs, self.inputs_evolution, axis = 0)
            self.train_outputs = np.append(self.train_outputs, self.state_evolution, axis = 0)
            self.model = GPy.models.GPRegression(self.train_inputs, self.train_outputs, self.GP_kernel)

            if self.progress_print == True:
                print self.goal_counter, '\t', self.total_counter, '\t', self.progress_counter ,'\t', self.current_states ,'\t', error, '\t', self.action

            self.total_reward_evolution = np.append(self.total_reward_evolution,self.total_reward, axis = 0)
            return self.total_reward

        else:
            while terminate == False:
                self.previous_states = self.current_states
                self.current_states = self.current_states_from_gp(self.current_inputs)
                self.avg_variance = ((self.avg_variance * self.counter) + np.average(self.current_states_from_gp(self.current_inputs)) / (self.counter + 1))
                self.counter += 1

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
                print self.goal_counter, '\t', self.total_counter, '\t', self.progress_counter ,'\t', self.current_states ,'\t', self.total_reward

            self.total_reward_evolution = np.append(self.total_reward_evolution,self.total_reward, axis = 0)
            return self.total_reward


#The following class provides plotting tools based on the variables stored from the previous class. Thus it inherits methods and variables from the above, but is its own instance.
class plot:
    def __init__(self, rl_components):
        self.inheritence = rl_components
        self._plot_counter = None

#Given a set of training points, this function plots a graph of the real state evolution (blue) and the predicted state evolotion (green), along with the absolute error between them.
#The predicted curve has a shaded region that shows ±2σ, which is the standard deviation.
#One can set their own GP kernel, as a input parameter.
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
        pylab.savefig('./Plots/allplot_%d.eps' %self.plot_counter, format='eps')
        plt.show()


#Given a set of policy parameters, this function plots a graph of the real state evolution if this policy was followed for a set number of iterations, which is its only input.
        def run_simulation_true(self, iterations):
            self.inheritence.state_evolution, self.inheritence.variance_evolution = self.inheritence.update_evolution(reset = True)
            print self.inheritence.policy_parameters
            for i in range(iterations):
                self.inheritence.action = self.inheritence.action_from_policy(self.inheritence.policy_parameters, self.inheritence.current_states)
                self.inheritence.previous_states = self.inheritence.current_states

                self.inheritence.current_states = self.inheritence.sim(self.inheritence.action, self.inheritence.current_states)
                self.inheritence.current_states = np.reshape(self.inheritence.current_states, ([1,1]))

                self.inheritence.state_evolution, self.inheritence.variance_evolution = self.inheritence.update_evolution(reset = False)

            plt.close()
            plt.figure()

            plt.plot(self.inheritence.state_evolution)
            #plt.legend('%f%f' %self.inheritence.policy_parameters[0], self.inheritence.policy_parameters[1])
            plt.xlabel('Time / units', fontsize=16)
            plt.ylabel('State / unit', fontsize=16)

            pylab.savefig('./Plots/final_solution_%d.svg' %self.plot_counter, format='svg')
            pylab.savefig('./Plots/final_solution_%d.pdf' %self.plot_counter, format='pdf')

#Given a set of policy parameters, this function plots a graph of the predicted state evolution if this policy was followed for a set number of iterations, which is its only input.
        def run_simulation_gp(self, iterations):
            self.inheritence.state_evolution, self.inheritence.variance_evolution = self.inheritence.update_evolution(reset = True)
            print self.inheritence.policy_parameters

            for i in range(iterations):
                self.inheritence.action = self.inheritence.action_from_policy(self.inheritence.policy_parameters, self.inheritence.current_states)
                self.inheritence.current_inputs = self.inheritence.generate_input()
                self.inheritence.previous_states = self.inheritence.current_states
                print self.inheritence.action, self.inheritence.current_inputs
                self.inheritence.current_states = self.inheritence.current_states_from_gp(self.inheritence.current_inputs)
                self.inheritence.current_states = np.reshape(self.inheritence.current_states, ([1,1]))
                self.inheritence.state_evolution, self.inheritence.variance_evolution = self.inheritence.update_evolution(reset = False)


            plt.close()
            plt.figure()

            plt.plot(self.inheritence.state_evolution)
            #plt.legend('%f%f' %self.inheritence.policy_parameters[0], self.inheritence.policy_parameters[1])
            plt.xlabel('Time / units', fontsize=16)
            plt.ylabel('State / unit', fontsize=16)

            pylab.savefig('./Plots/final_solution_%d.svg' %self.plot_counter, format='svg')
            pylab.savefig('./Plots/final_solution_%d.pdf' %self.plot_counter, format='pdf')

#Given a set of policy parameters, this function plots the shape of the policy for the one dimensional policy.
        def plot_policy(self, policy_parameters, steps):
            test_states = np.linspace(0,120,steps)
            test_states = np.reshape(test_states, ([steps,1]))
            test_actions = np.zeros([steps,1])
            test_actions = np.reshape(test_actions, ([steps,1,1]))

            for i in range(steps):
                test_actions[i,1] = self.inheritence.action_from_policy(policy_parameters, test_states[i])

            plt.close()
            plt.figure()

            plt.plot(test_actions, test_states)
            plt.xlabel('State / units', fontsize=16)
            plt.ylabel('Action / units', fontsize=16)

            pylab.savefig('./Plots/policy_plots%d.svg' %self.plot_counter, format='svg')
            pylab.savefig('./Plots/policy_plots%d.pdf' %self.plot_counter, format='pdf')
)
