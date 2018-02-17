#!/usr/bin/python
from random import seed
from random import random
import numpy as np
from math import exp
from math import pow
import matplotlib.pyplot as plt

#Configure a network with given number of input,hidden and output layer
def init_network(n_inputs,n_hidden,n_output):
    np.random.seed(5)
    network = list()
    hidden_layer = [{'weights': [np.random.uniform(-0.05,0.05) for i in range(n_inputs+1)],"hidden":list(),"encode":0} for i in range(n_hidden)]
    output_layer = [{'weights': [np.random.uniform(-0.05,0.05) for i in range(n_hidden+1)]} for i in range(n_output)]
    network.append(hidden_layer)
    network.append(output_layer)

    return network


#Return summation of weights and inputs
def summation(weights, input):
    sum = weights[-1] # bias
    for i in range(len(weights)-1):
        sum += weights[i]*input[i]
    return sum

#Forward propagate and store the outputs in the respective neuron for each layer
#Returns the values at the output layer
def forward_propagate(network, row, k=0):
    input = row
    i = 0
    for layers in network:
        new_inputs = list()
        for neuron in layers:
            sum_of_products = summation(neuron['weights'],input)
            neuron['output'] = 1.0/(1.0+ exp(-sum_of_products))
            new_inputs.append(neuron['output'])
            if i == 0 and k == 1:
                neuron['hidden'].append(neuron['output'])
            if i == 0:
                neuron['encode']= neuron['output']
        i +=1
        input = new_inputs
    return input

#backpropagate and store the delta values for each neuron for each layer
def backpropagate(network, expected, error_hist, n_input):
    for i in reversed(range(len(network))):
        error = list()
        layer = network[i]
        if i == 1: #output layer
            for j in range(len(layer)):
                neuron = layer[j]
                error.append(expected[j] - neuron['output'])

                #store sum of squared error in a list of lists for each output neuron
                if n_input == 0:
                    if len(error_hist[j]) == 0:
                        error_hist[j].append(pow(error[j],2))
                    else:
                        error_hist[j].append(pow(error[j],2))
                else:
                    error_hist[j][-1] = error_hist[j][-1] + pow(error[j],2)

        else: #hidden layer
            for j in range(len(layer)):
                sum = 0.0
                for neuron in network[i+1]:
                    sum += neuron['delta']*neuron['weights'][j] #downstream
                error.append(sum)
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = error[j] * neuron['output'] * (1-neuron['output'])


#update the weights for each neuron for each layer
def update_weights(network, row, rate, weight_hist):
    for i in range(len(network)):
        layer = network[i]

        if i == 0:
            input = row[:-1]
        else:
            input = [neuron['output'] for neuron in network[i-1]] # for output layer, hidden values are the inputs

        for k in range(len(layer)):
            neuron = layer[k]
            for j in range(len(input)):
                neuron['weights'][j] += rate * neuron['delta'] * input[j]
            neuron['weights'][-1] += rate * neuron['delta'] # for bias

            if i == 0 and k == 0: # for hidden layer and first neuron
                for m in range(len(weight_hist)):
                    weight_hist[m].append(neuron['weights'][m])


#Train the network with the given data, learning rate and number of iterations
def train_network(network, train, rate, iter, weight_hist, error_hist):
    for j in range(iter):
        for i in range(len(train)):
            input = train[i]
            expected = train[i]
            forward_propagate(network, input, i)
            backpropagate(network, expected, error_hist, i)
            update_weights(network, input, rate, weight_hist)


#Predict the output, given the input values
def predict(network,row):
    return forward_propagate(network,row)

#plot hidden unit encoding for input 01000000
def plot_hidden(list1,list2,list3):
    plt.plot(list1, label = "h1")
    plt.plot(list2, label = "h2")
    plt.plot(list3, label = "h2")
    plt.xlabel("Number of iteration")
    plt.legend()
    plt.title("Hidden unit encoding for input 01000000")
    plt.grid()
    plt.show()

#plot weights from input to one hidden neuron
def plot_weights(weight_hist):
    for i in range(len(weight_hist)-1):
        plt.plot(weight_hist[i],label = "w"+str(i+1))
    plt.plot(weight_hist[i], label="w_bias")

    plt.xlabel("Number of iteration")
    plt.ylabel("Weights")
    plt.legend()
    plt.title("Weights from inputs to one hidden unit")
    plt.grid()
    plt.show()


#plot sum of squared error for each output neuron
def plot_error(error_hist):
    for i in range(len(error_hist)):
        plt.plot(error_hist[i],label = "O"+str(i+1))

    plt.xlabel("Number of iteration")
    plt.ylabel("Sum of squared error")
    plt.legend()
    plt.title("Sum of squared errors for each output unit")
    plt.grid()
    plt.show()


#Driver program
def main():
    network = init_network(8, 3, 8)

    train = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]]

    weight_hist = list()
    error_hist = list()
    for i in range(len(network[0][0]['weights'])):
        weight_hist.append([network[0][0]['weights'][i]])

    for i in range(len(network[1])):
        error_hist.append(list())

    train_network(network, train, 0.3, 5000, weight_hist, error_hist)


    for i in range(len(network)):
        for j in range(len(network[i])):
            print network[i][j]['weights']
        print "\n\n"

    plot_hidden(network[0][0]['hidden'],network[0][1]['hidden'],network[0][2]['hidden'])
    plot_weights(weight_hist)
    plot_error(error_hist)

    for k in range(len(train)):
        print predict(network, train[k])
        print str(network[0][0]['encode'])+ ", "+ str(network[0][1]['encode']) + ", " + str(network[0][2]['encode'])


#calling main function
if __name__ == "__main__":
    main()