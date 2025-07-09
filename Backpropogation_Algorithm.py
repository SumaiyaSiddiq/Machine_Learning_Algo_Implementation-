# -*- coding: utf-8 -*-
"""


@author: sumaiya
"""

import numpy as np

input_neurons = 2
hidden_layer_neurons = 2
output_neurons = 2

input_ = np.random.randint(1, 100, input_neurons)
output = np.array([1.0, 0.0])

hidden_biass = np.random.rand(1, hidden_layer_neurons)
output_biass = np.random.rand(1, output_neurons)

hidden_weight = np.random.rand(input_neurons, hidden_layer_neurons)
output_weight = np.random.rand(hidden_layer_neurons, output_neurons)


def sigmoid(layer):
    return 1 / (1 + np.exp(-layer))


def gradient(layer):
    return layer * (1 - layer)


for i in range(2000):

    hidden_layer = np.dot(input_, hidden_weight)
    hidden_layer = sigmoid(hidden_layer + hidden_biass)

    output_layer = np.dot(hidden_layer, output_weight)
    output_layer = sigmoid(output_layer + output_biass)

    error = (output - output_layer)
    gradient_outputLayer = gradient(output_layer)

    error_terms_output = gradient_outputLayer * error
    error_terms_hidden = gradient(hidden_layer) * np.dot(error_terms_output, output_weight.T)


    gradient_hidden_weights = np.dot(input_.reshape(input_neurons, 1),
                                     error_terms_hidden.reshape(1, hidden_layer_neurons))
    gradient_output_weights = np.dot(hidden_layer.reshape(hidden_layer_neurons, 1),
                                    error_terms_output.reshape(1, output_neurons))


    hidden_weight = hidden_weight + 0.05 * gradient_hidden_weights
    output_weight = output_weight + 0.05 * gradient_output_weights

    print('***********************************')
    print('Iteration: ', i, ':::', error)
    print('####- output - #####', output_layer)


