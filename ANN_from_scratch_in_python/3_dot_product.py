
# 3 neuron with 3 inputs

#inputs won't change
inputs = [1,2,3, 2.5]

#1 set of weights for each neuron
weights = [
    [0.2, 0.8, -0.5 , 1.0],
    [0.5 , -0.91, 0.26, -0.5]
    [-0.26, -0.27, 0.17, 0.87]
]

#1 bias for each neuron
biases = [ 2, 3, 0.5]


layer_outputs = []

#note: weights are lists of lists, and biases are just lists
#  so for weights we have n weights lists for n biases
for neuron_weights, neuron_bias in zip(weights, biases): 
    neuron_output = 0

    for n_input, weight in zip(inputs, neuron_weights):
        #multiply this input by associated weight
        #and add to the neuron's output variable
        neuron_output += n_input*weight
    #add bias
    neuron_output += neuron_bias
    #put neuron's result to the layer's output list
    layer_outputs.append(neuron_output)


# output = [
#     inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1 , 
#     inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2 ,
#     inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3
# ]

# print(output)