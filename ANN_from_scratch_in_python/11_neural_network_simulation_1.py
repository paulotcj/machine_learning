import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------
class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
    
    def forward(self, inputs):
        self.output = np.dot( inputs, self.weight ) + self.bias

    def full_forward(self, input):
        self.forward(input)
        self.__relu(self.output)
    
    def __relu(self,inputs):
        self.output = np.maximum(0, inputs)
#----------------------------------------------


#----------------------------------------------
# Generate x values from 0 to 2*pi with a step of 0.1
x1 = np.arange(0, 2*np.pi, 0.1)
# Calculate y values using sine function
y1 = np.sin(x1)
#----------------------------------------------


#2*pi = 6.283185307179586
x2 = np.arange(0, 1, 0.2)
# y2 = np.sin(x2)
n1 = Neuron(0.9, 0.0)
n2 = Neuron(0.5, 0.0)
n3 = Neuron(2, 0.0)

y2 = []
for i in x2:
    n1.full_forward(i)
    n2.full_forward(n1.output)
    n3.full_forward(n2.output)
    y2.append(n3.output)


 

# Plot the sine wave
plt.plot(x1, y1)
plt.plot(x2, y2, 'ro-')  # Add red dots for x2 and y2
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Wave')
plt.show()


