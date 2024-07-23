import numpy as np
#----------------------------------------------
class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
    
    def forward(self, input):
        self.output = input * self.weight + self.bias

    def full_forward(self, input):
        self.forward(input)
        self.__relu(self.output)
    
    def __relu(self,input:int):
        self.output = max(0, input)
#----------------------------------------------
import matplotlib.pyplot as plt



n1 = Neuron(1, 0.5)
print(f'N1 - weight: {n1.weight}, bias: {n1.bias}')

x_values = []
y_values = []
for i in np.arange(-1, 2, 0.1):
    n1.full_forward(i)
    print(f'i: {round(i,3)} - output: {round(n1.output,3)}')
    y_values.append(n1.output)
    x_values.append(i)


plt.plot(x_values, y_values, 'o-')
plt.xlabel('n1.output')
plt.ylabel('i')
plt.title('Line Chart')
plt.show()

print('----------------------------------------------')


n2 = Neuron(-1, 0.5)
print(f'N2 - weight: {n2.weight}, bias: {n2.bias}')

x_values = []
y_values = []
for i in np.arange(-1, 2, 0.1):
    n2.full_forward(i)
    print(f'i: {round(i,3)} - output: {round(n2.output,3)}')
    y_values.append(n2.output)
    x_values.append(i)

plt.plot(x_values, y_values, 'o-')
plt.xlabel('n2.output')
plt.ylabel('i')
plt.title('Line Chart')
plt.show()
print('----------------------------------------------')


print(f'N1 - weight: {n1.weight}, bias: {n1.bias}')
print(f'N2 - weight: {n2.weight}, bias: {n2.bias}')
x_values = []
y_values = []
for i in np.arange(-1, 2, 0.1):
    n1.full_forward(i)
    # print(f'i: {round(i,3)} - output: {round(n1.output,3)}')
    n2.full_forward(n1.output)

    y_values.append(n2.output)
    x_values.append(i)


plt.plot(x_values, y_values, 'o-')
plt.xlabel('n2.output')
plt.ylabel('i')
plt.title('n1 -> n2')
plt.show()