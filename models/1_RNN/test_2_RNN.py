import math
import numpy as np


print('----------------------------------------')
print('dot product concepts')
matrix1 = [2 , 4]

matrix2 = [[3, 5],[7, 11]]

result = np.dot(matrix1, matrix2)
print('excepected: [2*3 + 4*7, 2*5 + 4*11] -> [6 + 28, 10 + 44] -> [34, 54]')
print(f'result: {result}')



#--------------------------------------------------------------------
#setting up variables

x = [1,2,3]

Wx = [-1.2760801,  -0.39949882]
Wh = [[-0.18188739, -0.98331934], [-0.98331934,  0.18188745]]

h0 = [0.0, 0.0]

Bh = [0.0, 0.0]



print('----------------------------------------')
print('h1 results')

result1 = np.dot(x[0], Wx)
print(result1)

result2 = np.dot(h0, Wh) 
print(result2)

result3 = Bh
print(result3)
h1 = result1 + result2 + result3
print(f'h1:{h1}')
exit()
print('----------------------------------------')
print('h2 results')

result4 = np.dot(x[1], Wx)
print(result4)

result5 = np.dot(h1, Wh)
print(result5)

result6 = Bh
print(result6)

h2 = result4 + result5 + result6
print(f'h2:{h2}')

print('----------------------------------------')
print('h3 results')
result7 = np.dot(x[2], Wx)
print(result7)

result8 = np.dot(h2, Wh)
print(result8)

result9 = Bh
print(result9)

h3 = result7 + result8 + result9
print(f'h3:{h3}')




