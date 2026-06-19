# Using @ for matrix multiplication
import numpy as np
a = np.array([[1, 2], 
              [3, 4]])

b = np.array([[2, 0], 
              [1, 2]])

result_matrix = a @ b                       # Me: Output: array([[4, 4], [10, 8]])
result_elementwise = a * b                  # Me: Output: array([[2 0], [3 8]])

print ("@ operation is for matrix multiplicaiton with the following result")
print (result_matrix)

print ("* operation is for elementwise multiplicaiton between two matrices with the following result")
print (result_matrix)
