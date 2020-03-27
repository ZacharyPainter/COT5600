import numpy as np
from mypackage.min_degree import annihilate_min_deg_poly

#This example demonstrates the utility of the method
matrix = [[0,1],[1,0]]
print("Matrix:")
print(np.array(matrix))
a = annihilate_min_deg_poly(matrix)
print("Min degree polynomial:")
print(a)
print()

matrix = [[0,1,0],[1,0,0],[0,0,1]]
print("Matrix:")
print(np.array(matrix))
a = annihilate_min_deg_poly(matrix)
print("Min degree polynomial:")
print(a)
print()

matrix = [[1,0],[0,1]]
print("Matrix:")
print(np.array(matrix))
a = annihilate_min_deg_poly(matrix)
print("Min degree polynomial:")
print(a)
print()

matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
print("Matrix:")
print(np.array(matrix))
a = annihilate_min_deg_poly(matrix)
print("Min degree polynomial:")
print(a)
print()

matrix = [[2,1],[0,2]]
print("Matrix:")
print(np.array(matrix))
a = annihilate_min_deg_poly(matrix)
print("Min degree polynomial:")
print(a)
print()

matrix = [[2,0],[0,2]]
print("Matrix:")
print(np.array(matrix))
a = annihilate_min_deg_poly(matrix)
print("Min degree polynomial:")
print(a)
print()

matrix = [[1,0,1],[1,0,1],[0,1,0]]
print("Matrix:")
print(np.array(matrix))
a = annihilate_min_deg_poly(matrix)
print("Min degree polynomial:")
print(a)
print()

matrix = [[1,-1,-1],[1,-2,1],[0,1,-3]]
print("Matrix:")
print(np.array(matrix))
a = annihilate_min_deg_poly(matrix)
print("Min degree polynomial:")
print(a)
print()

