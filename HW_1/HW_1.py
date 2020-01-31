import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import null_space

#Flattens a matrix row-first
def vec(matrix):
    return list(matrix.flatten())

#Produce the matrix A which has dimensions (n^2)x(n+1),
#and whose columns are the matrix powers of M
def get_flat_matrix_powers(matrix):
    m = []
    for i in range(len(matrix) + 1):
        #Get the flattened version of the i'th matrix power of the matrix
        flat_matrix = vec(matrix_power(matrix,i))
        m.append(flat_matrix) 
    
    #m is constructed such that the rows of m are the matrix powers of our matrix,
    #we tranpose m to make it so that the columns are the matrix
    #powers of our matrix instead (It was just easier to build the matrix this way in python)
    return np.array(m).T

def annihilate_min_deg_poly(matrix):
    #Calculate all matrix powers
    m = get_flat_matrix_powers(matrix)
    
    #Get the null space
    ns = null_space(m)
    
    #Answer returned by null_space() is normalized, this gives
    #us the vector which is not normalized
    ns = ns/ns.max() 
    
    #Python retuns this as a multi-dimensional vector, but poly1d requires
    #a 1d vector, so we take the first column of the basis vector
    #(There should only ever be a sigle column)
    return np.poly1d(ns[:,0])
        

example_matrix = [[0,1],[1,0]]
a = annihilate_min_deg_poly(example_matrix)
print(a)


        