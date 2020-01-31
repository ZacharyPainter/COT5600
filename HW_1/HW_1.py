import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import null_space

#Flattens a matrix row-first
def vec(matrix):
    return list(matrix.flatten())

#Produce the matrix A which has dimensions (n^2)x(n+1),
#and whose columns are the matrix powers of M
def annihilate_min_deg_poly(matrix):
    m = []
    
    #We iterate up to the length of the matrix, each loop adding the next
    #power of the given matrix to the matrix "m." On each loop, we take the
    #null_space of "m," and if it exists we break from the loop, using the
    #resulting basis from te null_space method to form our minimal polynomial
    for i in range(len(matrix)+1):
        
        #Get the flattened version of the i'th matrix power of the matrix
        #and append it to the running matrix "m"
        flat_matrix = vec(matrix_power(matrix,i))
        m.append(flat_matrix) 
        
        #m is constructed such that the rows of m are the matrix powers of our given matrix: M,
        #we tranpose m to make it so that the columns are the matrix
        #powers of M instead (It was just easier to build the matrix this way in python).
        #We then calculate the null space of this transposed matrix
        #ns = null_space(np.array(m).T)
        
        #If a basis for the null space exists, we have found the lowest degree
        #solution to AB=0
        #if ns.size > 0:
        #    break
    
    #Answer returned by null_space() is normalized, this gives
    #us the vector which is not normalized
    print(np.array(m).T)
    ns = null_space(np.array(m).T)
    ns = ns/ns.max()
    print("basis vector:")
    print(ns)
    return np.poly1d(np.flip(ns[:,0]))
        

#This example demonstrates the utility of the method
example_matrix = [[1,0,1],[1,0,1],[0,1,0]]
example_matrix = [[0,1],[1,0]]
a = annihilate_min_deg_poly(example_matrix)
print("basis vector as polynomial:")
print(a)


        