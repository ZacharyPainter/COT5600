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
        ns = null_space(np.array(m).T)
        
        #If a basis for the null space exists, we have found the lowest degree
        #solution to AB=0
        if ns.size > 0:
            #print(np.array(m).T)
            break
    
    #Answer returned by null_space() is normalized, this gives
    #us the vector which is not normalized
    ns = ns/ns.max()
    
    #Remove all extremely small numbers (returned in error by numpy)
    ns[np.abs(ns)<0.0000000001]=0
    #print("basis vector:")
    #print(ns)
    return np.poly1d(np.flip(ns[:,0]))
        


        