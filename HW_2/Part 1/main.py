import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

#Definitions
ket_zero = np.array([1,0])
bra_zero = np.conj(ket_zero).T #Transpose conjugate vector
ket_one = np.array([0,1])
bra_one = np.conj(ket_one).T #Transpose conjugate vector
identity_matrix = [[1,0],[0,1]]
c1 = (1/2**0.5)

#Gates
h_gate = c1 * np.array([[1,1],[1,-1]])

def simulate(u_gate):
    #Calculate the state of both qubits to start
    qubit_0 = ket_zero.copy()
    qubit_1 = ket_zero.copy()
    
    #Apply the Hadamard gate to the first qubit
    qubit_0 = np.matmul(h_gate,qubit_0)
    
    #Take the tensor product of both qubits to calculate the overall state
    state_s0 = np.kron(qubit_0, qubit_1)
    
    #Apply the controlled u gate to the state
    state_s0 = np.matmul(u_gate, state_s0)
    
    #Apply the second Hadamard gate to the first qubit only (using H otimes I)
    state_s0 = np.matmul(np.kron(h_gate, identity_matrix), state_s0)
    
    #Measure
    p0 = np.kron(np.outer(ket_zero, bra_zero), identity_matrix)
    probability_0 = linalg.norm(np.matmul(p0, state_s0))**2
    
    return probability_0

def main():
    pr_0 = []
    for i in range(1000):
        
        #Increment theta in steps of 1/1000
        theta = (i/1000)
        
        #Calculate angle using 2*pi*theta
        angle = 2*np.pi*theta
        
        #Calculate the controlled u gate using the formula from the slides and the rotation matrix
        u = [[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]]
        u_gate = np.kron(np.outer(ket_zero, bra_zero), identity_matrix) + np.kron(np.outer(ket_one, bra_one), u)
        
        #Save result
        pr_0.append(simulate(u_gate))
        
    plt.bar([(x/1000) for x in range(1000)], pr_0, 0.001, edgecolor='blue')
    plt.ylabel("Probability of 0 (Pr(0))")
    plt.xlabel("Value of theta")
    plt.savefig('dist.png')
    plt.show()
    
    
main()
    
    
        


