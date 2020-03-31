import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

#Definitions
ket_zero = np.array([1,0])
bra_zero = np.conj(ket_zero).T #Transpose conjugate vector
ket_one = np.array([0,1])
identity_matrix = [[1,0],[0,1]]
c1 = (1/2**0.5)

#Gates
h_gate = c1 * np.array([[1,1],[1,-1]])
swap_gate = np.array(
        [[1,0,0,0,0,0,0,0],
         [0,1,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0],
         [0,0,0,1,0,0,0,0],
         [0,0,0,0,1,0,0,0],
         [0,0,0,0,0,0,1,0],
         [0,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,0,1]])

def simulate(theta):
    #Calculate the state of both qubits to start
    qubit_0 = ket_zero.copy()
    qubit_1 = ket_zero.copy()
    qubit_2 = np.cos(2*np.pi*theta)*ket_zero + np.sin(2*np.pi*theta)*ket_one
    
    #Apply the Hadamard gate to the first qubit
    qubit_0 = np.matmul(h_gate,qubit_0)
    
    #Take the tensor product of both qubits to calculate the overall state
    state_s0 = np.kron(np.kron(qubit_0, qubit_1), qubit_2)
    
    #Apply the controlled swap gate to the state
    state_s0 = np.matmul(swap_gate, state_s0)
    
    #Apply the second Hadamard gate to the first qubit only (using H otimes I otimes I)
    h_q0 = np.kron(np.kron(h_gate, identity_matrix), identity_matrix)
    state_s0 = np.matmul(h_q0, state_s0)

    
    #Measure
    p0 = np.kron(np.outer(ket_zero, bra_zero), np.eye(4))
    probability_0 = linalg.norm(np.matmul(p0, state_s0))**2
    
    return probability_0

def main():
    pr_0 = []
    for i in range(1000): 
        #Save result
        pr_0.append(simulate(i/1000))
        
    plt.bar([(x/1000) for x in range(1000)], pr_0, 0.001, edgecolor='blue')
    plt.ylabel("Probability of 0 (Pr(0))")
    plt.xlabel("Value of theta")
    plt.savefig('dist.png')
    plt.show()
    
main()
    
        


