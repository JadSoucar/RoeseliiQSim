import numpy as np
#from CircuitSimulation import NoiseChannel
import copy


class StabState:
    '''
    all credit to Dr. Arannson himeself
    and of course the omnipotent Brayvi
    https://www.scottaaronson.com/qclec/28.pdf
    '''
    
    def __init__(self,X_matrix,Z_matrix):
        self.initial_X_matrix = X_matrix
        self.initial_Z_matrix = Z_matrix
        self.X_matrix = X_matrix
        self.Z_matrix = Z_matrix
        self.signs = np.array([1],dtype=complex)
        self.coeffs = np.array([1],dtype=complex)
        self.phase = np.array([0],dtype=complex)
        self.len = 1

    def apply_H(self, qubit=0):
        if self.Z_matrix[qubit] == 1 and self.X_matrix[qubit]==1 and self.signs[qubit] == 1:
            self.signs[qubit] *= -1
            self.phase[qubit] += 1/4

        elif self.Z_matrix[qubit] == 1 and self.X_matrix[qubit]==1 and self.signs[qubit] == -1:
            self.signs[qubit] *= -1
            self.phase[qubit] += -1/4

        self.X_matrix[qubit], self.Z_matrix[qubit] = self.Z_matrix[qubit], self.X_matrix[qubit]
        
    def apply_P(self, qubit=0):
        if self.Z_matrix[qubit] == 1 and self.X_matrix[qubit]==1:
            self.signs[qubit] *= -1

        if self.Z_matrix[qubit] == 1 and self.X_matrix[qubit]==0 and self.signs[qubit] == -1:
            self.phase[qubit] += 1/2

        self.Z_matrix[qubit] ^= self.X_matrix[qubit]
        
    def clifford_circuit(self,coeff,components,qubits):
        '''
        ex. components = PH <-- P first H second
            This is how clifford strings are stored
        '''
        for component in components[::-1]:
            #print(component)
            if component == 'H':
                for qubit in qubits:
                    self.apply_H(qubit)
            elif component == 'P':
                for qubit in qubits:
                    self.apply_P(qubit)

            #self.display_tableu()

        for qubit in qubits:
            self.coeffs[qubit] *= coeff

        #self.display_tableu()
        return None
            
    def get_ket(self,):
        final_ket = np.zeros((2,),dtype=complex)
        for qubit in range(len(self.coeffs)):
            if self.X_matrix[qubit]==1 and self.Z_matrix[qubit]==0 and self.signs[qubit]==1:
                ket = np.exp(np.pi*1j*self.phase[qubit])*self.coeffs[qubit]*np.sqrt(np.array([1/2,1/2],dtype=complex))
                final_ket += ket
            elif self.X_matrix[qubit]==0 and self.Z_matrix[qubit]==1 and self.signs[qubit]==1:
                ket = np.exp(np.pi*1j*self.phase[qubit])*self.coeffs[qubit]*np.array([1,0], dtype=complex)
                final_ket += ket
            elif self.X_matrix[qubit]==1 and self.Z_matrix[qubit]==1 and self.signs[qubit]==1:
                ket = np.exp(np.pi*1j*self.phase[qubit])*self.coeffs[qubit]*np.sqrt(1/2)*np.array([1+0j,0+1j])
                final_ket += ket
            elif self.X_matrix[qubit]==1 and self.Z_matrix[qubit]==0 and self.signs[qubit]==-1:
                ket = np.exp(np.pi*1j*self.phase[qubit])*self.coeffs[qubit]*np.sqrt(1/2)*np.array([1,-1],dtype=complex)
                final_ket += ket
            elif self.X_matrix[qubit]==0 and  self.Z_matrix[qubit]==1 and self.signs[qubit]==-1:
                ket = np.exp(np.pi*1j*self.phase[qubit])*self.coeffs[qubit]*np.array([0,1], dtype=complex)
                final_ket += ket
            elif self.X_matrix[qubit]==1 and self.Z_matrix[qubit]==1 and self.signs[qubit]==-1:
                ket = np.exp(np.pi*1j*self.phase[qubit])*self.coeffs[qubit]*np.sqrt(1/2)*np.array([1+0j,0-1j])
                final_ket += ket

        return final_ket
            
    def increase_tableu_size(self,n):
        self.X_matrix = np.tile(self.X_matrix, (n,))
        self.Z_matrix = np.tile(self.Z_matrix, (n,))
        self.coeffs = np.tile(self.coeffs,(n,))
        self.signs = np.tile(self.signs,(n,))
        self.phase = np.tile(self.phase,(n,))
        self.len *= n

    def display_tableu(self,):
        print('q_n__|s_n__|__X__|__Z__|c_n__')
        for qubit in range(len(self.coeffs)):
            X = self.X_matrix[qubit]
            Z = self.Z_matrix[qubit]
            s = int(self.signs[qubit].real)
            coeff = np.round(self.coeffs[qubit],4)
            if s>0:
                print(f'{qubit}----|({s})--|--{X}--|--{Z}--|--{coeff}')
            else:
                print(f'{qubit}----|({s})-|--{X}--|--{Z}--|--{coeff}')
        return None

    def deepcopy(self,):
        return copy.copy(self)

    def clear(self,):
        self.X_matrix = self.initial_X_matrix
        self.Z_matrix = self.initial_Z_matrix
        self.signs = np.array([1],dtype=complex)
        self.coeffs = np.array([1],dtype=complex)
        self.phase = np.array([0],dtype=complex)
        self.len = 1
