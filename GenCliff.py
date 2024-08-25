import numpy as np
from tqdm import tqdm 


H = np.sqrt(1/2)*np.array([[1, 1],
                             [1,-1]])
P = np.array([[1,0],
              [0,1j]])
I = np.eye(2)


def tree_gen(seed,depth,H_P_bool):
    tree = {0:[seed]}
    H_P = H_P_bool #True if last is P, False if last is H
    for d in tqdm(range(1,depth)):
        if H_P:
            new_seqs =  []
            for seq in tree[d-1]:
                new_seqs.append(seq + 'H')
            tree[d] = new_seqs
        else:
            new_seqs =  []
            for seq in tree[d-1]:
                new_seqs.append(seq + 'P')
                new_seqs.append(seq + 'PP')
                new_seqs.append(seq + 'PPP')
            tree[d] = new_seqs
        H_P = not H_P
    return tree

def compute(C):
    Cl = I
    for o in C:
        if o=='H':
            Cl = Cl@H
        elif o=='P':
            Cl = Cl@P
    return Cl
    
def get_order(X,n):
    I = np.eye(2)
    for i in range(1,n+1):
        X_i = np.linalg.matrix_power(X, i)
        if np.allclose(X_i, I, atol=1e-7):
            return i
    return None

def unique_up_to_scalar(matrix):
    normalized_matrix = np.zeros_like(matrix, dtype=np.complex128)
    for i, row in enumerate(matrix):
        first_non_zero = np.nonzero(row)[0]
        if len(first_non_zero) > 0:
            first_non_zero_index = first_non_zero[0]
            normalized_matrix[i] = row / row[first_non_zero_index]
    unique= np.unique(normalized_matrix,axis=0,return_index=True)
    return unique[1]


if __name__ == '__main__':
	tree = tree_gen('P',20,False)
	total = []
	for i in range(20):
	    total += tree[i]

	mat_reps = np.array([compute(C).flatten() for C in tqdm(total)])
	ixs = unique_up_to_scalar(mat_reps)
	print(len(ixs))

	with open('cliffords.txt','w') as f:
		for ix in ixs:
			f.writelines(total[ix]+'\n')

