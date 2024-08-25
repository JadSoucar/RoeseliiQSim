from MILP import MILP
from StabalizerSimulation import StabState
import numpy as np
import random
from tqdm import tqdm


class NoiseChannel:
	def __init__(self,KOs,probs=None,cliffords_path='cliffords.txt',):
		'''
		KOs: A list of (4x1) Krauss Operators in the Noise Channel 
		'''
		self.KOs = KOs
		self.solver = MILP(cliffords_path)
		self.probs_approx = probs
		self.probs_exact = probs

	def solve_approx(self,lambda_weight,scaling_factor,verbose=True):
		'''
		Get the Approximated Noise Channel using MILP class
		'''
		self.KOs_approx = []
		self.coeffs = []
		self.str_reps = []
		for K in self.KOs:
			self.solver.solve(K,lambda_weight,scaling_factor)
			if verbose:
				self.solver.view_solution_summary()
			self.coeffs.append(self.solver.get_coeffs_of_approx())
			self.str_reps.append(self.solver.get_str_cliffords_of_approx())
			self.KOs_approx.append(self.solver.get_approx().reshape(2,2))

		return None 

	def get_prob_distro_approx(self,):
		'''
		Grab Lower Bounds of Distro
		'''
		if self.probs_approx != None:
			self.probs_approx = []
			for K in self.KOs_approx:
				U, S, Vt = np.linalg.svd(K)
				self.probs_approx.append(np.linalg.norm(S[-1])**2)
		return self.probs_approx

	def get_prob_distro_exact(self,):
		'''
		Grab Lower Bounds of Distro
		'''
		if self.probs_exact != None:
			self.probs_exact = []
			for K in self.get_exact():
				U, S, Vt = np.linalg.svd(K)
				self.probs_exact.append(S[-1]**2)
		return self.probs_exact


	def get_approx(self,):
		'''
		returns 2x2 approx KOs
		'''
		return self.KOs_approx

	def get_exact(self,):
		'''
		returns 2x2 exact KOs
		'''
		return [K.reshape(2,2) for K in self.KOs]

	def get_stabstate_inputs(self,):
		return list(zip(self.coeffs,self.str_reps))


class Simulation:
	def __init__(self,NCs,lambda_weight,scaling_factor,verbose=True):
		'''
		NCs = List of Noise Channels to be applied in order

		#MILP Level Parameters
		lambda weight: same lambda weight defined in MILP
		scaling_factor: same scaling factor defined in MILP
		verbose: verbose for MILP approximations
		'''
		self.NCs = NCs 
		self.lambda_weight = lambda_weight
		self.scaling_factor = scaling_factor
		self.verbose = verbose


	@staticmethod
	def average_matrices(matrices):
		# Sum all the matrices element-wise
		sum_matrix = np.zeros((2,2),dtype=complex)
		for matrix in matrices:
			sum_matrix += matrix
		
		# Divide by the number of matrices to get the average
		average_matrix = sum_matrix / len(matrices)
		
		return average_matrix

	@staticmethod
	def split_array_into_k_parts(array, k):
	    n = len(array) // k
	    subarrays = [array[i * n:(i + 1) * n] for i in range(k)]
	    return subarrays

	############## Density Matrix Simulation ###########################

	def run_density_exact(self,initial_rho):
		self.final_rho_exact = initial_rho
		for NC in self.NCs:
			NC_exact = NC.get_exact()
			NC_sum = np.zeros((2,2),dtype=complex)
			for K in NC_exact:
				NC_sum += K@self.final_rho_exact@np.conjugate(K.T)
			self.final_rho_exact = NC_sum

		self.final_rho_exact = self.final_rho_exact #/np.trace(self.final_rho_exact)
		return self.final_rho_exact

	def run_density_approx(self,initial_rho):
		self.final_rho_approx = initial_rho
		for NC in self.NCs:
			NC.solve_approx(self.lambda_weight,self.scaling_factor,self.verbose)
			NC_approx = NC.get_approx()
			NC_sum = np.zeros((2,2),dtype=complex)
			for K in NC_approx:
				NC_sum += K@self.final_rho_approx@np.conjugate(K.T)
			self.final_rho_approx = NC_sum

		self.final_rho_approx = self.final_rho_approx #/np.trace(self.final_rho_approx)
		return self.final_rho_approx


	##### Experimentation with Uniform ##########
	def uniform_exact(self,initial_ket,nsim):
		final_kets = []
		for _ in tqdm(range(nsim)):
			scaler = 1
			final_ket = initial_ket
			for NC in self.NCs:
				NC_exact = NC.get_exact()
				scaler *= len(NC_exact)
				selected_K = random.choice(NC_exact)
				final_ket = selected_K@final_ket
			final_kets.append(final_ket)
		rho_finals = [np.outer(final_ket,np.conjugate(final_ket.T)) for final_ket in final_kets]
		self.final_uniform_rho_exact = scaler*self.average_matrices(rho_finals)
		return self.final_uniform_rho_exact

	def qt_exact(self,initial_ket,nsim):
		final_kets = []
		all_p_is = []
		for _ in tqdm(range(nsim)):
			final_ket = initial_ket
			one_sim_p_is = []
			for NC in self.NCs:
				NC_exact = NC.get_exact()
				final_ket = final_ket/np.linalg.norm(final_ket)

				#Get probs
				p_is = []
				for K in NC_exact:
					p_i = np.conjugate(final_ket.T)@np.conjugate(K.T)@K@final_ket
					p_is.append(p_i.real)

				p_is = np.array(p_is)
				selected_K_ix = np.random.choice(range(len(NC_exact)),p=p_is/np.sum(p_is))
				selected_K = NC_exact[selected_K_ix]
				final_ket = selected_K@final_ket/np.sqrt(p_is[selected_K_ix])
				one_sim_p_is.append(p_is)

			final_kets.append(final_ket)
			all_p_is.append(one_sim_p_is)


		rho_finals = [np.outer(final_ket,np.conjugate(final_ket.T)) for final_ket in final_kets]
		self.final_qt_rho_exact = self.average_matrices(rho_finals)
		return self.final_qt_rho_exact, all_p_is


	##### KET BASED SIMULATION (Quantum Trajectories) ##########

	def run_ket_exact(self,initial_ket,nsim):
		'''
		https://arxiv.org/pdf/2111.02396
		'''

		final_kets = []
		for _ in tqdm(range(nsim)):
			selected_K = []
			flag = False
			final_ket = initial_ket
			#Easy Krauss selection
			for NC in self.NCs:
				NC_exact = NC.get_exact()
				probs = NC.get_prob_distro_exact() #probs = np.ones(len(NC_exact))/len(NC_exact) 
				r = random.random()

				for K,prob in zip(NC_exact,probs):
					if r < prob:
						selected_K.append(K)
						flag = True
						break
					else:
						r = r - prob
				#Fuse Krauss
				if flag == True:
					flag = False
					continue

				fused = np.eye(2,dtype=complex)
				for K in selected_K:
					fused = K@fused
				final_ket = fused@final_ket / np.linalg.norm(fused@final_ket)
				selected_K = []

				#Hard K selection
				for K,prob in zip(NC_exact,probs):
					p_i = np.conjugate(final_ket.T)@np.conjugate(K.T)@K@final_ket
					if r < p_i-prob:
						final_ket = K@final_ket * (1/np.sqrt(p_i))
						break
					else:
						r = r - (p_i-prob)

			#Account for Missing
			if selected_K != []:
				fused = np.eye(2,dtype=complex)
				for K in selected_K:
					fused = K@fused
				final_ket = fused@final_ket / np.linalg.norm(fused@final_ket)
				selected_K = []

			final_kets.append(final_ket) #/np.linalg.norm(final_ket)
		rho_finals = [np.outer(final_ket,np.conjugate(final_ket.T)) for final_ket in final_kets]

		self.final_ket_exact = self.average_matrices(rho_finals)
		return self.final_ket_exact


	def run_ket_approx(self,initial_ket,nsim):
		'''
		https://arxiv.org/pdf/2111.02396
		'''

		#Solve Decomp
		for NC in self.NCs:
			for K in NC.get_exact():
				NC.solve_approx(self.lambda_weight,self.scaling_factor,self.verbose)

		final_kets = []
		for _ in tqdm(range(nsim)):
			selected_K = []
			flag = False
			final_ket = initial_ket
			#Easy Krauss selection
			for NC in self.NCs:
				NC_exact = NC.get_approx()
				probs = NC.get_prob_distro_approx()
				r = random.random()
				for K,prob in zip(NC_exact,probs):
					if r < prob:
						selected_K.append(K)
						flag = True
						break
					else:
						r = r - prob
				#Fuse Krauss
				if flag == True:
					flag = False
					continue

				fused = np.eye(2,dtype=complex)
				for K in selected_K:
					fused = K@fused
				final_ket = fused@final_ket / np.linalg.norm(fused@final_ket)
				selected_K = []

				#Hard K selection
				for K,prob in zip(NC_exact,probs):
					p_i = np.conjugate(final_ket.T)@np.conjugate(K.T)@K@final_ket
					if r < p_i-prob:
						final_ket = K@final_ket * (1/np.sqrt(p_i)) 
						break
					else:
						r = r - (p_i-prob)

			#Account for Missing
			if selected_K != []:
				fused = np.eye(2,dtype=complex)
				for K in selected_K:
					fused = K@fused
				final_ket = fused@final_ket / np.linalg.norm(fused@final_ket)
				selected_K = []
			final_kets.append(final_ket) #/np.linalg.norm(final_ket)

		rho_finals = [np.outer(final_ket,np.conjugate(final_ket.T)) for final_ket in final_kets]
		self.final_ket_approx = self.average_matrices(rho_finals)
		return self.final_ket_approx


	############################ Stabalizer Simulation ############################
	def get_stab_approx(self,initial_stab,nsim):
		#Solve Decomp
		for NC in self.NCs:
			for K in NC.get_exact():
				NC.solve_approx(self.lambda_weight,self.scaling_factor,self.verbose)

		final_kets = []
		for _ in tqdm(range(nsim)):
			s = initial_stab.deepcopy()
			ket = s.get_ket()
			for NC in self.NCs:
				#load in data
				Ks = NC.get_approx()
				Ks_stab_input = NC.get_stabstate_inputs()

				#get probs
				probs_w_imag = [np.inner(np.conjugate((K@ket).T),K@ket) for K in Ks]
				probs = np.array([prob.real for prob in probs_w_imag])
				selected_K_stab_input_ix = np.random.choice(range(len(Ks_stab_input)), p=probs/sum(probs))
				selected_K_stab_input = Ks_stab_input[selected_K_stab_input_ix]

				#Store Prob
				selected_K_prob = probs[selected_K_stab_input_ix]

				#claculate using tableu
				num_cliffords = len(selected_K_stab_input[0])
				s.increase_tableu_size(num_cliffords)
				subarrays = self.split_array_into_k_parts(np.arange(s.len),num_cliffords)
				for cx in range(len(selected_K_stab_input[0])):
					clifford = selected_K_stab_input[1][cx]
					coeff = selected_K_stab_input[0][cx]
					s.clifford_circuit(coeff/np.sqrt(selected_K_prob),clifford,qubits=subarrays[cx])
				#extract rho and ket
				ket = s.get_ket()

			final_kets.append(ket)
			s.clear()


		ket_to_rho_finals = [np.outer(final_ket,np.conjugate(final_ket.T)) for final_ket in final_kets]
		self.final_ket_stab_approx = self.average_matrices(ket_to_rho_finals)
		return self.final_ket_stab_approx


	def get_stab_approx_qt(self,initial_stab,nsim):
		#Solve Decomp
		for NC in self.NCs:
			for K in NC.get_exact():
				NC.solve_approx(self.lambda_weight,self.scaling_factor,self.verbose)

		final_kets = []
		for _ in tqdm(range(nsim)):

			#Easy Kruass selction
			s = initial_stab.deepcopy()
			ket = s.get_ket()
			for NC in self.NCs:
				#load in data
				Ks = NC.get_approx()
				Ks_stab_input = NC.get_stabstate_inputs()
				probs = NC.get_prob_distro_approx()
				r = random.random()
				for K_approx,K,prob in zip(Ks,Ks_stab_input,probs):
					if r < prob:
						num_cliffords = len(K[0])
						s.increase_tableu_size(num_cliffords)
						subarrays = self.split_array_into_k_parts(np.arange(s.len),num_cliffords)
						p_i = np.inner(np.conjugate((K_approx@ket).T),K_approx@ket)
						for cx in range(len(K[0])):
							clifford = K[1][cx]
							coeff = K[0][cx]
							s.clifford_circuit(coeff/np.sqrt(p_i),clifford,qubits=subarrays[cx])
						#extract rho and ket
						ket = s.get_ket()
						flag = True
						break
					else:
						r = r - prob

				#Hard Kruass Selection
				for K_approx,K,prob in zip(Ks,Ks_stab_input,probs):
					p_i = np.inner(np.conjugate((K_approx@ket).T),K_approx@ket)
					if r < p_i-prob:
						num_cliffords = len(K[0])
						s.increase_tableu_size(num_cliffords)
						subarrays = self.split_array_into_k_parts(np.arange(s.len),num_cliffords)
						for cx in range(len(K[0])):
							clifford = K[1][cx]
							coeff = K[0][cx]
							s.clifford_circuit(coeff/np.sqrt(p_i),clifford,qubits=subarrays[cx])
						#extract rho and ket
						ket = s.get_ket()
						break
					else:
						r = r - (p_i-prob)

			final_kets.append(ket)
			s.clear()


		ket_to_rho_finals = [np.outer(final_ket,np.conjugate(final_ket.T)) for final_ket in final_kets]
		self.final_ket_stab_qt = self.average_matrices(ket_to_rho_finals)
		return self.final_ket_stab_qt

	################################# View #########################################


	def get_summary(self,result_type):
		'''
		type: density, ket,stab or all
		'''

		print('*'*40)
		if result_type=='density' or result_type=='all':
			L2_error = np.linalg.norm(self.final_rho_approx.flatten()-self.final_rho_exact.flatten())
			print('Approx Rho prime: \n',self.final_rho_approx)
			print('Exact Rho prime: \n',self.final_rho_exact)
			print('L2_error',L2_error)
			print('*'*40)

		if result_type=='ket' or result_type=='all':
			L2_error = np.linalg.norm(self.final_ket_approx.flatten()-self.final_ket_exact.flatten())
			print('Exact Ket -> Rho prime: \n', self.final_ket_exact)
			print('Approx Ket -> Rho prime: \n', self.final_ket_approx)
			print('L2_error',L2_error)
			print('*'*40)

		if result_type=='stab' or result_type=='all':
			print('Stab Ket -> Rho Prime: \n', np.round(self.final_ket_stab_approx,5))
			print('Stab QT Ket -> Rho Prime: \n', np.round(self.final_ket_stab_qt,5))
		print('*'*40)



if __name__ == '__main__':
	import time
	import matplotlib.pyplot as plt


	#Define Noise OSRs 
	I = np.eye(2,dtype=complex).flatten()
	X = np.array([[0, 1], [1, 0]], dtype=complex).flatten()
	Y = np.array([[0, -1j], [1j, 0]], dtype=complex).flatten()
	Z = np.array([[1, 0], [0, -1]], dtype=complex).flatten()

	amplitude_dampening = lambda p: [np.array([1+0j, 0+0j, 0+0j, np.sqrt(1-p)+0j]), np.array([0+0j,np.sqrt(p)+0j, 0+0j, 0+0j])]
	dephase = lambda p: [np.sqrt(p)*I , np.sqrt(1-p)*Z]
	partial_depolarizing = lambda p: [np.sqrt(1-(3*p)/4)*I,np.sqrt(p/4)*X,np.sqrt(p/4)*Y,np.sqrt(p/4)*Z]

	PartD = NoiseChannel(partial_depolarizing(1/5),'cliffords.txt')
	DePhase = NoiseChannel(dephase(1/3),'cliffords.txt')
	AmpD_1 = NoiseChannel(amplitude_dampening(1/4),'cliffords.txt')
	AmpD_2 = NoiseChannel(amplitude_dampening(3/4),'cliffords.txt')

	#Define Initial States
	ket = np.sqrt(1/2)*np.array([1, 1j],dtype=complex)
	rho = np.outer(ket, np.conjugate(ket))
	s = StabState(X_matrix=np.array([1]),Z_matrix=np.array([1]))

	#Initialize Noise Channels
	noise_channels = [PartD,AmpD_1,DePhase,AmpD_2,PartD,AmpD_1,DePhase]
	noise_channels_str = ['PartD','AmpD_1','DePhase','AmpD_2','PartD','AmpD_1','DePhase']
	Sim = Simulation(noise_channels,lambda_weight=1/3,scaling_factor=50,verbose=False)	
	
	#Run Simulations
	Sim.uniform_exact(ket,nsim=1_000)
	Sim.run_density_exact(rho)
	Sim.run_density_approx(rho)
	Sim.run_ket_exact(ket,nsim=1_000)
	Sim.run_ket_approx(ket,nsim=1_000)
	Sim.get_stab_approx(s,nsim=1_000)
	Sim.get_stab_approx_qt(s,nsim=100_000)
	Sim.get_summary(result_type='all')




	










