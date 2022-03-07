#HW01.py
#Author: Luke Runyan
#Class: CS7641
#Date: 20220212


import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time



def plot_curve(name, curve, plot):
	plt.plot(curve)
	plt.title(name)
	plt.xlabel("Iterations")
	plt.ylabel("Fitness Score")
	filename = name + ".png"
	plt.savefig(filename, dpi='figure', format='png')
	if (plot):
		plt.show()
	return


def print_results(name, state, fitness, run_time):
	print("====================================")
	print(name + " Results")
	print("State: ", state)
	print("Fit: ", fitness)
	print("Runtime:", run_time)
	print()
	return

def results(name, state, fitness, curve, run_time, plot):
	
	plot_curve(name, curve, plot)
	print_results(name, state, fitness, run_time)
	return


# Create list of city coordinates
coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3), (4, 6),(5, 8),
				(3, 7), (10, 3), (8, 4), (1, 9), (7, 4), (9, 9), (5, 1), (5, 5), (4, 8), (2,9)]

# Define optimization problem object
problem = mlrose.TSPOpt(length = 20, coords = coords_list, maximize=False)


####################################################
#						RUN ALGS
####################################################
show_plots = True


#####################RHC###############################
Solve problem using Random Hill Climbing
iterations = np.linspace(1,100,100).astype(int)
fitnesses = []

for it in iterations:
	start_time =time.time()
	state, fitness, curve = mlrose.random_hill_climb(problem, max_attempts=it, max_iters=1000, 
													restarts=0, init_state=None, curve=True, 
													random_state=1)
	run_time = time.time()-start_time
	fitnesses.append(1/fitness)

plt.plot(iterations,fitnesses)
plt.title("RHC - Fitnesses v. Iterations")
plt.xlabel("Number of Iterations")
plt.ylabel("Fitness")
plt.savefig("RHC-f_v_it.png", dpi='figure', format='png')
plt.show()


start_time = time.time()
state, fitness, curve = mlrose.random_hill_climb(problem, max_attempts=80, max_iters=1000, 
													restarts=0, init_state=None, curve=True, 
													random_state=1)
run_time = time.time()-start_time
results("RANDOM HILL CLIMB", state, 1/fitness, -1/curve, run_time, show_plots)


# # ######################SA###############################
#Solve using simulated annealing
temps = np.linspace(1,10,10)
fitnesses = []
fitmean = 0

for temp in temps:
	fitmean = 0
	for i in range(4):
		schedule = mlrose.ExpDecay(init_temp=temp, exp_const=0.009, min_temp=0.001) # Define decay schedule
		state, fitness, curve = mlrose.simulated_annealing(problem, schedule = schedule,
	                                                      max_attempts = 100, max_iters = 1000,
	                                                      init_state = None, curve=True, random_state = 1)
		fitmean+=fitnesses
	fitmean/=5
	fitnesses.append(1/fitmean)

plt.plot(temps,fitnesses)
plt.title("SA - fitnesses v. Initial Temp")
plt.xlabel("Initial Temp")
plt.ylabel("Fitness")
plt.savefig("SA-f_v_Temp.png", dpi='figure', format='png')
plt.show()

costs = np.linspace(.001,.01,10)
fitnesses = []
for cost in costs:
	fitmean = 0
	for i in range(4):
		schedule = mlrose.ExpDecay(init_temp=5.0, exp_const=cost, min_temp=0.001) # Define decay schedule
		state, fitness, curve = mlrose.simulated_annealing(problem, schedule = schedule,
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = None, curve=True, random_state = 1)
		fitmean+=fitness
	fitmean/=4
	fitnesses.append(1/fitmean)

plt.plot(costs,fitnesses)
plt.title("SA - Fitness v. Decay Constant")
plt.xlabel("Decay Constant")
plt.ylabel("Fitness")
plt.savefig("SA-f_v_dc.png", dpi='figure', format='png')
plt.show()

schedule = mlrose.ExpDecay(init_temp=4.0, exp_const=0.006, min_temp=0.001) # Define decay schedule
start_time = time.time()
state, fitness, curve = mlrose.simulated_annealing(problem, schedule = schedule,
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = None, curve=True, random_state = 1)
run_time = time.time()-start_time
results("SIMULATED ANNEALING", state, 1/fitness, -1/curve, run_time, show_plots)


#####################GA###############################
# Solve using GA
mutations = np.linspace(.01,.2,20)
fitnesses = []

for mutation in mutations:
	state, fitness = mlrose.genetic_alg(problem, mutation_prob = mutation, max_attempts = 200,
                              						max_iters=1000, curve=False)
	print(mutation, " - ", fitness)
	fitnesses.append(1/fitness)

plt.plot(mutations,fitnesses)
plt.title("GA - Fitness v. Mutation Probability")
plt.xlabel("Mutation Probability")
plt.ylabel("Fitness")
plt.savefig("GA-f_v_mp.png", dpi='figure', format='png')
plt.show()

start_time = time.time()
state, fitness, curve = mlrose.genetic_alg(problem, mutation_prob = 0.13, max_attempts = 200,
                                          					max_iters=1000, curve=True, random_state = 1)
run_time = time.time()-start_time
results("GENETIC ALGORITHM", state, 1/fitness, -1/curve, run_time, show_plots)


#####################MIMIC###############################
#Solve using MIMIC
keeps = np.linspace(.1,.9,10)
fitnesses=[]

for keep in keeps:
	state, fitness, curve = mlrose.mimic(problem, pop_size=350, keep_pct=keep, max_attempts=20,
          												max_iters=100, curve=True, random_state=1, fast_mimic=False)
	fitnesses.append(1/fitness)

plt.plot(keeps, fitnesses)
plt.title("MIMIC - Fitness v. Keep Percentage")
plt.xlabel("Keep Percentage ")
plt.ylabel("Fitness")
plt.savefig("MIMIC-f_v_kp.png", dpi='figure', format='png')
plt.show()


pops = np.linspace(50, 500,10)
fitnesses=[]

for pop in pops:
	state, fitness, curve = mlrose.mimic(problem, pop_size=pop, keep_pct=.55, max_attempts=20,
          												max_iters=100, curve=True, random_state=1, fast_mimic=False)
	print(fitness)
	fitnesses.append(1/fitness)

plt.plot(pops, fitnesses)
plt.title("MIMIC - Fitness v. Population Size")
plt.xlabel("Population Size")
plt.ylabel("Fitness")
plt.savefig("MIMIC-f_v_pop.png", dpi='figure', format='png')
plt.show()


#Solve using MIMIC
start_time = time.time()
state, fitness, curve = mlrose.mimic(problem, pop_size=350, keep_pct=0.55, max_attempts=20,
          												max_iters=100, curve=True, random_state=1, fast_mimic=False)
run_time = time.time()-start_time

results("MIMIC", state, 1/fitness, -1/curve, run_time, show_plots)