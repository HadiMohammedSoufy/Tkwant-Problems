from math import cos, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
import functools as ft
import kwant
import tkwant

def am_master():
	"""Return true for the MPI master rank"""
	return tkwant.mpi.get_communicator().rank == 0
    
def make_system(a=1, gamma=1.0, radius=10, radius_time_dep=6):

	lat = kwant.lattice.square(a, norbs=1)
	syst = kwant.Builder()

	def circle(r):
		def inner(pos):
			(x, y) = pos
			return x ** 2 + y ** 2 < r ** 2
		return inner

	def potential(site, time):
		if time <10:
			return 4 * gamma+3
		else :
			return 4 * gamma 
		
	def side_color_func(site):
			time_dep = circle(radius_time_dep)
			return 'r' if time_dep(site.pos) else 'k'

	# Define the quantum dot
	syst[lat.shape(circle(radius), (0, 0))] = 4 * gamma
	syst[lat.shape(circle(radius_time_dep), (0, 0))] = potential
	# hoppings in x-direction
	syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = -gamma
	# hoppings in y-directions
	syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = -gamma

	# Plot system
	kwant.plot(syst, site_color=side_color_func)

	# It's a closed system, so no leads
	return syst

# construct a tight binding system and plot it
syst = make_system().finalized()

# set parameters
params = {}
tparams = params.copy()
tparams['time'] = 0  # the initial time

# create an observable for calculating the average radius of a wavefunction
def radius(site):
	x, y = site.pos
	return sqrt(x**2 + y**2)

density_operator = kwant.operator.Density(syst)
radius_operator = kwant.operator.Density(syst, radius, sum=True)

# create a time-dependent wavefunction that starts in the ground state
hamiltonian = syst.hamiltonian_submatrix(params=tparams)
eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

ground_state = tkwant.onebody.WaveFunction.from_kwant(syst=syst,psi_init=eigenvectors[:, 0],energy=eigenvalues[0],params=params)
# evolve forward in time, calculating the average radius of the wavefunction
times = np.arange(0,60, 1)
density0 = ground_state.evaluate(density_operator)
densities = []
average_radius=[]
for time in times:
	ground_state.evolve(time)
	density = ground_state.evaluate(density_operator)
	radial_density = ground_state.evaluate(radius_operator)
	average_radius.append(radial_density)
	if am_master():
		#densities.append(density )
		densities.append(density-density0)
			
plt.plot(times, average_radius)
plt.xlabel(r'time $t$')
plt.ylabel(r'average radius $n(t)$')
plt.show()
	
if am_master():
	normalization = 1 / np.max(densities)
	for time, density in zip(times, densities):
		density *= normalization
		print('time={}'.format(time))
		kwant.plotter.density(syst, density, relwidth=0.15,cmap='PuOr', vmin=-0.5, vmax=0.5)

