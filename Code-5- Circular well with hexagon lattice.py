import tkwant
import kwant
import numpy as np
import matplotlib.pyplot as plt
import functools as ft


def am_master():
	"""Return true for the MPI master rank"""
	return tkwant.mpi.get_communicator().rank == 0


def side_color_func(site):
	return 'r' if electrode_shape(site.pos) else ('k' if site.family == a else 'w')


def potential(site, time):
	if time <10:
		return 4 +10
	else :
		return 4  
		


def circle(pos, x0, y0, r):
	x, y = pos
	return (x - x0)**2 + (y - y0)**2 < r**2


def electrode_shape(pos):
	x, y = pos
	return x**2 +y**2<36


def make_system():
	# Define the graphene lattice.
	lat = kwant.lattice.honeycomb(a=1, norbs=1)
	a, b = lat.sublattices

	# Create graphene model.
	model = kwant.Builder(kwant.TranslationalSymmetry(lat.vec((1, 0)), lat.vec((0, 1))))
	model[[a(0, 0), b(0, 0)]] = 4
	model[lat.neighbors()] = -1

	# Central scattering region.
	funs = [ft.partial(circle, x0=0, y0=0, r=10)]
	funs1 = [ft.partial(circle, x0=0, y0=0, r=6)]
	syst = kwant.Builder()
	syst.fill(model, lambda site: any(f(site.pos) for f in funs), a(0, 0))
	syst.eradicate_dangling()
	syst[lat.shape(electrode_shape, (0, 2))] = potential


	return syst


chemical_potential = -2.5
times = [10, 15, 20, 25, 30]

syst = make_system()

# plot the system
lat = kwant.lattice.honeycomb(a=1, norbs=1)
a, b = lat.sublattices
if am_master():
	kwant.plot(syst, site_lw=0.1, site_color=side_color_func, lead_color='grey')

syst = syst.finalized()

params = {}
tparams = params.copy()
tparams['time'] = 0  # the initial time

# create an observable for calculating the average radius of a wavefunction
def radius(site):
	x, y = site.pos
	return np.sqrt(x**2 + y**2)

density_operator = kwant.operator.Density(syst)
radius_operator = kwant.operator.Density(syst, radius, sum=True)

# create a time-dependent wavefunction that starts in the ground state
hamiltonian = syst.hamiltonian_submatrix(params=tparams)
eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

ground_state = tkwant.onebody.WaveFunction.from_kwant(syst=syst,psi_init=eigenvectors[:, 0],energy=eigenvalues[0],params=params)
# evolve forward in time, calculating the average radius of the wavefunction
times = np.arange(0,100, 1)
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
	
T=[26.0,43.0,71.0,91.0]
if am_master():
	normalization = 1 / np.max(densities)
	for time, density in zip(times, densities):
		density *= normalization
		print('time={}'.format(time))
		kwant.plotter.density(syst, density, relwidth=0.15,cmap='PuOr', vmin=-0.5, vmax=0.5)

