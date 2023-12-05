from tkwant import onebody
import kwant
import numpy as np
import matplotlib.pyplot as plt

def make_system(L):

    # system building
    lat = kwant.lattice.square(a=1, norbs=1)
    syst = kwant.Builder()

    # central scattering region
    syst[(lat(x, 0) for x in range(L))] = 1
    syst[lat.neighbors()] = -1

    return syst

# build the system using kwant
syst = make_system(400).finalized()

# lattice sites and time steps
xi = np.array([site.pos[0] for site in syst.sites])
times = np.arange(0, 1201, 50)

# define observables using kwant
density_operator = kwant.operator.Density(syst)

# initial condition
k = np.pi / 6
psi0 = np.exp(- 0.001 * (xi - 100)**2 + 1j * k * xi)

# initialize the solver
wave_func = onebody.WaveFunction.from_kwant(syst, psi0)

# loop over timesteps and plot the result
for time in times:
    wave_func.evolve(time)
    density = wave_func.evaluate(density_operator)
    plt.plot(xi, 180 * density + time, color='black')

plt.xlabel(r'lattice side $i$')
plt.ylabel(r'time $t$')
plt.show()
