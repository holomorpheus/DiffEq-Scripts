#-----------------------------------------------------------------
# The Lotka-Volterra System
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

#-----------------------------------------------------------------
# Constants

# Prey reproduction rate
a = 2.
# Prey predation rate
b = 1.
# Pradator feeding rate
k = 1.
# Predator competition rate
l = 1.

# Define callable for the vector field in phase space
def f(p, t):
    x, y = p
    return [a*x -b*x*y, -l*y + k*x*y]
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Define region for the plot and grid size
xmin, xmax = -0.7, l/k+2.
ymin, ymax = -0.7, a/b+2.
SpatialGridSize = 22
Nx, Ny = SpatialGridSize, SpatialGridSize

x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)

X, Y = np.meshgrid(x, y)
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Define initial time
t = 0.

# Create array for vector field
Vx, Vy = np.zeros((Nx,Ny)), np.zeros((Nx,Ny))

for i in range(Nx):
    for j in range(Ny):
        x = X[i, j]
        y = Y[i, j]
        v = f([x, y], t)
        Vx[i,j] = v[0]
        Vy[i,j] = v[1]
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Integrating and plotting    

#Creates figure and axis
fig, ax = plt.subplots()

# Plots quiver of vector field
#ax.quiver(X, Y, Vx, Vy, color='k', alpha=0.7)


# Plots assorted orbits
tmax0 = 3.
TimeGridSize = 200
opacity = 0.2

for i in range(2,Nx-2):
    for j in range(2,Ny-2):
        tmax = tmax0
        # Set grid point as initial condition
        x0 = [X[i,j],Y[i,j]]   
        if (x0[0] < 0.) or (x0[1] < 0.):
            tmax=.6
        #Integrate forward
        tspan = np.linspace(0.0, tmax, TimeGridSize)
        xs = spi.odeint(f, x0, tspan)
        ax.plot(xs[:,0], xs[:,1], 'm-', alpha=opacity)
        #Integrate backwards
        tspan = np.linspace(0.0, -tmax, TimeGridSize)
        xs = spi.odeint(f, x0, tspan)
        ax.plot(xs[:,0], xs[:,1], 'm-', alpha=opacity) # path

# Find eigenvectors and plots stable manifolds

# List of equilibrium points
eqlist = [np.array([0.,0.]),np.array([l/k,a/b])]

# Integrate eigenvectors at each equilibrium point
eq = eqlist[0]
x, y = eq[0], eq[1]
# Eigenvalues and eigenvectors of linearized flow
lambd, vec = np.linalg.eig([[a, 0.],[0., -l]])
for i in range(2):
    for j in range(2):
        x0 = list(eq+((-1)**j)*0.001*vec[:,i])
        tspan = np.linspace(0, ((-1)**i)*20, TimeGridSize)
        xs = spi.odeint(f, x0, tspan)
        ax.plot(xs[:,0], xs[:,1], 'b-',lw=1) # path
        ax.plot([x], [y], 'o', markersize=3, color='k') # start

eq = eqlist[1]
x, y = eq[0], eq[1]        
ax.plot([x], [y], 'o', markersize=3, color='k') # start

# Creates streamplot
#plt.streamplot(X, Y, Vx, Vy, color='k', density=0.5, linewidth=0.7, arrowsize=1)

#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Adjust display and show plot

# Define margin
mrg = 0.1

# Define axis limits and labels
ax.set_xlabel('$p_1$')
ax.set_ylabel('$p_2$')
ax.set_xlim([xmin-mrg, xmax+mrg])
ax.set_ylim([ymin-mrg, ymax+mrg])

# Show or save
# plt.savefig('phase-portrait.png')
plt.show()
#-----------------------------------------------------------------    