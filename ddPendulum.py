#-----------------------------------------------------------------
# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

#-----------------------------------------------------------------
# Constants

# Damping constant
eps = 0.
# Driving amplitude
mu = 0.
# Driving frequency
omg = 0.8

# Define callable for the vector field in phase space
def f(p, t):
    x, y = p
    return [y, -np.sin(x) -eps*y - mu*np.cos(omg*t)]
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Define region for the plot and grid size
xmin, xmax = -4*np.pi, 6*np.pi
ymin, ymax = -4.0, 4.0
Nx, Ny = 24, 24

x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)

X, Y = np.meshgrid(x, y)
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Define initial time
t = 0

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
tmax = 5.0

for i in range(2,Nx-2):
    for j in range(2,Ny-2):
        # Set grid point as initial condition
        x0 = [X[i,j],Y[i,j]]        
        #Integrate forward
        tspan = np.linspace(0.0, tmax, 200)
        xs = spi.odeint(f, x0, tspan)
        ax.plot(xs[:,0], xs[:,1], 'm-', alpha=0.4)
        #Integrate backwards
        tspan = np.linspace(0.0, -tmax, 200)
        xs = spi.odeint(f, x0, tspan)
        ax.plot(xs[:,0], xs[:,1], 'm-', alpha=0.4) # path

# Find eigenvectors and plots stable manifolds

# List of equilibrium points
eqlist = [np.array([-np.pi,0]),np.array([np.pi,0]),np.array([3*np.pi,0])]
# Eigenvalues and eigenvectors of linearized flow
lambd, vec = np.linalg.eig([[0., 1.],[1, -eps]])

# Integrate eigenvectors at each equilibrium point
for eq in eqlist:
    for i in range(2):
        for j in range(2):
            x0 = list(eq+((-1)**j)*0.001*vec[:,i])
            tspan = np.linspace(0, ((-1)**i)*20, 200)
            xs = spi.odeint(f, x0, tspan)
            ax.plot(xs[:,0], xs[:,1], 'b-',lw=1) # path
            ax.plot([xs[0,0]], [xs[0,1]], 'o', markersize=3, color='k') # start
            
# Creates streamplot
plt.streamplot(X, Y, Vx, Vy, color='k', density=0.5, linewidth=0.7, arrowsize=1)

#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Adjust display and show plot

# Define margin
mrg = 2.

# Define axis limits and labels
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_xlim([xmin-mrg, xmax+mrg])
ax.set_ylim([ymin-mrg, ymax+mrg])

# Show or save
# plt.savefig('phase-portrait.png')
plt.show()
#-----------------------------------------------------------------    