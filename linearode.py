#-----------------------------------------------------------------
# Linear Systems of ODEs
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Parameter set up

# Define the matrix for the system

# Matrix of Eigenvectors (as columns)
Q = np.array([[1.,1.],[-1.2,-2.8]])

# Jordan form of the matrix (eigenvalues)
S = np.array([[-1.2,0.],[0.,-2.8]])

# Define the matrix A in standard coordinates
Q_inv = np.linalg.inv(Q)
A = Q.dot(S.dot(Q_inv))

# Define callable vector field for the SciPy integrator
def f(p, t):
    return A.dot(p)
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Define region for the plot and grid size

# Scope of x and y axes
xrange, yrange = 4.0, 4.0
# Grid size for x and y axes
Nx, Ny = 12, 12

xmin, xmax = -xrange, xrange
ymin, ymax = -yrange, yrange


x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)

X, Y = np.meshgrid(x, y)
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Define the vector field as an array for
# a stream plot or quiver plot

# Create array for vector field
Vx, Vy = np.zeros((Nx,Ny)), np.zeros((Nx,Ny))

for i in range(Nx):
    for j in range(Ny):
        x = X[i, j]
        y = Y[i, j]
        v = f([x, y], 0)
        Vx[i,j] = v[0]
        Vy[i,j] = v[1]
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Numerically ntegrating the differential equations

# Creates figure and axis
fig, ax = plt.subplots()

# Plots assorted orbits
tmax = 6.0
tgridsize = 300

for i in range(2,Nx-2):
    for j in range(2,Ny-2):
        # Set grid point as initial condition
        x0 = [X[i,j],Y[i,j]]        
        #Integrate forward
        tspan = np.linspace(0.0, tmax, tgridsize)
        xs = spi.odeint(f, x0, tspan)
        ax.plot(xs[:,0], xs[:,1], 'm-', alpha=0.6)
        #Integrate backwards
        tspan = np.linspace(0.0, -tmax, tgridsize)
        xs = spi.odeint(f, x0, tspan)
        ax.plot(xs[:,0], xs[:,1], 'm-', alpha=0.6) # path

#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Plotting

# Stream plot and quiver plot
        
# Creates streamplot (optional)
#plt.streamplot(X, Y, Vx, Vy, color='k', density=[0.5,1.0])

# Plots quiver of vector field (optional)
#ax.quiver(X, Y, Vx, Vy, color='k', alpha=1)

# Plotting eigenlines and origin

# Integrate eigenvectors at the origing
for i in range(2):
    for j in range(2):
        v0 = np.matrix(((-1)**j)*Q[:,i])
        tspan = np.matrix(np.linspace(0, 20, 200))
        tspan = tspan.T
        vs = tspan.dot(v0)
        # Plot path
        ax.plot(vs[:,0], vs[:,1], 'b-',lw=1)

# Define origin
origin = np.array([0,0])

ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')

# Plot origing
ax.plot(0, 0, 'o', markersize=3, color='k') # start
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Adjust display options and show plot

# Define scope, cuts off limits to axis on plot
scope = .5

# Define axis limits and labels
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim([xmin+scope, xmax-scope])
ax.set_ylim([ymin+scope, ymax-scope])

# Show or save
# plt.savefig('phase-portrait.png')
plt.show()
#-----------------------------------------------------------------    