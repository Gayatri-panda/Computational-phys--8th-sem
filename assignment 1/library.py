import matplotlib.pyplot as plt
import math
import numpy as np

def is_close_enough(x, prev_x):
    return abs(x - prev_x) < 1e-4

def g(x):
    return math.exp(-x)

def fixed_point_iteration(x0):
    prev_x = x0 - 1
    x = g(x0)
    n = 0
    while not is_close_enough(x, prev_x):
        prev_x = x
        x = g(x)
        n += 1
    return x, n

def simpsons(function, lower_limit, upper_limit, num_intervals):
	# Calculate the width of each interval
	interval_width = (upper_limit - lower_limit) / num_intervals

	# Generate the x-values for each interval
	x_values = np.linspace(lower_limit, upper_limit, num_intervals+1)

	# Calculate the y-values for each x-value
	y_values = function(x_values)

	# Apply Simpson's rule formula
	result = (interval_width / 3) * (y_values[0] 
									+ 4 * np.sum(y_values[1:-1:2]) 
									+ 2 * np.sum(y_values[2:-1:2]) 
									+ y_values[-1])
	return result


def gaussian_quadrature(f, a, b, n):
	# Compute the sample points and weights from legendre polynomials
	x, w = np.polynomial.legendre.leggauss(n)
	# Change of variables
	t = 0.5 * (x + 1) * (b - a) + a
	return np.sum(w * f(t)) * 0.5 * (b - a)



import scipy
from scipy import sparse
def crank_nicolson_1d(M, N, alpha, u_initial, T, L):
	x0, xL = 0, L
	dx = (xL - x0)/(M-1)
	t0, tF = 0, T 
	dt = (tF - t0)/(N-1)

	a0 = 1 + 2*alpha
	c0 = 1 - 2*alpha

	xspan = np.linspace(x0, xL, M)
	tspan = np.linspace(t0, tF, N)

	# Create the main diagonal for the left-hand side matrix with all elements as a0
	maindiag_a0 = a0*np.ones((1,M))

	# Create the off-diagonal for the left-hand side matrix with all elements as -alpha
	offdiag_a0 = (-alpha)*np.ones((1, M-1))

	# Create the main diagonal for the right-hand side matrix with all elements as c0
	maindiag_c0 = c0*np.ones((1,M))

	# Create the off-diagonal for the right-hand side matrix with all elements as alpha
	offdiag_c0 = alpha*np.ones((1, M-1))

	# Create the left-hand side tri-diagonal matrix
	# Get the length of the main diagonal
	a = maindiag_a0.shape[1]

	# Create a list of the diagonals
	diagonalsA = [maindiag_a0, offdiag_a0, offdiag_a0]

	# Create the tri-diagonal matrix using the sparse library
	# The matrix is then converted to a dense matrix using toarray()
	A = sparse.diags(diagonalsA, [0,-1,1], shape=(a,a)).toarray()

	# Modify specific elements of the matrix to apply certain boundary conditions
	A[0,1] = (-2)*alpha
	A[M-1,M-2] = (-2)*alpha

	# Create the right-hand side tri-diagonal matrix
	# Get the length of the main diagonal
	c = maindiag_c0.shape[1]

	# Create a list of the diagonals
	diagonalsC = [maindiag_c0, offdiag_c0, offdiag_c0]

	# Create the tri-diagonal matrix using the sparse library
	# The matrix is then converted to a dense matrix using toarray()
	Arhs = sparse.diags(diagonalsC, [0,-1,1], shape=(c,c)).toarray()

	# Modify specific elements of the matrix to apply certain boundary conditions
	Arhs[0,1] = 2*alpha
	Arhs[M-1,M-2] = 2*alpha

	#nitializes matrix U
	U = np.zeros((M, N))

	#Initial conditions
	U[:,0] = u_initial(xspan)

	#Boundary conditions
	f = np.arange(1, N+1)
	U[0,:] = 0
	f = U[0,:]
	
	g = np.arange(1, N+1)
	U[-1,:] = 0
	g = U[-1,:]
	
	#k = 1
	for k in range(1, N):
		ins = np.zeros((M-2,1)).ravel()
		b1 = np.asarray([4*alpha*dx*f[k], 4*alpha*dx*g[k]])
		b1 = np.insert(b1, 1, ins)
		b2 = np.matmul(Arhs, np.array(U[0:M, k-1]))
		b = b1 + b2  # Right hand side
		U[0:M, k] = np.linalg.solve(A,b)  # Solving x=A\b
	
	return (U, tspan, xspan)




def rk4_step(x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h/2, y + k1/2)
    k3 = h * f(x + h/2, y + k2/2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def solve_ode_rk4(initial_x, initial_y, interval_size, num_steps):

    solution = [(initial_x, initial_y)]
    x = initial_x
    y = initial_y
    for _ in range(num_steps):
        y = rk4_step(x, y, interval_size)
        x += interval_size
        solution.append((x, y))
    return solution