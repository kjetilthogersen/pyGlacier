import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import dsolve

def step_diffusion_1d(dt,dx,phi,sourceTerm,D,left_boundary_condition,left_boundary_condition_value,right_boundary_condition,right_boundary_condition_value):
	#
	# Solves 1D diffusion on staggered grid; give the midpotins only, not the artificial points outside the boundary, they are specified by the given boundary condition. 
	# Input values are dt, dx, phi (solution in the previous time-step) on the nodes, source term on the nodes, diffusion coefficient vector on midpoints (size N+1)
	# Boundary conditions can be vonNeuman or Dirichlet
	# 
	def boundary_modifier(x,dt):
		return {
        	'Dirichlet': [1,0,1],
        	'vonNeuman': [1,-1,dt]
    	}[x]
	N_intervals = np.size(phi)-1
	rhs = -dx*dx*(phi/dt+sourceTerm)
	left_modifier = boundary_modifier(left_boundary_condition,dt)
	right_modifier = boundary_modifier(right_boundary_condition,dt)
	rhs[0] = -left_modifier[2]*left_boundary_condition_value
	rhs[np.size(rhs)-1] = right_modifier[2]*right_boundary_condition_value
	A = csc_matrix(spdiags( [ np.hstack((D[0:N_intervals-1],right_modifier[1],0)),  np.hstack((left_modifier[0],-(D[0:N_intervals-1]+D[1:N_intervals]+dx*dx/dt),right_modifier[0])),   np.hstack((0,left_modifier[1],D[1:N_intervals]))  ], [-1,0,1], N_intervals+1, N_intervals+1))
	phi = dsolve.spsolve(A, rhs, use_umfpack=True)
	return phi


def step_advection_diffusion_1d(dt,dx,phi,sourceTerm,D,Vphi,left_boundary_condition,left_boundary_condition_value,right_boundary_condition,right_boundary_condition_value):
	#
	# Solves 1D advection diffusion on staggered grid; give the midpotins only, not the artificial points outside the boundary, they are specified by the given boundary condition. 
	# Input values are dt, dx, phi on the nodes, source term on the nodes, diffusion coefficient vector on midpoints (size N+1), as well as advection term Vphi on midpoints
	# Boundary conditions can be vonNeuman or Dirichlet
	# 
	def boundary_modifier(x,dt):
		return {
        	'Dirichlet': [1,0,1],
        	'vonNeuman': [1,-1,dt]
    	}[x]
	N_intervals = np.size(phi)-1
	left_modifier = boundary_modifier(left_boundary_condition,dt)
	right_modifier = boundary_modifier(right_boundary_condition,dt)
	rhs = -dx*dx*(phi/dt+sourceTerm + np.gradient(Vphi)/dx) #Here we are treating the advection term explicitly as an rhs contribution
	rhs[0] = -left_modifier[2]*left_boundary_condition_value
	rhs[-1] = right_modifier[2]*right_boundary_condition_value

	A = csc_matrix(spdiags( [ np.hstack((D[0:N_intervals-1],right_modifier[1],0)),  np.hstack((left_modifier[0],-(D[0:N_intervals-1]+D[1:N_intervals]+dx*dx/dt),right_modifier[0])),   np.hstack((0,left_modifier[1],D[1:N_intervals]))], [-1,0,1], N_intervals+1, N_intervals+1))
	phi = dsolve.spsolve(A, rhs, use_umfpack=True)
	return phi


def step_elliptical_1d(dx,W,alpha,beta,gamma,left_boundary_condition,left_boundary_condition_value,right_boundary_condition,right_boundary_condition_value):
	#
	# Solves 1D elliptical equation on staggered grid
	# (W(x) u_x)_x - alpha(x) u + gamma(x)*u_x = beta(x)
	#
	def boundary_modifier(x,val):
		return {
			'Dirichlet': [val,0,val],
			'vonNeuman': [-val,val,-val]
		}[x]
	N_intervals = np.size(alpha)-1
	val = np.mean(np.abs(2*W+dx**2.0*alpha - gamma*dx))
	left_modifier = boundary_modifier(left_boundary_condition,val)
	right_modifier = boundary_modifier(right_boundary_condition,val)
	rhs = dx*dx*beta
	rhs[0] = left_modifier[2]*left_boundary_condition_value
	rhs[np.size(rhs)-1] = right_modifier[2]*right_boundary_condition_value
	A = csc_matrix(spdiags( [ np.hstack((W[0:N_intervals-1], right_modifier[1],0)),  np.hstack((left_modifier[0],-(W[0:N_intervals-1]+W[1:N_intervals]+alpha[1:N_intervals]*dx*dx) - (gamma[1:N_intervals]+gamma[0:N_intervals-1])/2.0*dx,right_modifier[0])),   np.hstack((0,left_modifier[1],W[1:N_intervals] + (gamma[1:N_intervals]+gamma[0:N_intervals-1])/2.0*dx))  ], [-1,0,1], N_intervals+1, N_intervals+1))
	u = dsolve.spsolve(A, rhs, use_umfpack=True)
	return u




