import numpy as np
import math as m
import copy 
#Creating a zero matrix of order m*n
def zeromatrix(m,n):
        p= [[0 for i in range(n)] for j in range(m)]
        return(p)

#Creating a identity matrix of m*m
def identity_mat(m):
        p=zeromatrix(m,m)
        for i in range(m):
                p[i][i] = 1
        return(p)
        
def mat_vec_mult(A,B):
        n=len(B)
        if len(A[0])==n:
                p=[0 for i in range(n)]
                for i in range(n):
                        for j in range(n):
                                p[i] = p[i] + (A[i][j] * B[j])
                return(p)
        else:
                print('This combination is not suitable for multiplication')


#matrix multiplication
def mat_mult(a,b):
        if len(a[0])==len(b):
                p=zeromatrix(len(a),len(b[0]))
                for i in range(len(a)):
                        for j in range(len(b[0])):
                                for x in range(len(b)):
                                        p[i][j]+=(a[i][x]*b[x][j])
                return(p)
        else:
                print('The matrix combination is not suitable for multiplication')

#divide by a scaler 
def scaler_matrix_division(c,A):
        cA = zeromatrix(len(A), len(A[0]))
        for i in range(len(A)):
                for j in range(len(A[i])):
                        cA[i][j] = A[i][j]/c
        return cA

def scaler_matrix_multiplication(c,A):
    cA = zeromatrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            cA[i][j] = c * A[i][j]
    return cA

def matrix_addition(A, B):
    
    ra = len(A)
    ca = len(A[0])
    rb = len(B)
    cb = len(B[0])
    
    if ra != rb or ca != cb:
        raise ArithmeticError('Matrices are NOT of the same dimensions!.')
    
    C = zeromatrix(ra, cb)
    
    for i in range(ra):
        for j in range(cb):
            C[i][j]=A[i][j] + B[i][j]
    return C

def matrix_substraction(A, B):
    
    ra = len(A)
    ca = len(A[0])
    rb = len(B)
    cb = len(B[0])
    
    if ra != rb or ca != cb:
        raise ArithmeticError('Matrices are NOT of the same dimensions!.')
    
    C = zeromatrix(ra, cb)
    
    for i in range(ra):
        for j in range(cb):
            C[i][j]=A[i][j] - B[i][j]
    return C

def transpose(A):
        if not isinstance(A[0],list):
                A = [A]

        r = len(A)
        c = len(A[0])
        AT = zeromatrix(c, r)
        #Copy values from A to it's transpose AT
        for i in range(r):
                for j in range(c):
                        AT[j][i] = A[i][j]
        return AT

def inner_product(A,B):
    
    AT = transpose(A)

    C = mat_mult(AT, B)

    return C[0][0]

#Partial pivoting
def par_pivot(A,B):
        n=len(A)
        for r in range(n):
                if A[r][r]==0:
                        for r1 in range(r+1,n):
                                if abs(A[r1][r])>A[r][r] and A[r][r]==0:
                                        (A[r],A[r1])=(A[r1],A[r])
                                        (B[r],B[r1])=(B[r1],B[r])
                                else:
                                        continue
                        else:
                                continue

#Gauss-Jordan elimination
def gauss_jordan(A,B):
        m=len(A)
        n=len(A[0])
        for r in range(m):
                par_pivot(A,B)
                pivot=A[r][r]
                for c in range(r,n):
                        A[r][c]=A[r][c]/pivot
                B[r]=B[r]/pivot
                for r1 in range(m):
                        if r1==r or A[r1][r]==0:
                                continue
                        else:
                                factor=A[r1][r]
                                for c in range(r,n):
                                        A[r1][c]=A[r1][c]-A[r][c]*factor
                                B[r1]=B[r1]-B[r]*factor


#LU decomposition of a matrix
def lu_decompose(A,B):
        par_pivot(A,B)
        n=len(A)
        #To store in one matrix both L and U in matrix a
        try:
                import copy
                a=copy.deepcopy(A)
                for j in range(n):
                        for i in range(n):
                                factor=0

                                #for U(upper A) matrix
                                if i<=j:
                                        for k in range(i):
                                                factor+=a[i][k]*a[k][j]
                                        a[i][j]=A[i][j]-factor
                                #for L(lower) matrix
                                else:
                                        for k in range(j):
                                                factor+=a[i][k]*a[k][j]
                                        a[i][j]=1/a[j][j]*(A[i][j]-factor)
        except ZeroDivisionError:
                        print('LU decomposition is not possible.')

        return(a,B)

#for LUx=B
def lux(a,B):
        n=len(B)
        det =1
        for i in range(n):
                det*=a[i][i]
        if len(a)==n and det !=0:
                print
                y=[0 for i in range(n)]
                x=[0 for i in range(n)]

                #forward substitution i.e., Ly=B
                for i in range(len(a)):
                        factor = 0
                        for j in range(i):
                                factor+=a[i][j]*y[j]
                        y[i]=B[i]-factor
                #Backward substitution, i.e. Ux=y
                for i in range(len(a)-1,-1,-1):
                        factor=0
                        for j in range(i+1,len(a),1):
                                factor+=(a[i][j]*x[j])
                        x[i]=1/a[i][i]*(y[i]-factor)
        return(x)


#for bracketing
def bracket(f,a,b):
        if f(a) ==0:
                print(a,'is the root of the equation.')
        elif f(b)== 0:
                print(b,'is the root of the equation.')
        else:
                while f(a)*f(b)>0:
                        if abs(f(a)) < abs(f(b)):
                                a=a-1.5*(b-a)
                        elif abs(f(a)) > abs(f(b)):
                                b=b+1.5*(b-a)
        return a,b

#for finding a root using bisection method
def bisection(f,a,b):
        k=0
        err=[]
        print('SR.No.  Absolute error ')
        while abs(b-a)>10**(-6) and k<200:
                c = (a+b)/2
                if f(a)*f(c)<0:
                        b=c
                        k+=1
                else:
                        a=c
                        k+=1
                err.append(c)
        n= len(err)
        arr=[0 for i in range(n-1)]
        for i in range(n-1):
                arr[i]=abs(err[i+1]-err[i])
                print(i+1,'     ',arr[i])

        return c,arr

#for finding a root using false position method
def fal_pos(f,a,b):
        k=0
        err=[]
        c = b-(((b-a)*f(b))/(f(b)-f(a)))
        print('SR.No.  Absolute error ')
        while abs(f(c))>10**(-6) and k<200:
                c = b-(((b-a)*f(b))/(f(b)-f(a)))
                if f(a)*f(c)>0:
                        a=c
                        k+=1
                else:
                        b=c
                        k+=1
                err.append(c)
        n= len(err)
        arr=[0 for i in range(n-1)]
        for i in range(n-1):
                arr[i]=abs(err[i+1]-err[i])
                print(i+1,'     ',arr[i])
        return c, arr

#for finding a root using Newton-raphson method
def newtraph(f,a):
        i=0
        c=a
        err=[]
        print('SR.No.  Absolute error ')
        while abs(f(c))>= 10**(-10) and i<200:
                c = a - f(a)/der1(f,a)
                i+=1
                a=c
                err.append(c)
                n= len(err)
        arr=[0 for i in range(n-1)]
        for i in range(n-1):
                arr[i]=abs(err[i+1]-err[i])
                print(i+1,'     ',arr[i])
        return c,arr

#1st derivatives of a function
def der1(f,x):
        h=10**(-3)
        f_ = (f(x+h)-f(x-h))/(2*h)
        return f_

#2nd derivative of function
def der2(f,x):
        h=10**(-3)
        f__ = (der(f,x+h)-der(f,x-h))/(2*h)
        return f__

#Value of p(x)
def poly(f,x):
        value=0
        n = len(f)
        for i in range(n):
                value+=f[i]*(x**(n-1-i))
        return value

#1st derivatives of p(x) at point x
def der1_poly(f,x):
        value=0
        n= len(f)
        for i in range(n-1):
                value+=f[i]*(n-1-i)*(x**(n-i-2))
        return value

#2nd derivative of p(x) at a point x
def der2_poly(f,x):
        value=0
        n=len(f)
        for i in range(n-2):
                value+=f[i]*(n-1-i)*(n-2-i)*(x**(n-i-3))
        return value

def laguerre(f,x):
        h=10**(-8)#epsilon
        n=len(f)-1#degree of polynomial
        i=0
        if abs(poly(f,x))<h: #checking
                return x
        else:
                while abs(poly(f,x))>h and i<100:
                        g=der1_poly(f,x)/poly(f,x)
                        h=g**2-(der2_poly(f,x)/poly(f,x))
                        d1=g+(((n-1)*(n*h-g**2))**0.5)
                        d2=g-(((n-1)*(n*h-g**2))**0.5)
                        #denominator should be larger
                        if abs(d1)>abs(d2):
                                a=n/d1
                        else:
                                a=n/d2
                        x=x-a
                        i+=1#iteration number
                return x

#To find the root of polynomial using laguerre method
def root_poly(q2):
        deg = len(q2)-1#degree of polynomial
        #array to store root
        root = [0 for i in range(deg)]
        for i in range(deg):
                newp =[]
                for j in range(deg+1-i):
                        newp.append(q2[j])
                root[i] = laguerre(newp,5)
                r=0
                for j in range(deg-i):#Resizing the polynomial after synthetic devision
                        q2[j]+=r*(root[i])
                        r=q2[j]
        return root

import math as m
#integration of function 'f' with range [a,b], the range is divided with 'n' number of equal interval
def midpoint(f, a, b, n):
        h=(b-a)/n
        sum=0
        for i in range(n):
                x = a +(2*i+1)*(h/2)
                sum+=f(x)*h
        return(sum)
#To find n: 'f_max' is |f(x)''_max| in range [a,b] and 'err' is the maximum error to be compromised
def n_midpoint(f_max, a, b, err):
        n=((b-a)**3*f_max/(24*err))**0.5
        return m.ceil(n)


def trapezoidal(f, a, b, n):#trapezoidal method
        h=(b-a)/n
        sum=(f(a)+f(b))*(h/2)
        for i in range(1, n):
                x = a + i*h
                sum+=f(x)*h
        return(sum)
def n_trapezoidal(f_max, a, b, err):# to find the N for a particular error for trapezoidal method
        n=((b-a)**3*f_max/(12*err))**0.5
        return m.ceil(n)

def simpson(f, a, b, n):#simpson method
        h=(b-a)/n
        sum=(f(a)+f(b))*(h/3)
        for i in range(1, n):
                x = a + i*h
                if i%2==0:
                        sum+=f(x)*(2*h/3)
                else:
                        sum+=f(x)*(4*h/3)
        return(sum)
def n_simpson(f_max, a, b, err):#Calculating the value of N for an error "err"
    n=((b-a)**5*f_max/(180*err))**0.25
    if m.ceil(n)%2==0:#N should be smallest even number greater than n
        return m.ceil(n)
    else:
        return m.ceil(n)+1

#Monte Carlo methods
def monte_carlo(f, a, b, n):
        sum=0
        sum1=0
        import random as r
        for i in range(1,n+1):
                x=a+(b-a)*r.random()#creating random number between a to b
                sum+=f(x)/n
                sum1+=(f(x)**2)/n
                error=(sum1-(sum)**2)**0.5
        return((b-a)*sum,error)#returning both result and error
global seed
def mlcg_random_number(a, m):#random number generation between 0 to 1 with linear congruential generator method
	global seed
	seed = a* seed % m
	return seed/m
#Solution for differential equation using Euler's method of type: dy/dx=f(x) , y(x_0)=y
def euler(f,y,x,x_n,dx):#taking parameter(dy/dx,y(x_0),x_0,x_max,dx)
	arrx=[x]#storing the vaalue in list
	arry=[y]
	while x < x_n:#loop upto x_max
		y+=dx*f(y,x)
		x+=dx
		arrx.append(x)#updating list
		arry.append(y)
	return arry, arrx

def rk4(dvdx, y, v, x, dx, x_min, x_max):
#this function is for solving 2nd order ODE with given y(x_0), dy/dx at x_0 
#x_min and x_max are boundary of x 
#dvdx = y''
        arrx=[x]#creating list to store the values
        arry=[y]
        arrv=[v]
        while x>x_min:#backward loop for boundary
                k1y = -dx*v
                k1v = -dx*dvdx(y, v, x)
		
                k2y = -dx*(v + 0.5*k1v)
                k2v = -dx*dvdx(y + 0.5*k1y, v+0.5*k1y, x-0.5*dx)
		
                k3y = -dx*(v + 0.5*k2v)
                k3v = -dx*dvdx(y + 0.5*k2y, v+0.5*k2y, x-0.5*dx)
		
                k4y = -dx*(v + 0.5*k3v)
                k4v = -dx*dvdx(y + k3y, v + k3y, x - dx)
		
                y += (k2y + 2*k2y + 2*k3y + k4y)/6
                v += (k2v + 2*k2v + 2*k3v + k4v)/6
                x -= dx
		
                arry.append(y)#appending the values in list
                arrv.append(v)
                arrx.append(x)
        x=arrx[0]#assigning the initial values
        y=arry[0]
        v=arrv[0]
                #print(y,v,x)
        while x < x_max:#forward loop
                k1y = dx*v
                k1v = dx*dvdx(y, v, x)
		
                k2y = dx*(v + 0.5*k1v)
                k2v = dx*dvdx(y + 0.5*k1y, v+0.5*k1y, x+0.5*dx)
	
                k3y = dx*(v + 0.5*k2v)
                k3v = dx*dvdx(y + 0.5*k2y, v+0.5*k2y, x+0.5*dx)
		
                k4y = dx*(v + 0.5*k3v)
                k4v = dx*dvdx(y + k3y, v + k3y, x + dx)
		
                y += (k2y + 2*k2y + 2*k3y + k4y)/6
                v += (k2v + 2*k2v + 2*k3v + k4v)/6
                x += dx

                arry.append(y)#appending
                arrv.append(v)
                arrx.append(x)
        return arrv, arry, arrx#returning all the list

'''
def shooting(dvdx, x1, y1, x2, y2, z, dx):
#This function is for solving 2nd ODE with boundary values
#uses rk4 method
#y1 , y2 are boundary values at x1 and x2 respectively, dx is stepsize
	def yc(dvdx, y1, z, x1, dx, xl, xh):#defining a function to return the last value of list_y in rk4
	        v, y, x = rk4(dvdx, y1, z, x1, dx, xl, xh)
        	return y[-1]
	beta = yc(dvdx, y1, z, x1, dx, x1, x2)
	beta_zl=beta
	beta_zh=beta
	if abs(beta-y2)<0.001:#
		v, y, x = rk4(dvdx, y1, z, x1, dx, x1, x2)
		return v,y,x#returning all the list
	else:
		if beta>y2:#if we get the upper bound
			zh=z
			while beta>y2:
				z-=0.5#decreasing to get the lower bound
				beta=yc(dvdx, y1, z, x1, dx, x1, x2)
			zl=z
			beta_zl=beta
                        #langrange interpolation
			z=zl+(zh-zl)*(y2-beta_zl)/(beta_zh-beta_zl)
			v, y, x = rk4(dvdx, y1, z, x1, dx, x1, x2)
			return v,y,x
		else:
			zl=z
			while beta<y2:#if we get the lower bound
				z+=0.5#incresing to get upper bound
				beta=yc(dvdx, y1, z, x1, dx, x1, x2)
			zh=z
			beta_zh=beta
			#langrange interpolation
			z=zl+(zh-zl)*(y2-beta_zl)/(beta_zh-beta_zl)
			v, y, x = rk4(dvdx, y1, z, x1, dx, x1, x2)
		return v,y,x
 '''
#--------------------Jacobi Method----------------------------------------------------
 #Jacobi method for linear equation solution 
 #A is the equation matrix of linear equations
 #B is the scaler matrix or result
def jacobi(A,b,eps=1.0e-4): 
    max_iter = 1000
    import copy 
    A1=copy.deepcopy(A)
    b1=copy.deepcopy(b)
    D=[1/A[i][i] for i in range(len(A))]
    for i in range(len(A)):
        A[i][i]=0
    x_k = [1 for i in A]#initialising solution array
    convergence=[[],[]]#iteration step and error
    residue = [[],[]]#iteration step and residue
    for i in range(max_iter):
        Rx = mat_vec_mult(A,x_k)
        b_Rx=[b[i]-Rx[i] for i in range(len(b))]
        x_k1 = [D[i]*b_Rx[i] for i in range(len(D))]
        #print(x_k)
        
        #residue calculation
        AX=mat_vec_mult(A1,x_k1)
        del_Ax_b = [abs(AX[i] - b[i]) for i in range(len(x_k))]
        residue[0].append(i+1)
        residue[1].append(sum(del_Ax_b))

        #convergence calculation
        del_x = [abs(x_k1[i]-x_k[i]) for i in range(len(x_k))]
        error=sum(del_x)
        convergence[0].append(i+1)
        convergence[1].append(error)


        if error < eps:
            return x_k1,convergence,residue
        else:
            x_k = x_k1

def inf_norm(X,Y):
    max=0

    sum=0
    for i in range(len(X)):
        for j in range(len(X[i])):
            diff = abs(X[i][j]-Y[i][j])
            
            sum = sum + diff
            
        if sum>max:
            max = sum
    return max

#--------------------Gauss Seidel--------------------------------
 #Gauss Seidel for linear equation solution 
 #A is the equation matrix of linear equations
 #B is the scaler column matrix or result (len(A)x1)
 #A does not changes after the operation. So no need to deepcopy A
def gauss_seidel(A, B, eps=1.0e-4):
    
    # Check: A should not have zero on diagonals
    for i in range(len(A)):
        if A[i][i] == 0:
            return ("Main diagonal should not have zero!")

        
    xk0 = zeromatrix(len(A),1)
    xk1 = zeromatrix(len(A),1)
    for i in range(len(xk0[0])):
        for j in range(len(xk0)):
            xk0[j][i]=1

    #print("xk1",xk1)
    c=0
    residue = [[],[]]#iteration step and residue
    while inf_norm(xk1,xk0) >= eps:
        
        if c!=0:
                for i in range(len(xk1)):
                    for j in range(len(xk1[i])):
                        xk0[i][j]=xk1[i][j]
        for i in range(len(A)):
            sum1 = 0
            sum2 = 0
            for j in range(i+1,len(A[i])):
                sum2 = sum2 + (A[i][j]*xk0[j][0])
            for j in range(0,i):
                sum1 = sum1 + (A[i][j]*xk1[j][0])
            xk1[i][0] = (1/A[i][i])*(B[i][0]-sum1-sum2)
            
        c=c+1

        #residue calculation
        AX=mat_mult(A,xk1)
        del_Ax_b = [abs(AX[i][0] - B[i][0]) for i in range(len(xk1))]
        residue[0].append(c)
        residue[1].append(sum(del_Ax_b))
    #print("c=",c)
        
    return xk1, residue   

#inverse of matrix A using Gauss seidel method
def gauss_seidel_inverse(A,eps=1.0e-4):
        I=identity_mat(len(A))
        for i in range(len(A)):
                I[i],residue=gauss_seidel(A,transpose(I[i]),eps)
        A_inv=zeromatrix(len(A),len(A))
        for i in range(len(A)):
                for j in range(len(A)):
                        A_inv[i][j]=I[i][j][0]
        A_inv=transpose(A_inv)
        return A_inv,residue
        


#----------------------------Power Method-------------------------------------------------
#For getting the largest eigen value of the matrix 
import math
def frob_norm(A):
    sum = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            sum = sum + (A[i][j]**2)
    return math.sqrt(sum)

def power_normalize(A):
    max = 0
    for i in range(len(A)):
        if max <= A[i][0]:
            max = A[i][0]
    normA = scaler_matrix_division(max,A)
    return normA



def power_method(A, x0 = [[1],[1],[1]], eps=1.0e-4):
    i = 0
    lam0 = 1
    lam1 = 0
    while abs(lam1-lam0) >= eps:
        #print("error=",abs(lam1-lam0))
        if i != 0:
            lam0 = lam1
        
        Ax0 = mat_mult(A,x0)
        AAx0 = mat_mult(A,Ax0)
        #print("Ax0=",Ax0)
        #print("AAx0=",AAx0)
        dotU = inner_product(AAx0,Ax0)
        dotL = inner_product(Ax0,Ax0)
        #print("U=",dotU)
        #print("L=",dotL)
        lam1 = dotU/dotL
        
        x0 = Ax0
        i = i+1
        #print("i=",i)
        
        #print("eigenvalue=",lam1)
        ev = power_normalize(x0)
        #print ("eigenvector=",ev)
    return lam1, ev#returns lam1=largest eigen value and ev = coressponding eigen vec

#---------------------Conjugate gradient-------------------------------------
def conju_norm(A):
    sum=0
    for i in range(len(A)):
        sum = sum + abs(A[i][0])
    return sum

def conjugate_gradient(A, B, x0 , eps=1.0e-4):
    #r0 = make_matrix(len(A), 1)
    xk = [i for i in x0]
    
    #r0=b-Ax0
    Ax0 = mat_mult(A, x0)
    rk = matrix_substraction(B, Ax0)
    #print("rk",rk)
    dk = [i for i in rk]
    #print("dk",dk)
    i=0
    
    while conju_norm(rk)>=eps and i in range(len(A)):
        adk = mat_mult(A,dk)
        #print("adk=",adk)
        rkrk = inner_product(rk, rk)
        #print("rkrk = ", rkrk)
        alpha = rkrk/inner_product(dk, adk)
        #print("alpha = ",alpha)
        xk = matrix_addition(xk, scaler_matrix_multiplication(alpha, dk))
        #print("xk1=",xk)
        rk = matrix_substraction(rk, scaler_matrix_multiplication(alpha, adk))
        #print("rk1=",rk)
        beta = inner_product(rk, rk)/rkrk
        dk = matrix_addition(rk, scaler_matrix_multiplication(beta, dk))
        
        i = i+1
        #print("norm=",conju_norm(rk))
        #print("i=",i)
    return xk


def matrix_multiplication_on_the_fly(Afn,B):
    n = int(math.sqrt(len(B)))
    #print('B',len(B))
    #print('n',n)
    m = zeromatrix(len(B),1)
    for i in range(len(B)):
        for j in range(len(B)):
            m[i][0] = m[i][0] + (Afn(i,j,n) * B[j][0])
    #print('m',m)
    return m



def conjugate_gradient_on_the_fly(Afn, B, eps):
    x0 = []
    a=[1]
    for i in range(len(B)):
        x0.append(a)
    #print('x01',x0)
    
    import copy
    xk = copy.deepcopy(x0)
    

    #r0=b-Ax0
    Ax0 = matrix_multiplication_on_the_fly(Afn, x0)
    #print("Ax0",Ax0)
    rk = matrix_substraction(B, Ax0)
    #print("rk",rk)
    i = 0
    dk = copy.deepcopy(rk)
    #print("dk",dk)
    
    iteration=[]
    residue=[]
    while math.sqrt(inner_product(rk,rk))>=eps and i <= 1000:# and i in range(len(A)):
        adk = matrix_multiplication_on_the_fly(Afn,dk)
        #print("adk=",adk)
        rkrk = inner_product(rk, rk)
        #print("rkrk = ", rkrk)
        alpha = rkrk/inner_product(dk, adk)
        #print("alpha = ",alpha)
        xk = matrix_addition(xk, scaler_matrix_multiplication(alpha, dk))
        #print("xk1=",xk)
        rk = matrix_substraction(rk, scaler_matrix_multiplication(alpha, adk))
        #print("rk1=",rk)
        beta = inner_product(rk, rk)/rkrk
        dk = matrix_addition(rk, scaler_matrix_multiplication(beta, dk))
        
        i = i+1
        #print("norm=",math.sqrt(inner_product(rk,rk)))
        #print("i=",i)
        iteration.append(i)
        residue.append(math.sqrt(inner_product(rk,rk)))
    return xk, iteration, residue


#----------------------------linear regression-------------------------------------------------------

def linear_regression(X,Y, sig):
    
    chi2 = 0
    S = 0
    Sx = 0
    Sy = 0
    Sxx = 0
    Sxy = 0
    Syy = 0
    Y1 = [0 for i in range(len(X))]
    
    for i in range(len(X)):
        
        S = S + 1/(sig[i]**2)
        Sx = Sx + X[i]/(sig[i]**2)
        Sy = Sy + Y[i]/(sig[i]**2)
        Sxx = Sxx + (X[i]**2)/(sig[i]**2)
        Sxy = Sxy + (X[i]*Y[i])/(sig[i]**2)
        
    delta = S*Sxx - (Sx**2)
    a = (Sxx*Sy - Sx*Sxy)/delta
    b = (S*Sxy - Sx*Sy)/delta
        
    covab = -Sx/delta
    sig2_a = Sxx/delta
    err_a = math.sqrt(sig2_a)
    sig2_b = S/delta
    err_b = math.sqrt(sig2_b)
    for i in range(len(X)):
        Y1[i] = a + b * X[i]
        chi2 = chi2 + ((Y[i] - Y1[i])/sig[i])**2
        
    return a,b, covab, err_a, err_b

def readfile(filename,start):
    with open(filename,"r+") as f:
        lines = f.readlines()
        A=[]
        for i in range(start,len(lines)):
            A.append([float(j) for j in lines[i].split()])
        del lines
        return A

#--------------------------------------Polynomial fitting---------------------------------------------
import copy
def poly_value(X, param):#return y dataset of a polynomial with given X dataset and parameters
    Y = []
    for i in range(len(X)):
        sum = 0
        for j in range(len(param)):
            sum = sum + param[j] * (X[i]**j)
        Y.append(sum)
    return Y


def poly_fit(X, Y, sig, order = 2):
    if len(X) != len(Y):
        print("Error: X and Y must be same length")
    else:
        N=len(X)
    A = zeromatrix(order+1,order+1)
    B = [0 for i in range(order + 1)]
    for i in range(N):
        for j in range(order+1):
            B[j] += (Y[i]*X[i]**j) / sig[i]**2
            for k in range(order+1):
                A[j][k] += X[i]**(j+k) / sig[i]**2

    A1 = copy.deepcopy(A)
    A1,B = lu_decompose(A1,B)
    B = lux(A1,B)

    chi2=0
    for i in range(N):
        sum=0
        for j in range(order+1):
            sum += B[j]*X[i]**j
        chi2 += (Y[i]-sum)**2/sig[i]**2

    return B,chi2

def special_func_value(X, phi, coefficient):
    Y = []
    for i in range(len(X)):
        sum = 0
        for j in range(len(phi)):
            sum = sum + coefficient[j] * phi[j](X[i])
        Y.append(sum)
    return Y

def poly_fit_special(X, Y, sig, phi):
    if len(X) != len(Y):
        print("Error: X and Y must be same length")
    else:
        N=len(X)
    order = len(phi)-1
    A = zeromatrix(order+1,order+1)
    B = [0 for i in range(order + 1)]
    for i in range(N):
        for j in range(order+1):
            B[j] += ((phi[j](X[i])) * Y[i]) / (sig[i]**2)
            for k in range(order+1):
                A[j][k] = A[j][k] + (((phi[j](X[i]))*(phi[k](X[i]))) / (sig[i]**2))

    A1 = copy.deepcopy(A)
    A1,B = lu_decompose(A1,B)
    B = lux(A1,B)

    chi2=0
    for i in range(N):
        sum=0
        for j in range(order+1):
            sum += B[j]*X[i]**j
        chi2 += (Y[i]-sum)**2/sig[i]**2
    return B, chi2


#-------------------------gauss-quadrature----------------------------------


def gaussian_quadrature(f, degree, a, b):
    res = np.polynomial.legendre.leggauss(degree)#zeros and degree has been taken from library
    zeros = res[0]
    weights = res[1]

    #print(res)
    sum = 0

    for i in range(len(zeros)):
        y = ((b-a)*0.5*zeros[i])+((b+a)*0.5)
        sum = sum + weights[i]*f(y)
    ans = (b-a)*0.5*sum
    return ans, res

#----Random walk-----------------
def rand_walk(length, step):
    x1 = 0
    y1 = 0
    xdata = [x1]  # x cordinate of the random walk
    ydata = [y1]  # y cordinate of the random walk
    for i in range(step - 1):
        theta = 2 * m.pi * mlcg_random_number()
        x1 += length * m.cos(theta)
        y1 += length * m.sin(theta)
        xdata.append(x1)
        ydata.append(y1)
    return xdata, ydata  # returning the coordinte of each steps in a list


def creat_walk(length, step):
    x = []  #
    y = []
    r_x = []
    r_y = []
    for i in range(500):# for 500 walks
        x1, y1 = rand_walk(length, step)
        x.append(x1)
        y.append(y1)
        r_x.append(x1[-1])
        r_y.append(y1[-1])
    return x, y, r_x, r_y#



def power_method_find(A :list,x0: list,tol = 1e-6):
    A=np.array(A)
    x0=np.array(x0)
    x_copy = np.copy(x0)
    lam_0 = np.matmul(np.matmul(np.linalg.matrix_power(A,2),x0).T,np.matmul(np.linalg.matrix_power(A,1),x0))/np.matmul(np.matmul(np.linalg.matrix_power(A,1),x0).T,np.matmul(np.linalg.matrix_power(A,1),x0))
    lam_1 = np.matmul(np.matmul(np.linalg.matrix_power(A,3),x0).T,np.matmul(np.linalg.matrix_power(A,2),x0))/np.matmul(np.matmul(np.linalg.matrix_power(A,2),x0).T,np.matmul(np.linalg.matrix_power(A,2),x0))
    i=3
    while abs(lam_1-lam_0)>tol:
        lam_0 = lam_1
        lam_1 = np.matmul(np.matmul(np.linalg.matrix_power(A,i+1),x0).T,np.matmul(np.linalg.matrix_power(A,i),x0))/np.matmul(np.matmul(np.linalg.matrix_power(A,i),x0).T,np.matmul(np.linalg.matrix_power(A,i),x0))
        i+=1

    eigval = lam_1
    eigvec = np.matmul(np.linalg.matrix_power(A,i-1),x_copy)
    norm = np.linalg.norm(eigvec)
    eigvec = eigvec/norm
    return eigval,eigvec,i  