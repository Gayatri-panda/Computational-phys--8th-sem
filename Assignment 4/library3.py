import matplotlib.pyplot as plt
import math
import numpy as np
import random


def power_method_find(A :list,x0: list,tol = 1e-6):
    '''
    # Power Method
    This function finds the largest eigenvalue and the corresponding eigenvector

    ## Condition
    - n x n matrix A has n linearly independent eigenvectors
    - Eigenvalues can be ordered in magnitude : |λ1| > |λ2| > · · · > |λn|. The λ1 is called the dominant eigenvalue and the corresponding eigenvector is the dominant eigenvector of A.

    ## Paremeters
    - A: The matrix for which the eigenvalues and eigenvectors are to be found
    - x0: The initial guess for the eigenvector
    - tol: The tolerance for the solution
    ## Returns
    - eigval: The largest eigenvalue
    - eigvec: The corresponding eigenvector
    '''
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






def QR_factorize(A):
    A = np.array(A) if type(A) != np.ndarray else A
    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)
    for i in range(A.shape[1]):
        u_i = A[:,i]
        sum = 0
        for j in range(i):
            sum += np.dot(A[:,i],Q[:,j])*Q[:,j]
        u_i = u_i - sum
        Q[:,i] = u_i/np.linalg.norm(u_i)
        for j in range(i+1):
            R[j,i] = np.dot(A[:,i],Q[:,j])
            
    return Q,R


def eigen_QR(A,tolerance = 1e-6):
    A = np.array(A)
    copy_A = np.copy(A)
    Q,R = QR_factorize(A)
    A = np.matmul(R,Q)
    i=1
    while np.linalg.norm(A-copy_A)>tolerance:
        copy_A = np.copy(A)
        Q,R = QR_factorize(A)
        A = np.matmul(R,Q)
        i+=1
    return np.diag(A),i



def InverseGJ(l):
    for i in range(len(l)):
        for j in range(len(l),2*len(l)):
            if j==(i+len(l)):
                l[i].append(1)
            else:
                l[i].append(0)
    for i in range(len(l)-1,0,-1):
        if l[i-1][0]<l[i][0]:
            l[i-1],l[i]=l[i],l[i-1]
            
    for i in range(len(l)):
        for j in range(len(l)):
            if i!=j:
                p=l[j][i]/l[i][i]
                for k in range(2*len(l)):
                    l[j][k]=l[j][k]-p*l[i][k]
    for i in range(len(l)):
        p=l[i][i]
        for j in range(2*len(l)):
            l[i][j]=l[i][j]/p
    #print(l)

    l_inv=[[0 for col in range(len(l))] for row in range(len(l))]
    for i in range(len(l)):
        for j in range(len(l),2*len(l)):
            l_inv[i][j-len(l)]=l[i][j]
    return l_inv


#Least Square fOR POLYNOMIALS
def LeastSquareFit(l):
    sum=0
    r_pt,h_pt=[],[]
    for i in range(len(l)):
        r_pt.append(l[i][0])
        h_pt.append(l[i][1])
    plt.scatter(r_pt,h_pt)
    x=[[0 for col in range(len(l[0])+2)] for row in range(len(l[0])+2)]
    y=[[0 for col in range(len(l[0])+2)] for row in range(len(l[0])+2)]
    
    #building matrix for x
    for m in range(len(l[0])+2):
        for j in range(m,len(l[0])+m+2):
            sum=0
            for k in range(len(l)):
                sum=sum+(l[k][0])**j           
            x[m][j-m]=sum
    
    
    #building matrix for y
    for i in range(len(l[0])+2):
        sum=0
        for k in range(len(l)):
            sum=sum+(l[k][1])*(l[k][0])**i
        y[i]=sum

    x_inv = InverseGJ(x)
    a=[0 for col in range(len(x_inv))]
    
    for i in range(len(x_inv)):
        for j in range(len(y)):
            a[i]=a[i]+(x_inv[i][j])*(y[j])
    
    print("Coefficients",a)
    x_plot, y_plot = [], []
    for i in range(len(l)):
        sum=0
        for j in range(len(a)):
            sum=sum+(a[j])*((l[i][0])**j)
        y_plot.append(sum)
        x_plot.append(l[i][0])
      
    plt.plot(x_plot, y_plot, label='Fitted Curve', color='red')
    plt.title("Least square fit with f(x)=a_0+a_1x+a_2x^2+a_3x^3")
    plt.ylabel("y axis")
    plt.xlabel("x axis")
    plt.show()
    return a,x_plot,y_plot




def UandL(A):
    u=[[0 for col in range(len(A))] for row in range(len(A))]
    l=[[0 for col in range(len(A))] for row in range(len(A))]
    for i in range(len(A)):
        u[0][i]=A[0][i]
        l[i][i]=1
    for j in range(len(A)):
        for i in range(1,j+1):
            sum=0
            for k in range(i):
                sum=sum+(l[i][k])*(u[k][j])
            u[i][j]=A[i][j]-sum

        for i in range(j,len(A)):
            sum=0
            for k in range(j):
                sum=sum+(l[i][k])*(u[k][j])
            if u[j][j]==0:
                continue
            else:
                l[i][j]=(A[i][j]-sum)/(u[j][j])
    return u,l



def LUBack(l,U,L):
    y=[0 for i in range(len(L))]
    x=[0 for i in range(len(L))]
    y[0]=l[0]

    #Solve for y from L.y=b using forward substitution
    for i in range(1,len(L)):
        sum=0
        for j in range(i):
            sum=sum+L[i][j]*y[j]
        y[i]=(l[i]-sum)/L[i][i]
    x[len(L)-1]=y[len(L)-1]/U[len(L)-1][len(L)-1]
    #Solve for x from U.x=y using backward substitution
    for i in range(len(L)-2,-1,-1):
        sum=0
        for j in range(len(L)):
            sum=sum+U[i][j]*x[j]
        x[i]=(y[i]-sum)/U[i][i]
    return(x)



def chebyshev(x,order):
    if order==0:
        return np.ones_like(x)
    elif order==1:
        return (2*x)-1
    elif order==2:
        return (8*(x**2))-(8*x)+1
    elif order==3:
        return (32*(x**3)-(48*(x**2))+(18*x)-1)
    elif order==4:
        return (128*(x**4)-256*(x**3)+160*(x**2)-(32*x)+1)
    
#Defining the function for chebyshev fit
def fit_chebyshev(l,order):
    x,y=[],[]
    for i in range(len(l)):
        x.append(l[i][0])
        y.append(l[i][1])
    parameters=order+1
    A=np.zeros((parameters,parameters))
    b = np.zeros(parameters)
    n=len(x)
    for i in range(parameters):
        for j in range(parameters):
            total=0
            for k in range(n):
                total+=chebyshev(x[k],j)*chebyshev(x[k],i)
            A[i,j]=total
    for i in range(parameters):
        total=0
        for k in range(n):
            total += chebyshev(x[k], i)*y[k]
        b[i]=total
    u,l=UandL(A)
    coeff=LUBack(b,u,l)
    return coeff,A


def chisquare(ob_bin,ex_bin,constrain = 1):
    #chisquare for observed and expected results
    degrees_of_freedom = len(ex_bin)-constrain
    chisqr = 0
    for j in range(len(ex_bin)):
        if ex_bin[j]<= 0:
            print("The Error observed is in expected number of instances of bin")
        temp = ob_bin[j]-ex_bin[j]
        chisqr=chisqr+ temp*temp/ex_bin[j]
    return chisqr





class rng():
    def __init__(self,seed, a = 1103515245, c = 12345 ,m = 32768):
        # initiation of data input
        self.term = seed
        self.a = a
        self.c = c
        self.m = m
    def gen(self):
        # generates a random number
        self.term = (((self.a * self.term) + self.c) % self.m)
        return self.term/self.m
    def genlist(self,length):
        # returns a list of 'n' random numbers in the range (0,1) where 'n' is 'length'.
        RNs = []
        for i in range(length):
            self.term = (((self.a * self.term) + self.c) % self.m)
            RNs.append(self.term / self.m)
        return RNs 

def LGC(a=1103515245, m=2**32, c=12345, no_sample=1, x0=0.1, repeat = True):
    rand = []
    x = x0
#     rand.append(x0)
    if repeat:
        x = x0*m
        for i in range(no_sample):
            x = (a*x+c)%m
            rand.append(x/m)
        return rand

    for i in range(no_sample):
        x = (a*x+c)%m
        rand.append(x/m)
    return rand

def monte_integrate(f: float,a: float,b: float,N: int,seed: int,multiplier: float,m: float,c: float):
    '''
    # Monte Carlo Integration
    ## Parameters
    - f: Function to be integrated
    - a: Lower limit of the integral
    - b: Upper limit of the integral
    - N: Number of random numbers to be generated
    - seed: Seed for the random number generator
    ## Returns
    - F: The value of the integral
    '''
    p=rng(seed,m=m,c=c,a=multiplier)#using the random number in problem 1
    F=0
    for i in range(N):
        k=p.gen()
        k=((b-a)*(k))+a
        F+=((b-a)*f(k))/N   
    return F  

#general monte carlo using random
def MonteCarlo(F,f,N,b,a):
    e=0.00001
    X,y_plot=[],[]
    for i in range(N):
        X.append(a+(b-a)*(random.uniform(0,1)))
    F_N = F(f, N, b, a, X)
    y_plot.append(F_N)
    return F_N
