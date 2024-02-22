import math
import matplotlib.pyplot as plt
import math
import numpy as np
from tabulate import tabulate
import lib as lb

m=32768
def rand(r):
    
    a = 1103515245
    c = 12345
    m = 32768

    x = ((a*r+c)%m)/m
        
    return x



def com_i(a,b,f,N=4):
    h=(b-a)/N
    I=0
    for i in range(1,N+1):
        mid=(a+(2*i-1)*h/2)
        fm= f(mid)
        I+= round(fm*h,8)
    return(I)

def bracketing(a0,b0,func):
    counter=0
    a0=1.5
    b0=2.5
    while(func(a0)*func(b0)>0):
        c0=abs(a0-b0)/2
        if(abs(func(a0))<abs(func(b0))):
            a0=a0-c0
        else:
            b0=b0+c0
        counter+=1
        print("The appropriate interval for root finding is:")
        return a0,b0

#Function to find root using Regula-Falsi method
def Rf(fn, a1, b1,e=6):
    a,b = a1, b1
    if (fn(a) * fn(b) >= 0):
        print("bracket is not correct")
        return None
    n = 0
    c=a
    c1 = c-1
    while abs(c-c1) > 10**-e or abs(fn(c)) > 10**-e:
        n += 1
        c = (a * fn(b) - b * fn(a))/ (fn(b) - fn(a))
        #if abs(fn(c)) <= 10**-e:
            #break
        if (fn(c)*fn(a)<0):
            b = c
        elif (fn(c)*fn(b)<0):
            a = c
        c1 = c
        print(f"Iteration no: {n} \troot -> {c:{1}.{e}f}")
    print(f"\nThe root in the given interval converges to {c:{1}.{e}f} and the value of function is {fn(c):{1}.{e}f}")
    print("Total no of iterations = ",n)
    #return c

#Function to find root using Newton_raphson method
def Nr(fn,d_fn,x = 0.5,e=6):
    h = fn(x)/d_fn(x)
    n = 0
    while abs(h)>10**-e:
        x -= h
        h = fn(x)/d_fn(x)
        n += 1
        print(f"Iteration no: {n} \troot -> {x:{1}.{e}f}")
    print(f"The root converges to {x:{1}.{e}f} and the value of function is {fn(x):{1}.{e}f}")
    print("\nTotal no of iterations = ",n)
    return round(x,e)

def make_table(c1,c2,h):
    n=max(len(c1),len(c2))
    data=np.zeros((n,3))
    for i in range(len(c1)):
        data[i][0]=i+1
        data[i][1]=c1[i]
    for i in range(len(c2)):
        data[i][2]=c2[i]
    return tabulate(data, headers=h, tablefmt="grid") 




def Shoot(d2ydx2, dydx, x0, xf, y0, yf, z0, dx, tol, yes):
    yf1=yf+1
    r=10
    def Lag(zh,zl,yh,yl,yf):
        return zl + ((zh-zl)*(yf-yl)/(yh-yl))
    
    while abs(yf1-yf) >= tol:
        zg=z0
        
        x0_1, y0_1, z0_1 = RK3(d2ydx2, dydx, x0, y0, z0, xf, dx, 'no')
        plt.plot(x0_1,y0_1,'o',label='guess plot')
        
        n = len(y0_1)-1
        
        zg=rand(zg)
        x0_11, y0_11, z0_11 = RK3(d2ydx2, dydx, x0, y0, zg, xf, dx, 'no')
        plt.plot(x0_11,y0_11,'o',label='guess plot')
        
        if yf > y0_1[n]:

            if yf > y0_11[n]:
                zg=rand(zg)
                
                x0_11, y0_11, z0_11 = RK3(d2ydx2, dydx, x0, y0, zg, xf, dx, 'no')
                plt.plot(x0_11,y0_11,'o',label='guess plot')
                zl = z0_1[0]
                yl = y0_1[n]
                zh = z0_11[0]
                yh = y0_11[n]
    
            else:
                zh = z0_1[0]
                yh = y0_1[n]
                zl = z0_11[0]
                yl = y0_11[n]
      
                
        if yf < y0_1[n]:

            if yf < y0_11[n]:
                zg=rand(zg)
                x0_11, y0_11, z0_11 = RK3(d2ydx2, dydx, x0, y0, zg, xf, dx, 'no')
                plt.plot(x0_11,y0_11,'o',label='guess plot')
                zh = z0_1[0]
                yh = y0_1[n]
                zl = z0_11[0]
                yl = y0_11[n]
                #print('h1')
            else:
                zl = z0_1[0]
                yl = y0_1[n]
                zh = z0_11[0]
                yh = y0_11[n]
                #print('h2')
                
        z0 = Lag(zh,zl,yh,yl,yf)
        #print('L',z0)
        
        yf1 = y0_1[n]
    
    if str(yes)=='yes':
        plt.plot(x0_1,y0_1,'ro-',label="Final Plot")
        plt.legend()
        plt.show()
        print('Line here is not a fitting of the polynomial. Has been added to aid the eye to track the points.')
    return x0_1, y0_1, z0_1

import matplotlib.pyplot as plt
def RK3(d2ydx2, dydx, x0, y0, z0, xf, st, yes):
   
    x = [x0]
    y = [y0]
    z = [z0]      # dy/dx

    n = int((xf-x0)/st)     # no. of steps
    for i in range(n):
        x.append(x[i] + st)
        k1x = st * dydx(x[i], y[i], z[i])
        k1y = st * d2ydx2(x[i], y[i], z[i])
        k2x = st * dydx(x[i] + st/2, y[i] + k1x/2, z[i] + k1y/2)
        k2y= st * d2ydx2(x[i] + st/2, y[i] + k1x/2, z[i] + k1y/2)
        k3x = st * dydx(x[i] + st/2, y[i] + k2x/2, z[i] + k2y/2)
        k3y= st * d2ydx2(x[i] + st/2, y[i] + k2x/2, z[i] + k2y/2)
        k4x= st * dydx(x[i] + st, y[i] + k3x, z[i] + k3y)
        k4y= st * d2ydx2(x[i] + st, y[i] + k3x, z[i] + k3y)

        y.append(y[i] + (k1x + 2*k2x + 2*k3x + k4x)/6)
        z.append(z[i] + (k1y + 2*k2y + 2*k3y + k4y)/6)
        
    if str(yes)=='yes':
        plt.plot(x,y,'bo-',label='Final Plot')
        plt.legend()
        plt.show()
        print('Line here is not a fitting of the polynomial.')

    return x, y, z

def epde(u,time_step,x_step,x_limit,total_time):
    
    total_time=10
    time=round(total_time/time_step)
    nx=round(x_limit/x_step) #number of steps in x  
    r=time_step/(x_step)**2 #eqvivalent to alpha
    print("alpha = ", r)
    
    
    for t in range(time):
        u[0]+=r*(u[1]-(2*u[0]))
        u[nx-1]+=r*(u[nx-2]-(2*u[nx-1]))
        
        for i in range(1,nx-1):
            u[i]+=r*(u[i+1]+u[i-1]-(2*u[i]))
            
        if t%50==0 or t<10:
            if t < 101:
               plt.title("profile in intial time steps")
            else:
               plt.title("profile after 100 time steps")
            plt.plot(u)
            
        if t==100:
            plt.plot(u)
            plt.show()
    plt.show()
    return u

def LU_decompose(A,b):
    n  = len(A)  
    #convert the matrix to upper and lower triangular matrix
    for j in range(n):
        for i in range(n):
            if i <= j :
                    sum = 0
                    for k in range(i):
                        sum += A[i][k]*A[k][j]
                    A[i][j] = A[i][j] - sum
            else  :
                    sum = 0
                    for k in range(j):
                        sum += A[i][k]*A[k][j]
                    A[i][j] = (A[i][j] - sum)/A[j][j]       
#forward substitution
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += A[i][j]*b[j]
        b[i] = b [i] - sum       
#backward substitution
    for i in range(n-1,-1,-1):
        sum = 0 
        for j in range(i+1,n):
            sum += A[i][j]*b[j]
        b[i] = (b[i] - sum)/(A[i][i])
    return b  


