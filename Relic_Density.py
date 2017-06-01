
# coding: utf-8

#Outline of algorithm from micrOMEGAs https://arxiv.org/pdf/1402.0787.pdf
import numpy as np
import math
import scipy.special as sc
from scipy.interpolate import interp1d
from scipy.special import kn
from scipy.integrate import quad
from scipy.integrate import quadrature
from scipy.optimize import brentq

from constants import *

Delta_Y_Condition=0.1
Y_ACCURACY=1e-6

from Runge_Kutta import rkqs1d

#Degrees of freedom from https://arxiv.org/pdf/1609.04979.pdf
dof= np.loadtxt("data/degrees_of_freedom.dat")
dof_e = interp1d(dof[:,0],dof[:,2],fill_value=(dof[0,2],dof[-1,2]),bounds_error=False)
dof_s = interp1d(dof[:,0],dof[:,4],fill_value=(dof[0,4],dof[-1,4]),bounds_error=False)

#Energy density of the universe
def Erho(T):
    return math.pi**2/30*T**4*dof_e(T)
#Entropy
def entropy(T):
    return 2*math.pi**2/45.0*T**3*dof_s(T)
#Hubble Constant
def Hub(T):
    return math.sqrt(8.0*math.pi/3.0*Erho(T))/Mpl
#Equilibrium number density of a particle
def neq_integrand(En,m,T,g,spinfact,mu):
    try:
        return g/(2.0*math.pi**2)*En*math.sqrt(En**2-m**2)/(math.exp((En-mu)/T)+spinfact)
        #return g/(2.0*math.pi**2)*En*math.sqrt(En**2-m**2)/(math.exp((En-mu)/T)+spinfact)
    except OverflowError:
        return 0.0
    except ZeroDivisionError:
        return 0.0
#Need to add a low temperature approximation.
def neq(m,T,g,spinfact,mu):
    if m/T>25:
        #print("Adopting high-x behavior.", m, T)
        return g*(m*T/(2*math.pi))**1.5*math.exp(-m/T)
    else:
        val,err = quad(neq_integrand,m,np.inf,limit=600,epsrel=1e-6,epsabs=1e-300,args=(m,T,g,spinfact,mu,))
    if err*100 > abs(val):
        print("neq integration failure",val,err,m,T)
    if val<=0.0:
        print("neq is weird",val,err,m,T)
    return val
#Equilibrium value of Y=n/s. Normally do the calculations with spinfact=mu=0
#as they do not make noticeable contributions. g is the number of degrees of
#freedom.
def Yeq(m,T,g,spinfact,mu):
    return neq(m,T,g,spinfact,mu)/entropy(T)
def DeltaY(m,x,Deltax,g,sigma):
    hold =x*Hub(m/x)*(Yeq(m,m/x,g,0,0)-Yeq(m,m/(x+Deltax),g,0,0))/Deltax/(2*sigma(x)*neq(m,m/x,g,0,0))
    return hold

def DeltaCond(x,m,Deltax,g,sigma,cond):
    return DeltaY(m,x,Deltax,g,sigma)-cond*Yeq(m,m/x,g,0,0)

def Omega_from_Y(Y,m):
    return entropy_today*Y*m/rhocrit

#R Ratio from PDG
def format_rratio(rratio):
 for line in rratio:
     linearr=line.split()
     if len(linearr)>=14:
         try:
             yield [float(linearr[0]),float(linearr[3])]
         except ValueError:
             continue

with open('data/rratio.dat','r') as infile:
 rratio1=infile.read()
rratio_rough = rratio1.splitlines()
rratio_clean=np.array(list(format_rratio(rratio_rough)),dtype=float)
f_rratio = interp1d(rratio_clean[:,0],rratio_clean[:,1],fill_value=(rratio_clean[0,1],rratio_clean[-1,1]),bounds_error=False)
def rratio(s):
 if s<rratio_clean[0,0]:
     return 0
 else:
     return f_rratio(s)

#Momentum of outgoing particles 2 and 3 produced by at-rest decay of particle 1.
def TriangleFunc(m1,m2,m3):
    try:
        return 1.0/(2.0*m1)*math.sqrt(m1**4+m2**4+m3**4-2*(m1*m2)**2-2*(m1*m3)**2-2*(m3*m2)**2)
    except ValueError:
        print("ValueError in TriangleFunc!")
        print(m1,m2,m3)

#Width of the V
def Gamma_V_ll(mv,kappa,ml):
    if mv>2*ml:
        return 4*kappa**2*alpha_em*(2*melec**2+mv*mv)*math.sqrt(mv*mv/4-melec**2)/(6.0*mv*mv);
    else:
        return 0.0

def Gamma_V_dm_dm(mv,mx,alphaD):
    if mv>2*mx:
        return alphaD*(mv*mv-4*mx*mx)*math.sqrt(mv*mv/4.0-mx*mx)/(6.0*mv*mv)
    else:
        return 0.0

def Gamma_V(mv, mx, kappa, alphaD):
    return Gamma_V_dm_dm(mv,mx,alphaD)+Gamma_V_ll(mv,kappa,melec)+Gamma_V_ll(mv,kappa,mmuon)
    #rratio could be added here I think.

def sigma_dmdm_to_ll(alpha_D,kappa,mv,mx,ml,s):
    if s>4*ml**2:
        return -math.pi*alpha_em*alpha_D*kappa**2*(ml**2-s)*math.sqrt(s-4*ml**2)*(s-4*mx**2)**1.5/(3*s*(s/4.0-mx**2)*            ((Gamma_V(mv,mx,kappa,alpha_D)*mv)**2+(mv**2-s)**2))
    else:
        return 0.0

def sigma_dmdm(s,alpha_D,kappa,mv,mx):
    return sigma_dmdm_to_ll(alpha_D,kappa,mv,mx,melec,s)+sigma_dmdm_to_ll(alpha_D,kappa,mv,mx,mmuon,s)*(1+rratio(math.sqrt(s)))

#Thermally averaged dark matter cross section
def bessel_ratio(mx,T,s):
    if(math.sqrt(s)/T>100):
        return math.sqrt(2.0/math.pi)*mx/math.sqrt(T)/s**0.25*math.exp((2*mx-math.sqrt(s))/T)
    else:
        return kn(1,math.sqrt(s)/T)/kn(2,mx/T)**2
def sigmav_integrand(s,alpha_D,kappa,mv,mx,T):
    #print(s,alpha_D,kappa,mv,mx,T,math.sqrt(s),(s-4*mx**2),sigma_dmdm(s,alpha_D,kappa,mv,mx),bessel_ratio(mx,T,s))
    return math.sqrt(s)*(s-4*mx**2)*sigma_dmdm(s,alpha_D,kappa,mv,mx)*bessel_ratio(mx,T,s)
def sigmav(T, alpha_D,kappa,mv,mx):
    val, err=quad(sigmav_integrand,4*mx**2,np.inf,args=(alpha_D,kappa,mv,mx,T,),limit=400,epsabs=1e-300,epsrel=10**-3)
    if err*100 > val or val==0.0:
        if kappa==0 or alpha_D==0:
            return 0.0
        #print("Things going badly!",val,err,mx,T,mv)
        val, err=quad(sigmav_integrand,4*mx**2,max(10*T**2,10*mx**2),args=(alpha_D,kappa,mv,mx,T,),limit=400,epsabs=1e-300,epsrel=10**-3)
        #print("Correction Attempt",val, err,mx,T,mv)
        for n in range(10):
            val2, err2=quad(sigmav_integrand,4*mx**2,max(50*(n+1)*T**2,50*(n+1)*mx**2),args=(alpha_D,kappa,mv,mx,T,),limit=400,epsabs=1e-300,epsrel=10**-3)
            #print(n,val2,val-val2,err2)
            if abs(val2-val)<0.01*abs(val2):
                #print("correction worked!")
                val=val2
                err=err2
                break
            else:
                #print("Iterating on correction",val2,err2)
                val=val2
                err=err2
        #print(val2,err2)
        if err*100>val or val==0.0:
            print("Possible error in sigmav integration!")
            print(val,err)
            print(mv,mx,T)

    #print(val,err)
    return val/(8*mx**4*T)


def Yevolution_integrand(x,mx,sigmav):
    return entropy(mx/x)*sigmav(x)/(x*Hub(mx/x))
#sigmav is a function that only accepts x.
def Ystep(g,mx,sigmav,xstep=1e-2,Deltax=1e-4):
    Yeqset = lambda x: Yeq(mx,mx/x,g,0,0)
    neqset = lambda x: neq(mx,mx/x,g,0,0)

    #Find a point shortly before freezeout
    xstart=brentq(DeltaCond,1,100,args=(mx,Deltax,g,sigmav,Delta_Y_Condition,))
    Y = Yeqset(xstart)
    xi=xstart
    deltaxlast=xstep
    xmax=xstart+20
    dydx = lambda x, Y: -Yeqset(x)/x*neqset(x)*sigmav(x)/Hub(mx/(x))*((Y/Yeqset(x))**2-1)
    while True:
        if Y>2.5*Yeqset(xi) or xi>xmax:
            break
        if xi+xstep>xmax:
            xstep = xmax-xi

        xi,Y,deltaxlast,xstep=rkqs1d(Y, dydx(xi,Y), xi, xstep, Y_ACCURACY, dydx)
        #print(xi,Y,Yeqset(xi),deltaxlast,xstep)

    Yinf_val,Yinf_error = quad(Yevolution_integrand,xi,1000,epsabs=1e-300,epsrel=1e-2,limit=200,args=(mx,sigmav,))
    if Yinf_val < 100*Yinf_error:
        print("Error in Ystep integration")
        print(Yinf_val,Yinf_error)
        print(xi,mx)
    Yinf = 1.0/(1.0/(2.5*Yeqset(xi))+Yinf_val)
    return Yinf,xi


def Ysearch(g,alpha_D,mv,mx,tol=1e-3,xstep=1e-2,Deltax=1e-4):
    kappa = math.sqrt(relic_density_sigma/sigmav(mx/20.0,alpha_D,1.0,mv,mx)/conversion)
    print("Initial kappa estimate={}".format(str(kappa)))
    it=0
    while True:
        sig = lambda x: sigmav(mx/x,alpha_D,kappa,mv,mx)
        Y,xf = Ystep(g,mx,sig,xstep,Deltax)
        Omega = Omega_from_Y(Y,mx)
        if abs(OmegaCDM-Omega)<tol:
            print("Accepted kappa={} Omega_CDM={}".format(str(kappa),str(Omega)))
            break
        it+=1
        print("Iteration {} kappa={} Omega_CDM={}".format(str(it),str(kappa),str(Omega)))
        kappa = math.sqrt(kappa**2*Omega/OmegaCDM)
    return kappa,Omega,xf,Y


'''
#First test evolution function. Oddly, performs only a little worse than Runge-Kutta.
def Euler_Step(x,xres,dydx,Y):
    return dydx(x,xres,Y)*xres,xres

def Ystep_euler(g,mx,sigmav,xstep=1e-2,Deltax=1e-4):
    Yeqset = lambda x: Yeq(mx,mx/x,g,0,0)
    neqset = lambda x: neq(mx,mx/x,g,0,0)

    #Find a point shortly before freezeout
    xstart=brentq(DeltaCond,1,100,args=(mx,Deltax,g,sigmav,Delta_Y_Condition,))
    Y = Yeqset(xstart)
    xi=xstart
    xmax=xstart+20
    dydx = lambda x, xstep, Y: -Yeqset(x+xstep)/x*neqset(x+xstep)*sigmav(x+xstep)        /Hub(mx/(x+xstep))*((Y/Yeqset(x+xstep))**2-1)
    while True:
        if Y>2.5*Yeqset(xi) or xi>xmax:
            break
        deltay,xstep = Euler_Step(xi,xstep,dydx,Y)
        #print(xi,Y,Yeqset(xi),deltay)
        Y+=deltay
        xi+=xstep

    Yinf_val,Yinf_error = quad(Yevolution_integrand,xi,1000,epsabs=1e-300,epsrel=1e-4,limit=400,args=(mx,sigmav,))
    if Yinf_val < 100*Yinf_error:
        print("Error in Ystep integration")
        print(Yinf_val,Yinf_error)
        print(xi,mx)
    Yinf = 1.0/(1.0/(2.5*Yeqset(xi))+Yinf_val)
    return Yinf,xi+xstep



def Ysearch_euler(g,alpha_D,mv,mx,tol=1e-3,xstep=1e-2,Deltax=1e-4):
    kappa = math.sqrt(relic_density_sigma/sigmav(mx/20.0,alpha_D,1.0,mv,mx)/conversion)
    #print(kappa)
    while True:
        sig = lambda x: sigmav(mx/x,alpha_D,kappa,mv,mx)
        Y,xf = Ystep_euler(g,mx,sig,xstep,Deltax)
        Omega = Omega_from_Y(Y,mx)
        if abs(OmegaCDM-Omega)<tol:
            break
        #print(kappa,Omega)
        kappa = math.sqrt(kappa**2*Omega/OmegaCDM)
    return kappa,Omega,xf,Y
'''
from Hidden_Sec_Utilities import *

def Y_func(mv,mx,alpha_p,kappa):
    return [mv,mx,kappa**2*alpha_p*(mx/mv)**4]

def relic_table(mass_arr,alpha_D=0.5,run_name="",func=Y_func):
    mass_arr = np.array(mass_arr)

    relic_tab=[func(mv,mx,alpha_D,Ysearch(2,alpha_D,mv,mx)[0]) for mv,mx in mass_arr]

    np.savetxt(run_name+"relic_density.dat",relic_tab)

mass_arr=[[0.006,0.002],[0.03,0.01],[0.1,0.0333],[0.3,0.1],[1,.333],[3,1]]

relic_table(mass_arr,run_name="Test_Run")
#import time


#mvtest=0.006
#mxtest=0.002

#print(sigmav_integrand(2.7999e-5,0.5,1,mvtest,mxtest,2.88e-6))

#print(sigmav(2.88e-6,0.5,1.0,mvtest,mxtest))
#print(sigmav(0.01/20.0,0.5,1.0,0.03,0.01))
#print(sigmav(0.1/20.0,0.5,1.0,0.3,0.1))
#print(sigmav(1/20.0,0.5,1.0,3,1))

#start = time.time()
#print(Ysearch(2,0.5,mvtest,mxtest))
#end = time.time()
#print(end-start)

#start = time.time()
#print(Ysearch_euler(2,0.5,mvtest,mxtest))
#end = time.time()
#print(end-start)
