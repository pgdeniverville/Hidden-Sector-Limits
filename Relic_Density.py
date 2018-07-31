
# coding: utf-8

#Outline of algorithm from micrOMEGAs https://arxiv.org/pdf/1402.0787.pdf
import numpy as np
import math
import scipy.special as sc
from scipy.interpolate import interp1d
from scipy.special import kn
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.optimize import brenth
from scipy.optimize import bracket

from constants import *
#Start tracking Y_evolution when
#Delta_Y_Condition*Y_eq(x) = Y(x)-Y_eq(x)
Delta_Y_Condition=0.1
#How well to track Y during the Y evolution.
#It is not recommended that this be increased.
#Decreasing this value will improve accuracy at
#the cost of runtime. Further increasing the ac
Y_ACCURACY=1e-6
#This determies the precision to which epsilon will
#be determined.
EPSILON_TOLERANCE=1e-2
X_LARGE=1000

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
        val,err = quad(neq_integrand,m,np.inf,limit=1200,epsrel=1e-3,epsabs=1e-300,args=(m,T,g,spinfact,mu,))
    #if err*100 > abs(val):
    #    print("neq integration failure",val,err,m,T)
    if val<=0.0:
        print("neq is weird",val,err,m,T)
    return val
#Equilibrium value of Y=n/s. Normally do the calculations with spinfact=mu=0
#as they do not make noticeable contributions. g is the number of degrees of
#freedom.
def Yeq(m,T,g,spinfact,mu):
    return neq(m,T,g,spinfact,mu)/entropy(T)
def DeltaY(m,x,Deltax,g,sigma):
    #print(m,x,Deltax,g,sigma(x))
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
    f_rratio = interp1d(rratio_clean[:,0],rratio_clean[:,1],fill_value=(0,rratio_clean[-1,1]),bounds_error=True)

def rratio(s):
    if s<rratio_clean[0,0]:
        return 0
    elif s>rratio_clean[-1,0]:
        return rratio_clean[-1,1]
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
        return -math.pi*alpha_em*alpha_D*kappa**2*(ml**2-s)*math.sqrt(s-4*ml**2)*(s-4*mx**2)**1.5/(3*s*(s/4.0-mx**2)*((Gamma_V(mv,mx,kappa,alpha_D)*mv)**2+(mv**2-s)**2))
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

#Omega returns the difference between the relic density
#produced by
#sigmav should be a funcion sigmav(x,epsilon).
def Omega_Cond(kappa,g,mx,sigk,xstep=1e-2,Delta=1e-4):
    Omega=Ystep(kappa,g,mx,sigk,xstep,Delta)
    print("Brenth search epsilon={} Omega={}".format(str(kappa),str(Omega)))
    return OmegaCDM-Omega

def Omega_Cond_bracket(kappa,g,mx,sigk,xstep=1e-2,Delta=1e-4):
    Omega=Ystep(kappa,g,mx,sigk,xstep,Delta)
    print("Bracket search epsilon={} Omega={}".format(str(kappa),str(Omega)))
    return OmegaCDM-Omega

FACTOR=0.5

def Find_Crossing(k1,k2,g,mx,sigk):
    OC1 = Omega_Cond_bracket(math.exp(k1),g,mx,sigk)
    OC2 = Omega_Cond_bracket(math.exp(k2),g,mx,sigk)
    for n in range(20):
        if(OC1*OC2<0):
            return math.exp(k1),math.exp(k2),OC1,OC2
        if(abs(OC1)<abs(OC2)):
            k1=k1+FACTOR*(k1-k2)
            OC1=Omega_Cond_bracket(math.exp(k1),g,mx,sigk)
        else:
            k2=k2+FACTOR*(k2-k1)
            OC2=Omega_Cond_bracket(math.exp(k2),g,mx,sigk)

#Slower than a more aggressive implementation, but also guaranteed to find a solution.
def Ysearch(g,alpha_D,mv,mx,tol=5e-3,xstep=1e-2,Deltax=1e-4):
    kappa = math.sqrt(relic_density_sigma/sigmav(mx/20.0,alpha_D,1.0,mv,mx)/conversion)
    sigk = lambda x, kap: sigmav(mx/x,alpha_D,kap,mv,mx)
    print("Initial epsilon estimate={} for mv={} mx={}".format(str(kappa),masstext(mv),masstext(mx)))
    #brenth seems faster than brentq by approximately 30%
    #ea,eb,ec,fa,fb,fc,calls=bracket(Omega_Cond_bracket,kappa,kappa*2,args=(g,mx,sigk,))
    ea,ec,fa,fc = Find_Crossing(math.log(kappa),math.log(kappa*2),g,mx,sigk)
    kappa_final= brenth(Omega_Cond,ea,ec,args=(g,mx,sigk,),rtol=EPSILON_TOLERANCE)
    print("Accepted epsilon={}".format(str(kappa_final)))
    return kappa_final


def Yevolution_integrand(x,mx,sigmav):
    return entropy(mx/x)*sigmav(x)/(x*Hub(mx/x))
#A fairly general algorithm for determining the present day
#relic density of single component dark matter.
def Ystep(kappa, g,mx,sigk,xstep=1e-2,Deltax=1e-4):
    Yeqset = lambda x: Yeq(mx,mx/x,g,0,0)
    neqset = lambda x: neq(mx,mx/x,g,0,0)

    sigx = lambda x: sigk(x,kappa)

    #Find a point shortly before freezeout
    xstart=brentq(DeltaCond,1,100,args=(mx,Deltax,g,sigx,Delta_Y_Condition,))
    Y = Yeqset(xstart)
    xi=xstart
    deltaxlast=xstep
    xmax=xstart+20
    dydx = lambda x, Y: -Yeqset(x)/x*neqset(x)*sigx(x)/Hub(mx/(x))*((Y/Yeqset(x))**2-1)
    while True:
        if Y>2.5*Yeqset(xi) or xi>xmax:
            break
        if xi+xstep>xmax:
            xstep = xmax-xi
        #Using Runge-Kutta from Numerical Recipes for C.
        xi,Y,deltaxlast,xstep=rkqs1d(Y, dydx(xi,Y), xi, xstep, Y_ACCURACY, dydx)
        #print(xi,Y,Yeqset(xi),deltaxlast,xstep)

    #X_LARGE is normally set to ~1000, but as long as it is very large its exact value
    #is unimportant. Epsabs is set very small because I only care about epsrel.
    Yinf_val,Yinf_error = quad(Yevolution_integrand,xi,X_LARGE,epsabs=1e-300,epsrel=1e-2,limit=400,args=(mx,sigx,))
    if Yinf_val < 100*Yinf_error:
        print("Error in Ystep integration")
        print(Yinf_val,Yinf_error)
        print(xi,mx)
    Yinf = 1.0/(1.0/(2.5*Yeqset(xi))+Yinf_val)
    return Omega_from_Y(Yinf,mx)


from Hidden_Sec_Utilities import *

def Y_func(mv,mx,alpha_p,kappa):
    return [mv,mx,kappa**2*alpha_p*(mx/mv)**4]

def relic_table(mass_arr,alpha_D=0.5,run_name="",func=Y_func):
    mass_arr = np.array(mass_arr)

    relic_tab=[func(mv,mx,alpha_D,Ysearch(2,alpha_D,mv,mx)) for mv,mx in mass_arr]

    np.savetxt(run_name+"relic_density.dat",relic_tab)

#mv_arr=[2]+[10*mv+10 for mv in range(19)]+[25*mv+200 for mv in range(30)]+[100*mv+1000 for mv in range(20)]

mx_arr=[1,5,10,30,50,70,90,95,97,100,102,103,105,107,110,150,200,250,300]+[mx for mx in range(305,605,5)]+[mx for mx in range(625,1001,25)]

#mass_arr=[[mv/1000.0,mv/3000.0] for mv in mv_arr]
mass_arr=[[3*mx/1000.0,mx/1000.0] for mx in mx_arr]
#mass_arr=[[0.006,0.002],[0.01,0.01/3.0],[0.03,0.01],[0.1,0.0333],[0.3,0.1],[1,.333],[3,1]]
#mass_ar=[[0.01,0.01/3.0]]
#mass_arr=[[2,0.65]]
#sigk = lambda x,kappa: sigmav(0.001/x,0.5,kappa,0.003,0.001)
#print(Omega_Cond_bracket(3e-6,2,0.001,sigk))

make_sure_path_exists("output/")
import time
start = time.time()
relic_table(mass_arr,alpha_D=0.05,run_name="output/y3_0.05")
end = time.time()
print("Total Runtime={}".format(str(end-start)))

