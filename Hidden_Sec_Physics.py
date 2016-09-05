import numpy as np
import math
from scipy.special import kn
from scipy.interpolate import interp1d
from scipy.interpolate import SmoothBivariateSpline
from scipy.optimize import brentq
from scipy.integrate import quad
#import matplotlib.pyplot as plt

from Hidden_Sec_Utilities import *

#Units
gev=1;mev=1e-3*gev;kev=1e-6*gev;

#masses
mp=0.938272046*gev;melec=511*kev;mmuon=105.658*mev;mpi=134.9767*mev;mpip=139.57018*mev;mkaon=0.493667*gev;mj_psi=3.097*gev;

#Widths, Branching Ratios, Constants
tauK=1.23e-8;alpha_em=1.0/137.035999139;Brj_psi_to_ee=0.0594;Brj_psi_to_mumu=0.0593;Brj_psi_to_invis=7e-4;
hbar=float(1.054*1e-34/(1.6e-19)/(1e9));speed_of_light=3e8;conversion=hbar**2*speed_of_light**2;

#Relid density cross section for rough estimates of relic density
relic_density_sigma=1e-40

#Reduced Mass
def reduced_mass(m1,m2):
    return m1*m2/(m1+m2)

#Scattering cross section for kinetic mixing
def sigman(mv,mx,kappa,alpha_p):
    return 16*math.pi*kappa**2*alpha_em*alpha_p*reduced_mass(mp,mx)**2/mv**4*0.25
def sigman_to_kappa(sigma,mv,mx,alpha_p):
    return math.sqrt(sigma/conversion/100**2/(16*math.pi*alpha_em*alpha_p*reduced_mass(mp,mx)**2/mv**4*0.25))

def sigman_to_kappa2(sigma,mv,mx,alpha_p):
    return math.sqrt(sigma/conversion/100**2/sigman(mv,mx,1,alpha_p))

#Scattering cross section for baryonic with kappa=0
def sigman_B(mv,mx,alpha_b):
    return 16*math.pi*alpha_b**2*reduced_mass(mp,mx)**2/mv**4
def sigman_to_alpha_b(sigman,mv,mx):
    return math.sqrt(sigman/conversion/100**2/sigman_B(mv,mx,1))


#RRatio
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
f_rratio = interp1d(rratio_clean[:,0],rratio_clean[:,1])
def rratio(s):
    if s<rratio_clean[0,0]:
        return 0
    else:
        return f_rratio(s)


#This returns the momentum of particles with masses m2 and m3 produced by the decay of a
#particle at rest with mass m1
def lambda_m(m1,m2,m3):
    return 1.0/(2*m1)*math.sqrt(m1**4+m2**4+m3**4-2*m1**2*m2**2-2*m3**2*m2**2-2*m1**2*m3**2)

#Relativistic gamma
def gamma(beta):
    return 1.0/math.sqrt(1-beta**2)
def Epart(beta, m):
    return m*gamma(beta)
def GammaV(alpha_p, kappa, mv, mx):
    term = 0;
    if mv>2*mx:
        term += alpha_p*(mv*mv-4*mx*mx)*math.sqrt(mv*mv/4.0-mx*mx)
    if mv>2*melec:
        term += 4*pow(kappa,2)*alpha_em*(2*pow(melec,2)+mv*mv)*math.sqrt(mv*mv/4.0-pow(melec,2))
    if mv>2*mmuon:
        term += 4*pow(kappa,2)*alpha_em*(2*pow(mmuon,2)+mv*mv)*math.sqrt(mv*mv/4.0-pow(mmuon,2))*(1+rratio(2*Epart(dm_beta,mx)))
    return 1.0/(6.0*mv*mv)*(term)
#Only includes V->DM+DM at the moment
def GammaVB(alpha_B, kappa, mv, mx):
    return 2.0/3.0*alpha_B*lambda_m(mv,mx,mx)**3/mv**2


#Rough relic density stuff!
dm_beta=0.3
def sigma_ann_lepton(alphap,kappa,mv,mx,mlepton):
    if Epart(dm_beta,mx)>mlepton:
        return 8.0*math.pi/3*alphap*alpha_em*kappa**2/((4*Epart(dm_beta,mx)**2-mv**2)**2+mv**2*GammaV(alphap,kappa,mv,mx)**2)*    (2*Epart(dm_beta,mx)**2+mlepton**2)*dm_beta**2*math.sqrt(1-mlepton**2/Epart(dm_beta,mx)**2)
    return 0
def sigma_annihilation_dm(kappa,alphap,mv,mx):
    return sigma_ann_lepton(alphap,kappa,mv,mx,melec)+sigma_ann_lepton(alphap,kappa,mv,mx,mmuon)*(1+rratio(2*Epart(dm_beta,mx)))

def gen_relic_dm(mv,mx,alpha_p):
    g=lambda kappa: sigma_annihilation_dm(kappa,alpha_p,mv,mx)*conversion-relic_density_sigma
    try:
        return brentq(g,0,1)
    except ValueError:
        print(mv," ",mx," ",sigma_annihilation_dm(1e-3,alpha_p,mv,mx))
        return 1



#Muon and Electron g-2
def al_int(z,mv,ml):
    return 2*ml**2*z*(1-z)**2/(ml**2*(1-z)**2+mv**2*z)
def al(mv,mlepton):
    return alpha_em/(2*math.pi)*quad(al_int,0,1,args=(mv,mlepton))[0]
def kappa_muon_lim(mv):
    return math.sqrt(7.4e-9/al(mv,mmuon))
def kappa_fav_low(mv):
    return math.sqrt(1.3e-9/al(mv,mmuon))
def kappa_fav_high(mv):
    return math.sqrt(4.8e-9/al(mv,mmuon))
def Delta_a_electron(mv,kappa):
    return alpha_em/(3*math.pi)*kappa**2*melec**2/mv**2*(1-(7*alpha_em**2-alpha_em/(4*math.pi)))
#Constraints on electron g - 2 from Motoi Endo, Koichi Hamaguchi, and Go Mishima's paper 1209.2558
def kappa_electron_lim(mv):
    g=lambda kappa: Delta_a_electron(mv,kappa)-(-1.06+2*0.82)*1e-12
    return brentq(g,0,1)

