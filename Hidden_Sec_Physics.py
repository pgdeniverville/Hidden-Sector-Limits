import numpy as np
import math
from scipy.special import kn
from scipy.interpolate import interp1d
from scipy.interpolate import SmoothBivariateSpline
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.interpolate import griddata
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

#Functions for returning arrays of V mass, DM mass and some function f(mv,mx,alpha_p,kappa)
#Returns an array of the V mass, DM mass and kappa^4*alpha_p
def k4al(mv,mx,alpha_p,kappa):
    return [mv,mx,kappa**4*alpha_p]

def kappa(mv,mx,alpha_p,kappa):
    return [mv,mx,kappa]

def Y_func(mv,mx,alpha_p,kappa):
    return [mv,mx,kappa**2*alpha_p*(mx/mv)**4]

def alphab(mv,mx,alpha_b,kappa):
    return [mv,mx,alpha_b]

#Reduced Mass
def reduced_mass(m1,m2):
    return m1*m2/(m1+m2)

#Scattering cross section for kinetic mixing
#These are only applicable for mV>>alpha_em^2*m_e^2 (see https://arxiv.org/pdf/1108.5383v3.pdf)
def sigman(mv,mx,kappa,alpha_p):
    return 16*math.pi*kappa**2*alpha_em*alpha_p*reduced_mass(mp,mx)**2/mv**4*0.25
def sigman_to_kappa(sigma,mv,mx,alpha_p):
    return math.sqrt(sigma/conversion/100**2/(16*math.pi*alpha_em*alpha_p*reduced_mass(mp,mx)**2/mv**4*0.25))

def sigman_to_kappa2(sigma,mv,mx,alpha_p):
    return math.sqrt(sigma/conversion/100**2/sigman(mv,mx,1,alpha_p))

def sigmae(mv,mx,kappa,alpha_p):
    return 16*math.pi*kappa**2*alpha_em*alpha_p*reduced_mass(melec,mx)**2/(mv**2+alpha_em**2*melec**2)**2
def sigmae_to_kappa(sigma,mv,mx,alpha_p):
    return math.sqrt(sigma/conversion/100**2/(16*math.pi*alpha_em*alpha_p*reduced_mass(melec,mx)**2/(mv**2+alpha_em**2*melec**2)**2))
    #return math.sqrt(sigma/conversion/100**2/sigmae(mv,mx,1,alpha_p))

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
        return 8.0*math.pi/3*alphap*alpha_em*kappa**2/((4*Epart(dm_beta,mx)**2-mv**2)**2+mv**2*GammaV(alphap,kappa,mv,mx)**2)*(2*Epart(dm_beta,mx)**2+mlepton**2)*dm_beta**2*math.sqrt(1-mlepton**2/Epart(dm_beta,mx)**2)
    return 0
def sigma_annihilation_dm(kappa,alphap,mv,mx):
    return sigma_ann_lepton(alphap,kappa,mv,mx,melec)+sigma_ann_lepton(alphap,kappa,mv,mx,mmuon)*(1+rratio(2*Epart(dm_beta,mx)))

def gen_relic_dm(mv,mx,alpha_p):
    g=lambda kappa: sigma_annihilation_dm(kappa,alpha_p,mv,mx)*conversion-relic_density_sigma
    try:
        return brentq(g,0,10)
    except ValueError:
        print("Value error encountered for mv,mx,sigma_ann",mv,mx,sigma_annihilation_dm(1e-3,alpha_p,mv,mx))
        return 1000

#This is much faster, but could be tripped up if sigma_annihilation_dm does not
#scale as kappa**2.
def gen_relic_dm_fast(mv,mx,alpha_p):
    return math.sqrt(relic_density_sigma/(sigma_annihilation_dm(1,alpha_p,mv,mx)*conversion))

#######################
#Muon and Electron g-2#
#######################
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



################
#Monojet Limits#
################

#From aritz e-mail 2012. Should find a citation for this.
def monojet_limit():
    return 0.02/math.sqrt(4*math.pi*alpha_em)
#1112.5457
def monojet_limit_baryonic():
    return 9*0.021**2/(4*math.pi)


##########################
#K^+ --> pi^+ + invisible#
##########################

#Formulae from 0808.2459 and data from E949 0903.0030
kpip_invis_dat_1 = np.loadtxt("data/kpipinvis1.dat")
kpip_invis_dat_2 = np.loadtxt("data/kpipinvis2.dat")
def W2(mv):
    return 1e-12*(3+6*mv**2/mkaon**2)
def Gamma_K_Vpi(mv,kappa):
    return alpha_em*kappa**2*mv**2*W2(mv)*4*lambda_m(mkaon,mpip,mv)**3/(2**9*math.pi**4*mkaon**4)
def Br_K_Vpi(mv,kappa):
    return Gamma_K_Vpi(mv,kappa)*tauK/hbar
def gen_K_Vpi_lim(arr):
    return np.array([[mv, math.sqrt(kdat/Br_K_Vpi(mv,1))] for mv,kdat in arr])

#Baryonic
def Br_K_VpiB(mv,alpha_B):
    return 1.4e-3*alpha_B*(mv/0.1)**2
def gen_K_VpiB_lim(arr):
    return np.array([[mv, kdat/Br_K_VpiB(mv,1)] for mv,kdat in arr])
def gen_K_VpiB_lim_conservative(arr):
    return np.array([[mv, 10*kdat/Br_K_VpiB(mv,1)] for mv,kdat in arr])


########################
#Electroweak Fit limits#
########################

#Model Independent Bounds on Kinetic Mixing - Anson Hook, Eder Izaguirre, Jay G.Wacker. 1006.0973
zprimedat=np.loadtxt("data/zprime.dat")

############
#Babar line#
############

#Possible sensitivity from repurposed analysis of http://arxiv.org/abs/arXiv:0808.0017
#Full analysis in http://arxiv.org/pdf/1309.5084v2.pdf

babar_dat=np.loadtxt("data/babar.dat")
babar_interp = interp1d(babar_dat[:,0],babar_dat[:,1])
babar_max = max(babar_dat[:,0])

#Sensitivity from BaBar analysis https://arxiv.org/abs/1702.03327
#Should maybe add a branching ratio to this, but normally Br(V->invis)~1 for our parameters
babar2017_dat = np.loadtxt("data/babar2017_formatted.dat")
babar2017_interp = interp1d(babar2017_dat[:,0],babar2017_dat[:,1])

babar2017_min = min(babar2017_dat[:,0])
babar2017_max = max(babar2017_dat[:,0])
# In[23]:

def babar_func(mv,mx,alpha_p,fill_value=1000):
    if mv < 0.2:
        term = babar_interp(0.2)
    elif mv > babar_max:
        term= fill_value
    else:
        term = babar_interp(mv)
    if 2*mx>mv:
        term = 1.0/math.sqrt(alpha_p)*term
    return term

def babar_func2017(mv,mx,alpha_p,fill_value=1000):
    if mv <= babar2017_min:
        term = babar2017_interp(babar2017_min)
    elif mv >= babar2017_max:
        term = fill_value
    else:
        term = babar2017_interp(mv)
    if 2*mx>mv:
        term = 1.0/math.sqrt(alpha_p)*term
    return term

#################################
#Baryonic Limits from 1705.06726#
#################################
#These largely eliminate the non-anomaly-free version of
#the model. I expect an anomaly-free version will need to
#be found for the model to be viable.
anomalon_1705_06726_dat = np.loadtxt("data/Anomalon_formatted.dat")
anomalon_1705_06726_dat[:,1]=anomalon_1705_06726_dat[:,1]**2/4.0/math.pi
BtoKX_1705_06726_dat = np.loadtxt("data/1705.06726.BtoKX_formatted.dat")
BtoKX_1705_06726_dat[:,1]=BtoKX_1705_06726_dat[:,1]**2/4.0/math.pi
ZtogammaX_1705_06726_dat = np.loadtxt("data/1705.06726.ZtogammaX_formatted.dat")
ZtogammaX_1705_06726_dat[:,1]=ZtogammaX_1705_06726_dat[:,1]**2/4.0/math.pi
KtopiX_1705_06726_dat = np.loadtxt("data/1705.06726.KtopiX_formatted.dat")
KtopiX_1705_06726_dat[:,1]=KtopiX_1705_06726_dat[:,1]**2/4.0/math.pi

#############################
#Baryonic Neutron Scattering#
#############################

def neutron_scatter(mv):
    return 3.4e-11*(mv/0.001)**4

#########################
#Limits from rare decays#
#########################
#Rare Decay Limits
def Br_Jpsi_to_V(mv,mx,alpha_p,kappa):
    return Brj_psi_to_ee*kappa**2*alpha_p/(4*alpha_em)*mj_psi**4/((mj_psi**2-mv**2)**2+mv**2*GammaV(alpha_p,kappa,mv,mx)**2)
def rarelimit(mv,mx,alpha_p):
    g= lambda kappa: Br_Jpsi_to_V(mv,mx,alpha_p,kappa) - Brj_psi_to_invis
    return brentq(g,0,1)
#Baryonic
def Br_Jpsi_to_VB(mv,mx,alpha_B,kappa):
    return Brj_psi_to_ee/9.0*alpha_B**2/(4*alpha_em**2)*mj_psi**4/((mj_psi**2-mv**2)**2+mv**2*GammaVB(alpha_B,kappa,mv,mx)**2)
def rarelimitB(mv,mx,kappa):
    if mv == mj_psi:
        return 0
    g= lambda alpha_B: (Br_Jpsi_to_VB(mv,mx,alpha_B,kappa) - Brj_psi_to_invis)
    return brentq(g,0,1)

#######################
#Invisible Pion Limits#
#######################

#http://inspirehep.net/record/333625?ln=en, Atiya 1992
invispiondat = np.loadtxt("data/invis_pion.dat")
#Note, this curve has kappa dependence. It is currently assuming kappa=0
invispionbaryonicdat = np.loadtxt("data/invis_pion_baryonic.dat")

######
#NA64#
######

#https://arxiv.org/abs/1610.02988
#These limits are only valid in the case that V->\chi\bar\chi is dominant decay channel
#NA64dat = np.loadtxt("data/NA64_formatted.dat")

#https://arxiv.org/abs/1710.00971
NA64dat = np.loadtxt("data/NA64_2017_formatted.dat")
#Projections from Physics Beyond Colliders Working Group meeting
#NA64_2016dat = np.loadtxt("data/NA64_2016_formatted.dat")
#NA64_2017dat = np.loadtxt("data/NA64_2017_formatted.dat")
#NA64_2018dat = np.loadtxt("data/NA64_2018_formatted.dat")

######
#E137#
######
#E137 has to be handled separately. These limits are provided by Brian
#Batell. See https://arxiv.org/abs/1406.2698.
E137tab = np.loadtxt("data/E137-kappa4XalphaD-mV-mX.csv",delimiter=',')

######
#LSND#
######
#See arXiv:1107.4580 and arXiv:1411.1055.
LSNDtab = np.loadtxt("data/lsnd.dat",delimiter='\t')

#Direct_Detection_Limits
#1105.5191
xenon10_dat = np.loadtxt("data/xenon10.dat")
#1207.5988
xenon100_dat1 = np.loadtxt("data/xenon100_1.dat")
xenon100_dat2 = np.loadtxt("data/xenon100_2.dat")
#1105.5191
damic_dat=np.loadtxt("data/damic.dat")
#arXiv:1509.01515
cressII2015_dat_unscaled = np.loadtxt("data/cressII2015.dat")
cressII2015_dat = zip(cressII2015_dat_unscaled[:,0],cressII2015_dat_unscaled[:,1]*1e-36)
#arXiv:1402.7137
with open("data/SuperCDMS.dat") as infile:
    scdms1=infile.read()
    scdms2=[line.split() for line in scdms1.split(';')]
    SuperCDMS_dat=[[float(x),float(y)] for x,y in scdms2]
#arXiv:1509.02448
CDMSlite_dat = np.loadtxt("data/cdmslite2015.dat")
#1512.03506
LUX_dat_unscaled = np.loadtxt("data/lux2015.dat")
LUX_dat = zip(LUX_dat_unscaled[:,0],LUX_dat_unscaled[:,1]*1e-45)
Direct_Det_Tab =[xenon10_dat,xenon100_dat1,xenon100_dat2,damic_dat,cressII2015_dat,SuperCDMS_dat,CDMSlite_dat,LUX_dat]

Direct_Det_Func=[interp1d(np.array(tab)[:,0],np.array(tab)[:,1],bounds_error=False,fill_value=1e-25) for tab in Direct_Det_Tab]

def Direct_Det(mx):
    return min([func(mx) for func in Direct_Det_Func])

#arxiv:.pdf
#See also https://arxiv.org/pdf/1505.00011.pdf for comparison
xenon10e_dat = np.loadtxt("data/xenon10e_2017_formatted.csv",delimiter=",")
xenon10efunc=interp1d(xenon10e_dat[:,0],xenon10e_dat[:,1],bounds_error=False,fill_value=1e-15)
xenon100e_dat = np.loadtxt("data/xenon100e_2017_formatted.csv",delimiter=",")
xenon100efunc=interp1d(xenon100e_dat[:,0],xenon100e_dat[:,1],bounds_error=False,fill_value=1e-15)
#arxiv:1206.2644v1.pdf
#xenon10e_dat = np.loadtxt("data/xenon10e_formatted.csv",delimiter=",")
#xenon10e_dat = np.loadtxt("data/xenon10e_2017_formatted.csv",delimiter=",")
#1804.10697
SCDMSe_dat = np.loadtxt("data/CDMS_electron_2018_formatted.dat",delimiter=" ")
SCDMSefunc=interp1d(SCDMSe_dat[:,0],SCDMSe_dat[:,1],bounds_error=False,fill_value=1e-15)
#1804.00088
SENSEIe_dat = np.loadtxt("data/SENSEI2018_formatted.dat",delimiter=" ")
SENSEIefunc=interp1d(SENSEIe_dat[:,0],SENSEIe_dat[:,1],bounds_error=False,fill_value=1e-15)
