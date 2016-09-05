
# coding: utf-8

# In[1]:

import numpy as np
import math
from scipy.special import kn
from scipy.interpolate import interp1d
from scipy.interpolate import SmoothBivariateSpline
from scipy.optimize import brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt


# In[2]:

gev=1;mev=1e-3*gev;kev=1e-6*gev;


# In[3]:

mp=0.938272046*gev;melec=511*kev;mmuon=105.658*mev;mpi=134.9767*mev;mpip=139.57018*mev;mkaon=0.493667*gev;mj_psi=3.097*gev;


# In[4]:

tauK=1.23e-8;alpha_em=1.0/137.035999139;Brj_psi_to_ee=0.0594;Brj_psi_to_mumu=0.0593;Brj_psi_to_invis=7e-4;


# In[5]:

hbar=float(1.054*1e-34/(1.6e-19)/(1e9));speed_of_light=3e8;conversion=hbar**2*speed_of_light**2;


# In[6]:

relic_density_sigma=1e-40


# In[7]:

#Checks that a string is actually a number.
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False    


# In[8]:

#Generate a string from a mass. Only set up for keV and GeV right now.
def masstext(m):
    mstr=""
    if m>=gev:
        if m==1.0 or m==2.0:
            mstr=mstr+str(int(m))
        else:
            mstr=mstr+str(m)
        mstr=mstr+"gev"
    else:
        if m*1000%1 == 0.0:
            mstr=mstr+str(int(m*1000))+"mev"
        else:
            mstr=mstr+str(m*1000)+"mev"
    return mstr


# In[9]:

def reduced_mass(m1,m2):
    return m1*m2/(m1+m2)
def sigman(mv,mx,kappa,alpha_p):
    return 16*math.pi*kappa**2*alpha_em*alpha_p*reduced_mass(mp,mx)**2/mv**4*0.25
def sigman_to_kappa(sigma,mv,mx,alpha_p):
    return math.sqrt(sigma/conversion/100**2/(16*math.pi*alpha_em*alpha_p*reduced_mass(mp,mx)**2/mv**4*0.25))

def sigman_to_kappa2(sigma,mv,mx,alpha_p):
    return math.sqrt(sigma/conversion/100**2/sigman(mv,mx,1,alpha_p))

def sigman_B(mv,mx,alpha_b):
    return 16*math.pi*alpha_b**2*reduced_mass(mp,mx)**2/mv**4
def sigman_to_alpha_b(sigman,mv,mx):
    return math.sqrt(sigman/conversion/100**2/sigman_B(mv,mx,1))


# In[10]:

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


# In[11]:

#This returns the momentum of particles with masses m2 and m3 produced by the decay of a
#particle at rest with mass m1
def lambda_m(m1,m2,m3):
    return 1.0/(2*m1)*math.sqrt(m1**4+m2**4+m3**4-2*m1**2*m2**2-2*m3**2*m2**2-2*m1**2*m3**2)
#Reduced mass
def mu_r(m1,m2):
    return m1*m2/(m1+m2)
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


# In[12]:

#Dark Matter Relic Density
dm_beta=0.3
def sigma_ann_lepton(alphap,kappa,mv,mx,mlepton):
    if Epart(dm_beta,mx)>mlepton:
        return 8.0*math.pi/3*alphap*alpha_em*kappa**2/((4*Epart(dm_beta,mx)**2-mv**2)**2+mv**2*GammaV(alphap,kappa,mv,mx)**2)*    (2*Epart(dm_beta,mx)**2+mlepton**2)*dm_beta**2*math.sqrt(1-mlepton**2/Epart(dm_beta,mx)**2)
    return 0
def sigma_annihilation_dm(kappa,alphap,mv,mx):
    return sigma_ann_lepton(alphap,kappa,mv,mx,melec)+sigma_ann_lepton(alphap,kappa,mv,mx,mmuon)*(1+rratio(2*Epart(dm_beta,mx)))

def gen_relic_dm(mv,mx,alpha_p):
    g=lambda kappa: sigma_annihilation_dm(kappa,alpha_p,mv,mx)*conversion-relic_density_sigma
    return brentq(g,0,1)

# def sigma_ann_lepton(alphap,kappa,mv,mx,mlepton,p):
#     if math.sqrt(p**2+mx**2)>mlepton:
#         return 8.0*math.pi/3.0*alpha_em*alphap*kappa**2/((4*(mx**2+p**2)-mv**2)**2+mv**2*GammaV(alphap,kappa,mv,mx)**2)*\
#     (2*(mx**2+p**2)+mlepton**2)*p/math.sqrt(p**2+mx**2)*math.sqrt(1-mlepton**2/(p**2+mx**2))
#     else:
#         return 0
    
# def sigma_annihilation_dm(kappa,alphap,mv,mx,p):
#     return sigma_ann_lepton(alphap,kappa,mv,mx,melec,p)+sigma_ann_lepton(alphap,kappa,mv,mx,mmuon,p)*(1+rratio(2*math.sqrt(p**2+mx**2)))

# def sigmav_ann_arg(p,kappa,alphap,mv,mx,T):
#     return p**2*sigma_annihilation_dm(kappa,alphap,mv,mx,p)*8*math.sqrt((mx*p)**2+p**4)*kn(1,math.sqrt(4*(p**2+mx**2))/T)

# def sigmav_annihilation_dm(kappa,alphap,mv,mx,T):
#     return quad(sigmav_ann_arg,0.0,10*mx,args=(kappa,alphap,mv,mx,T),epsrel=1e-3,epsabs=0,limit=200)[0]/(mx**4*T*kn(2,mx/T)**2)

# def gen_relic_dm_slow(mv,mx,alpha_p):
#     g=lambda kappa: sigmav_annihilation_dm(kappa,alpha_p,mv,mx,mx/20.0)*conversion-relic_density_sigma
#     return brentq(g,0,1)

# def gen_relic_dm(mv,mx,alpha_p):
#     return math.sqrt(relic_density_sigma/(sigmav_annihilation_dm(1,alpha_p,mv,mx,mx/20.0)*conversion))


# In[13]:

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


# In[14]:

#Direct_Detection_Limits
#1105.5191
xenon10_dat = np.loadtxt("data/xenon10.dat")
#1207.5988
xenon100_dat1 = np.loadtxt("data/xenon100_1.dat")
xenon100_dat2 = np.loadtxt("data/xenon100_2.dat")
#Where from?
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


# In[15]:

Direct_Det_Func=[interp1d(np.array(tab)[:,0],np.array(tab)[:,1],bounds_error=False,fill_value=1e-25) for tab in Direct_Det_Tab]
def Direct_Det_Best(x):
    return min([f(x) for f in Direct_Det_Func])


# In[16]:

def Direct_Det(mx):
    return min([func(mx) for func in Direct_Det_Func])


# In[17]:

#From aritz e-mail 2012. Should find a citation for this.
def monojet_limit():
    return 0.02/math.sqrt(4*math.pi*alpha_em)
#1112.5457
def monojet_limit_baryonic():
    return 9*0.021**2/(4*math.pi)


# In[18]:

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


# In[19]:

#Invisible Pion Limits
#http://inspirehep.net/record/333625?ln=en, Atiya 1992
invispiondat = np.loadtxt("data/invis_pion.dat")
#Note, this curve has kappa dependence. It is currently assuming kappa=0
invispionbaryonicdat = np.loadtxt("data/invis_pion_baryonic.dat")


# In[20]:

#K^+ --> pi^+ + invisible
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


# In[21]:

#Model Independent Bounds on Kinetic Mixing - Anson Hook, Eder Izaguirre, Jay G.Wacker. 1006.0973
zprimedat=np.loadtxt("data/zprime.dat")


# In[22]:

#Possible sensitivity from repurposed analysis of http://arxiv.org/abs/arXiv:0808.0017
babar_dat=np.loadtxt("data/babar.dat")
babar_interp = interp1d(babar_dat[:,0],babar_dat[:,1])


# In[23]:

def babar_func(mv,mx,alpha_p):
    if mv < 0.2:
        term = babar_interp(0.2)
    else:
        term = babar_interp(mv)
    if 2*mx>mv:
        term = 1.0/sqrt(alpha_p)*term
    return term


# In[24]:

#Baryonic Neutron Scattering
def neutron_scatter(mv):
    return 3.4e-11*(mv/0.001)**4


# In[48]:

#E137 has to be handled separately.
#E137tab = np.loadtxt("data/E137-kappa4XalphaD-mV-mX.csv",delimiter=',')
#E137func = SmoothBivariateSpline(E137tab[:,0],E137tab[:,1],E137tab[:,2])


# In[49]:

# #Produce Tables
# def tab_prod(mass_arr,alpha_p,run_name):
#     #direct_det = [[mv, mx, Direct_Det(mx)] for mv,mx in mass_arr]
#     #np.savetxt(directory_out+"direct_det.dat",direct_det)
#     relic_tab=[[mv/1000.0,mx/1000.0,gen_relic_dm(mv/1000.0,mx/1000.0,alpha_p)] for mv,mx in mass_arr]
#     babar_tab=[[mv/1000.0,mx/1000.0,babar_func(mv/1000.0,mx/1000.0,alpha_p)] for mv,mx in mass_arr]
#     rare_tab=[[mv/1000.0,mx/1000.0,rarelimit(mv/1000.0,mx/1000.0,alpha_p)] for mv,mx in mass_arr]
#     monojet_tab=[[mv/1000.0,mx/1000.0,monojet_limit()] for mv,mx in mass_arr]
#     g_minus_2_tab=[[mv/1000.0,mx/1000.0,min(kappa_muon_lim(mv/1000.0),kappa_electron_lim(mv/1000.0))] for mv,mx in mass_arr]
#     g_muon_fav_tab=[[mv/1000.0,mx/1000.0,kappa_fav_low(mv/1000.0),kappa_fav_high(mv/1000.0)] for mv,mx in mass_arr]
#     g_K_Vpi_tab_1=gen_K_Vpi_lim(kpip_invis_dat_1)
#     g_K_Vpi_tab_2=gen_K_Vpi_lim(kpip_invis_dat_2)
#     np.savetxt(run_name+"babar.dat",babar_tab)
#     np.savetxt(run_name+"relic_density.dat",relic_tab)
#     np.savetxt(run_name+"rare_decay.dat",rare_tab)
#     np.savetxt(run_name+"monojet.dat",monojet_tab)
#     np.savetxt(run_name+"precision.dat",g_minus_2_tab)
#     np.savetxt(run_name+"precisionfav.dat",g_muon_fav_tab)
#     np.savetxt(run_name+"kpipinvisk1.dat",g_K_Vpi_tab_1)
#     np.savetxt(run_name+"kpipinvisk2.dat",g_K_Vpi_tab_2)
#     np.savetxt(run_name+"invispionkappa.dat",invispiondat)


# In[50]:

#def kappa_to_Y(arr,alpha_p=0.1):
#    return [[mv,mx,kappa**2*alpha_p*(mx/mv)**4] for mv,mx,kappa in arr]


# In[25]:

def Y_func(mv,mx,alpha_p,kappa):
    return kappa**2*alpha_p*(mx/mv)**4
def kappa_to_Y(arr,alpha_p=0.1):
    return [np.append([line[0],line[1]],[Y_func(line[0],line[1],alpha_p,kappa) for kappa in line[2:]]) for line in arr]


# In[26]:

def tab_prod_Y(mass_arr,alpha_p,run_name,fill_val=1000,kappa_to_Y=kappa_to_Y):
    #direct_det = [[mv, mx, Direct_Det(mx)] for mv,mx in mass_arr]
    #np.savetxt(directory_out+"direct_det.dat",direct_det)
    relic_tab=kappa_to_Y([[mv,mx,gen_relic_dm(mv,mx,alpha_p)] for mv,mx in mass_arr],alpha_p=alpha_p)
    babar_tab=kappa_to_Y([[mv,mx,babar_func(mv,mx,alpha_p)] for mv,mx in mass_arr],alpha_p=alpha_p)
    rare_tab=kappa_to_Y([[mv,mx,rarelimit(mv,mx,alpha_p)] for mv,mx in mass_arr],alpha_p=alpha_p)
    monojet_tab=kappa_to_Y([[mv,mx,monojet_limit()] for mv,mx in mass_arr],alpha_p=alpha_p)
    g_minus_2_tab=kappa_to_Y([[mv,mx,min(kappa_muon_lim(mv),kappa_electron_lim(mv))] for mv,mx in mass_arr],alpha_p=alpha_p)
    g_muon_fav_tab=kappa_to_Y([[mv,mx,kappa_fav_low(mv),kappa_fav_high(mv)] for mv,mx in mass_arr],alpha_p=alpha_p)
    
    direct_det_tab = kappa_to_Y([[mv,mx,sigman_to_kappa(Direct_Det(mx),mv,mx,alpha_p)] for mv,mx in mass_arr],alpha_p=alpha_p)
  
    K_Vpi_tab1=gen_K_Vpi_lim(kpip_invis_dat_1)
    K_Vpi_func_1=interp1d(K_Vpi_tab1[:,0],K_Vpi_tab1[:,1],bounds_error=False,fill_value=fill_val)
    K_Vpi_tab_1 = kappa_to_Y([[mv,mx,K_Vpi_func_1(mv)] for mv,mx in mass_arr],alpha_p=alpha_p)
    
    K_Vpi_tab2=gen_K_Vpi_lim(kpip_invis_dat_2)
    K_Vpi_func_2=interp1d(K_Vpi_tab2[:,0],K_Vpi_tab2[:,1],bounds_error=False,fill_value=fill_val)
    K_Vpi_tab_2 =kappa_to_Y([[mv,mx,K_Vpi_func_2(mv)] for mv,mx in mass_arr],alpha_p=alpha_p)
    
    invispion_func=interp1d(invispiondat[:,0],invispiondat[:,1],bounds_error=False,fill_value=fill_val)
    invispion_tab=kappa_to_Y([[mv,mx,invispion_func(mv)] for mv,mx in mass_arr],alpha_p=alpha_p)
    
    zprime_func=interp1d(zprimedat[:,0],zprimedat[:,1],bounds_error=False,fill_value=fill_val)
    zprime_tab=kappa_to_Y([[mv,mx,zprime_func(mv)] for mv,mx in mass_arr],alpha_p=alpha_p)
    
    
    np.savetxt(run_name+"direct_det.dat",direct_det_tab)
    np.savetxt(run_name+"babar.dat",babar_tab)
    np.savetxt(run_name+"relic_density.dat",relic_tab)
    np.savetxt(run_name+"rare_decay.dat",rare_tab)
    np.savetxt(run_name+"monojet.dat",monojet_tab)
    np.savetxt(run_name+"precision.dat",g_minus_2_tab)
    np.savetxt(run_name+"precisionfav.dat",g_muon_fav_tab)
    np.savetxt(run_name+"kpipinvisk1.dat",K_Vpi_tab_1)
    np.savetxt(run_name+"kpipinvisk2.dat",K_Vpi_tab_2)
    np.savetxt(run_name+"invispion.dat",invispion_tab)
    np.savetxt(run_name+"zprime.dat",zprime_tab)


# In[60]:

marr=[[mv/1000.0,mv/3000.0] for mv in range(3,4000,1)]


# In[29]:

marr2=[[mv/1000.0,mv/6000.0] for mv in range(3,4000,1)]


# In[27]:

marr3=[[mv/1000.0,mv/5000.0] for mv in range(3,5000,1)]


# In[61]:

test1=tab_prod_Y(marr,0.5,"output/Y_")


# In[ ]:




# In[215]:

np.savetxt("test.dat",test1)


# In[28]:

test2=tab_prod_Y(marr3,0.5,"output/Y_")


# In[57]:

#def GF_func(mv,alpha_b):
#    return 4*math.pi*alpha_b/mv**2


# In[58]:

def GF_func(mv,alpha_b):
    return alpha_b


# In[59]:

def alpha_b_to_GF(arr):
    return [np.append([line[0],line[1]],[GF_func(line[0],alpha_b) for alpha_b in line[2:]]) for line in arr]


# In[60]:

def tab_prod_GF(mass_arr,run_name,fill_val=1000):
    neutron_tab = alpha_b_to_GF([[mv, mx, neutron_scatter(mv)] for mv,mx in mass_arr])
    
    direct_det_tab = alpha_b_to_GF([[mv,mx,sigman_to_alpha_b(Direct_Det(mx),mv,mx)] for mv,mx in mass_arr])
    
    invispion_func=interp1d(invispionbaryonicdat[:,0],invispionbaryonicdat[:,1],bounds_error=False,fill_value=fill_val)
    invispion_tab=alpha_b_to_GF([[mv,mx,invispion_func(mv)] for mv,mx in mass_arr])
    
    K_Vpi_tab1=gen_K_VpiB_lim(kpip_invis_dat_1)
    K_Vpi_func_1=interp1d(K_Vpi_tab1[:,0],K_Vpi_tab1[:,1],bounds_error=False,fill_value=fill_val)
    K_Vpi_tab_1 = alpha_b_to_GF([[mv,mx,K_Vpi_func_1(mv)] for mv,mx in mass_arr])
    
    K_Vpi_tab2=gen_K_VpiB_lim(kpip_invis_dat_2)
    K_Vpi_func_2=interp1d(K_Vpi_tab2[:,0],K_Vpi_tab2[:,1],bounds_error=False,fill_value=fill_val)
    K_Vpi_tab_2 =alpha_b_to_GF([[mv,mx,K_Vpi_func_2(mv)] for mv,mx in mass_arr])
    
    K_Vpi_tab1c=gen_K_VpiB_lim_conservative(kpip_invis_dat_1)
    K_Vpi_func_1c=interp1d(K_Vpi_tab1c[:,0],K_Vpi_tab1c[:,1],bounds_error=False,fill_value=fill_val)
    K_Vpi_tab_1c = alpha_b_to_GF([[mv,mx,K_Vpi_func_1c(mv)] for mv,mx in mass_arr])
    
    K_Vpi_tab2c=gen_K_VpiB_lim_conservative(kpip_invis_dat_2)
    K_Vpi_func_2c=interp1d(K_Vpi_tab2c[:,0],K_Vpi_tab2c[:,1],bounds_error=False,fill_value=fill_val)
    K_Vpi_tab_2c =alpha_b_to_GF([[mv,mx,K_Vpi_func_2c(mv)] for mv,mx in mass_arr])
    
    monojet_tab=alpha_b_to_GF([[mv,mx,monojet_limit_baryonic()] for mv,mx in mass_arr])
        
    rare_tab=alpha_b_to_GF([[mv,mx,rarelimitB(mv,mx,0)] for mv,mx in mass_arr])
    
    np.savetxt(run_name+"neutronscatter.dat",neutron_tab)
    np.savetxt(run_name+"direct_det.dat",direct_det_tab)
    np.savetxt(run_name+"invispion.dat",invispion_tab)
    np.savetxt(run_name+"kpipinvisk1.dat",K_Vpi_tab_1)
    np.savetxt(run_name+"kpipinvisk2.dat",K_Vpi_tab_2)
    np.savetxt(run_name+"kpipinvisk1conservative.dat",K_Vpi_tab_1c)
    np.savetxt(run_name+"kpipinvisk2conservative.dat",K_Vpi_tab_2c)
    np.savetxt(run_name+"monojet.dat",monojet_tab)
    np.savetxt(run_name+"rare_decay.dat",rare_tab)


# In[61]:

tab_prod_GF(marr,"output/GF_")


# In[61]:



