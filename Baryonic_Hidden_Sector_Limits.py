#!/usr/bin/env python

from Hidden_Sec_Utilities import *
from Hidden_Sec_Physics import *

import itertools

_kappa = 0.0

def table_of_limits_baryonic(mass_arr,kappa=_kappa,run_name="",fill_val=1000,func=alphab):
    mass_arr = np.array(mass_arr)

    #Limits from effects on neutron scattering
    print("Generating limits from neutron scattering")
    neutron_tab = [func(mv,mx,neutron_scatter(mv),kappa) for mv,mx in mass_arr]

    #Limits from direct detection
    print("Generating limits from direct detection")
    direct_det_tab = [func(mv,mx,sigman_to_alpha_b(Direct_Det(mx),mv,mx),kappa) for mv,mx in mass_arr]

    #Decays of pions to V_B -> invisible
    print("Generating limits from invisible pion decays")
    invispion_func=interp1d(invispionbaryonicdat[:,0],invispionbaryonicdat[:,1],bounds_error=False,fill_value=fill_val)
    invispion_tab = [func(mv,mx,invispion_func(mv),kappa) for mv,mx in mass_arr]

    #Invisible kaon decays
    print("Generating limits from invisible kaon decays")
    K_Vpi_tab1=gen_K_VpiB_lim(kpip_invis_dat_1)
    K_Vpi_func_1=interp1d(K_Vpi_tab1[:,0],K_Vpi_tab1[:,1],bounds_error=False,fill_value=fill_val)
    K_Vpi_tab2=gen_K_VpiB_lim(kpip_invis_dat_2)
    K_Vpi_func_2=interp1d(K_Vpi_tab2[:,0],K_Vpi_tab2[:,1],bounds_error=False,fill_value=fill_val)
    k_Vpi_tab = [func(mv,mx,min(K_Vpi_func_1(mv),K_Vpi_func_2(mv)),kappa) for mv,mx in  mass_arr]

    K_Vpi_tab1=gen_K_VpiB_lim_conservative(kpip_invis_dat_1)
    K_Vpi_func_1=interp1d(K_Vpi_tab1[:,0],K_Vpi_tab1[:,1],bounds_error=False,fill_value=fill_val)
    K_Vpi_tab2=gen_K_VpiB_lim_conservative(kpip_invis_dat_2)
    K_Vpi_func_2=interp1d(K_Vpi_tab2[:,0],K_Vpi_tab2[:,1],bounds_error=False,fill_value=fill_val)
    k_Vpi_tabc = [func(mv,mx,min(K_Vpi_func_1(mv),K_Vpi_func_2(mv)),kappa) for mv,mx in  mass_arr]



    print("Generating limits from monojet")
    monojet_tab = [func(mv,mx,monojet_limit_baryonic(),kappa) for mv,mx in mass_arr]

    print("Generating limits from rare decays (J\Psi->V)")
    rare_tab = [func(mv,mx,rarelimitB(mv,mx,kappa),kappa) for mv,mx in mass_arr]

    np.savetxt(run_name+"neutronscatter.dat",neutron_tab)
    np.savetxt(run_name+"direct_det.dat",direct_det_tab)
    np.savetxt(run_name+"invispion.dat",invispion_tab)
    np.savetxt(run_name+"kpipinvisk.dat",k_Vpi_tab)
    np.savetxt(run_name+"kpipinviskconservative.dat",k_Vpi_tabc)
    np.savetxt(run_name+"monojet.dat",monojet_tab)
    np.savetxt(run_name+"rare_decay.dat",rare_tab)

#marr=[[mv/1000.0,mx/1000.0] for mv in range(10,100) for mx in range(1,mv/2,1)
marr=[[3*mx/1000.0,mx/1000.0] for mx in range(1,1500)]

make_sure_path_exists("output/")

table_of_limits_baryonic(marr,run_name="output/bar3_")

mxset=0.01
runname="output/barmx"+masstext(mxset)+"_"
marr2=[[mv/1000.0,mxset] for mv in range(mx,4000)]
table_of_limits_baryonic(marr2,run_name=runname)

