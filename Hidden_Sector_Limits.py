#!/usr/bin/env python

from Hidden_Sec_Utilities import *
from Hidden_Sec_Physics import *

import itertools

"""
I use kappa and epsilon interchangeably. They mean the same thing.
Many of these limits are not valid in the off-shell regime, or
change dramatically. Use at your own risk!
"""

#Default value of alpha_p to use
_alpha_p_set = 0.5

#Takes an array of masses mass_arr and generates some experimental limits for kinetically mixed hidden sector dark matter. These limits are written to text files.
#func can be any function that accepts arguments in the form (mv,mx,alpha_p,kappa).
def table_of_limits(mass_arr,alpha_p=_alpha_p_set,run_name="",fill_val=1000,func=Y_func):
	mass_arr = np.array(mass_arr)

	#Relic Density, using the fast option.
	print("Generating epsilons to reproduce relic density")
	relic_tab=[func(mv,mx,alpha_p,gen_relic_dm_fast(mv,mx,alpha_p)) for mv,mx in mass_arr]

	#Best limits of muon and electron g-2
	print("Generating g-2 epsilon limits")
	g_minus_2_tab = [func(mv,mx,alpha_p,min(kappa_muon_lim(mv),kappa_electron_lim(mv))) for mv,mx in  mass_arr]
	g_minus_2_electron = [func(mv,mx,alpha_p,kappa_electron_lim(mv)) for mv,mx in  mass_arr]
	g_minus_2_muon = [func(mv,mx,alpha_p,kappa_muon_lim(mv)) for mv,mx in  mass_arr]
	print("Generating g-2 epsilon favoured region")
	g_muon_fav_low_tab = [func(mv,mx,alpha_p,kappa_fav_low(mv)) for mv,mx in mass_arr]
	g_muon_fav_high_tab = [func(mv,mx,alpha_p,kappa_fav_high(mv)) for mv,mx in mass_arr]

	print("Generating BaBar limits")
	babar_tab=[func(mv,mx,alpha_p,babar_func(mv,mx,alpha_p)) for mv,mx in mass_arr]

	print("Generating BaBar 2017 limits")
	babar2017_tab=[func(mv,mx,alpha_p,babar2017_interp(mv)) for mv,mx in mass_arr]

        print("Generating limits from rare decays (J\Psi->V)")
	rare_tab = [func(mv,mx,alpha_p,rarelimit(mv,mx,alpha_p)) for mv,mx in mass_arr]

	print("Generating Monojet limits")
	monojet_tab = [func(mv,mx,alpha_p,monojet_limit()) for mv,mx in mass_arr]

	print("Generating rare kaon decay limits (K->pi+V)")
	K_Vpi_tab1=gen_K_Vpi_lim(kpip_invis_dat_1)
	K_Vpi_func_1=interp1d(K_Vpi_tab1[:,0],K_Vpi_tab1[:,1],bounds_error=False,fill_value=fill_val)
	K_Vpi_tab2=gen_K_Vpi_lim(kpip_invis_dat_2)
	K_Vpi_func_2=interp1d(K_Vpi_tab2[:,0],K_Vpi_tab2[:,1],bounds_error=False,fill_value=fill_val)
	k_Vpi_tab = [func(mv,mx,alpha_p,min(K_Vpi_func_1(mv),K_Vpi_func_2(mv))) for mv,mx in  mass_arr]

	print("Generating pion->invisible limits")
	invispion_func=interp1d(invispiondat[:,0],invispiondat[:,1],bounds_error=False,fill_value=fill_val)
	invispion_tab=[func(mv,mx,alpha_p,invispion_func(mv)) for mv,mx in mass_arr]

	#Electroweak/shift in Z mass etc.
	print("Generating Electroweak fit limits")
	zprime_func=interp1d(zprimedat[:,0],zprimedat[:,1],bounds_error=False,fill_value=fill_val)
	zprime_tab = [func(mv,mx,alpha_p,zprime_func(mv)) for mv,mx in mass_arr]

	#E137
	print("Generating E137 limits - Note that this currently does not use the supplied func, as odd c runtime errors are encountered. Uncomment the following commented lines if another function is required, but test carefully!")
	#e137dat=griddata(E137tab[:,0:2],E137tab[:,2],mass_arr,fill_value=fill_val,method='linear')
	e137_vals= np.array([func(mv,mx,alpha_p,(k4alphap/alpha_p)**0.25) for mv,mx,k4alphap in E137tab])
	e137dat = griddata(E137tab[:,0:2],e137_vals[:,2],mass_arr,fill_value=fill_val,method='linear')

	E137_tab = zip(mass_arr[:,0],mass_arr[:,1],e137dat)

	#LSND
	print("Generating LSND limits")
	lsnd_vals= np.array([func(mv,mx,alpha_p,(k4alphap/alpha_p)**0.25) for mv,mx,k4alphap in LSNDtab])
	LSNDdat = griddata(LSNDtab[:,0:2],lsnd_vals[:,2],mass_arr,fill_value=fill_val,method='linear')
	LSND_tab = zip(mass_arr[:,0],mass_arr[:,1],LSNDdat)

        print("Generating limits from Direct Detection")
	direct_det_tab = [func(mv,mx,alpha_p,sigman_to_kappa(Direct_Det(mx),mv,mx,alpha_p)) for mv,mx in mass_arr]

        print("Generating limits from Direct Detection - Electron")
        direct_det_e_tab = [func(mv,mx,alpha_p,sigmae_to_kappa(xenon10efunc(mx),mv,mx,alpha_p)) for mv,mx in mass_arr]

	print("Generating NA64 limits")
        NA64_func=interp1d(NA64dat[:,0],NA64dat[:,1],bounds_error=False,fill_value=fill_val)
	NA64_tab=[func(mv,mx,alpha_p,NA64_func(mv)) for mv,mx in mass_arr]

	np.savetxt(run_name+"relic_density.dat",relic_tab)
	np.savetxt(run_name+"precision_g_minus_2.dat",g_minus_2_tab)
	np.savetxt(run_name+"precision_g_minus_2_electron.dat",g_minus_2_electron)
	np.savetxt(run_name+"precision_g_minus_2_muon.dat",g_minus_2_muon)
	np.savetxt(run_name+"precision_g_minus_2_fav_low.dat",g_muon_fav_low_tab)
	np.savetxt(run_name+"precision_g_minus_2_fav_high.dat",g_muon_fav_high_tab)
	#np.savetxt(run_name+"direct_det.dat",direct_det_tab)
	np.savetxt(run_name+"babar.dat",babar_tab)
	np.savetxt(run_name+"babar2017.dat",babar2017_tab)
	np.savetxt(run_name+"relic_density.dat",relic_tab)
	np.savetxt(run_name+"rare_decay.dat",rare_tab)
	np.savetxt(run_name+"monojet.dat",monojet_tab)
	np.savetxt(run_name+"kpipinvisk.dat",k_Vpi_tab)
	np.savetxt(run_name+"invispion.dat",invispion_tab)
	np.savetxt(run_name+"zprime.dat",zprime_tab)
	np.savetxt(run_name+"lsndlim.dat",LSND_tab)
	np.savetxt(run_name+"e137lim.dat",E137_tab)
	np.savetxt(run_name+"direct_det.dat",direct_det_tab)
	np.savetxt(run_name+"direct_det_e.dat",direct_det_e_tab)
	np.savetxt(run_name+"NA64.dat",NA64_tab)


#Make an array of masses!
#marr=[[mv/1000.0,mx/1000.0] for mv in range(10,1000) for mx in range(1,mv/2,1)]
#marr=[[mv/1000.0,mx/1000.0] for mv in range(10,100) for mx in range(1,mv/2,1)]

marr=[[3*mx/1000.0,mx/1000.0] for mx in range(1,1000)]

make_sure_path_exists("output/")

#Masses are quite large, so this will take awhile.
table_of_limits(marr,run_name="output/y3_")

mxset=10
runname="output/mx"+masstext(mxset/1000.0)+"_"
marr2=[[mv/1000.0,mxset/1000.0] for mv in range(mxset,4000)]+[[(mv-0.5)/1000.0,mxset/1000.0],[(mv+0.5)/1000.0,mxset/1000.0]]
table_of_limits(marr2,run_name=runname,func=kappa,alpha_p=0.1)
