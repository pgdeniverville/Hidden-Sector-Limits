#!/usr/bin/env python

from Hidden_Sec_Utilities import *
from Hidden_Sec_Physics import *

import itertools

"""
I use kappa and epsilon interchangeably. They mean the same thing.
Many of these limits are not valid in the off-shell regime, or
change dramatically! Use at your own risk!
"""

#Default value of alpha_p to use
_alpha_p_set = 0.1

#Returns an array of the V mass, DM mass and kappa^4*alpha_p
def k4al(mv,mx,alpha_p,kappa):
	return [mv,mx,kappa**4*alpha_p]

#Takes an array of masses mass_arr and generates some experimental limits for kinetically mixed hidden sector dark matter. These limits are written to text files. 
#func can be any function that accepts arguments in the form (mv,mx,alpha_p,kappa).
def table_of_limits(mass_arr,alpha_p=_alpha_p_set,run_name="",fill_val=1000,func=k4al):
	
	#Relic Density, using the fast option.
	print("Generating epsilons to reproduce relic density")
	relic_tab=[k4al(mv,mx,alpha_p,gen_relic_dm_fast(mv,mx,alpha_p)) for mv,mx in mass_arr]
	
	#Best limits of muon and electron g-2
	print("Generating g-2 epsilon limits")
	g_minus_2_tab = [k4al(mv,mx,alpha_p,min(kappa_muon_lim(mv),kappa_electron_lim(mv))) for mv,mx in  mass_arr]
	print("Generating g-2 epsilon favoured region")
	g_muon_fav_low_tab = [k4al(mv,mx,alpha_p,kappa_fav_low(mv)) for mv,mx in mass_arr]
	g_muon_fav_high_tab = [k4al(mv,mx,alpha_p,kappa_fav_high(mv)) for mv,mx in mass_arr]

	print("Generating BaBar limits")
	babar_tab=[k4al(mv,mx,alpha_p,babar_func(mv,mx,alpha_p)) for mv,mx in mass_arr]

	print("Generating limits from rare decays (J\Psi->V)")
	rare_tab = [k4al(mv,mx,alpha_p,rarelimit(mv,mx,alpha_p)) for mv,mx in mass_arr]

	print("Generating Monojet limits")
	monojet_tab = [k4al(mv,mx,alpha_p,monojet_limit()) for mv,mx in mass_arr]

	print("Generating rare kaon decay limits (K->pi+V)")
	K_Vpi_tab1=gen_K_Vpi_lim(kpip_invis_dat_1)
	K_Vpi_func_1=interp1d(K_Vpi_tab1[:,0],K_Vpi_tab1[:,1],bounds_error=False,fill_value=fill_val)
	K_Vpi_tab2=gen_K_Vpi_lim(kpip_invis_dat_2)
	K_Vpi_func_2=interp1d(K_Vpi_tab2[:,0],K_Vpi_tab2[:,1],bounds_error=False,fill_value=fill_val)
	k_Vpi_tab = [k4al(mv,mx,alpha_p,min(K_Vpi_func_1(mv),K_Vpi_func_2(mv))) for mv,mx in  mass_arr]

	print("Generating pion->invisible limits")
	invispion_func=interp1d(invispiondat[:,0],invispiondat[:,1],bounds_error=False,fill_value=fill_val)
	invispion_tab=[k4al(mv,mx,alpha_p,invispion_func(mv)) for mv,mx in mass_arr]

	#Electroweak/shift in Z mass etc.
	print("Generating Electroweak fit limits")
	zprime_func=interp1d(zprimedat[:,0],zprimedat[:,1],bounds_error=False,fill_value=fill_val)
	zprime_tab = [k4al(mv,mx,alpha_p,zprime_func(mv)) for mv,mx in mass_arr]

	np.savetxt(run_name+"relic_density.dat",relic_tab)
	np.savetxt(run_name+"precision_g_minus_2.dat",g_minus_2_tab)
	np.savetxt(run_name+"precision_g_minus_2_fav_low.dat",g_muon_fav_low_tab)
	np.savetxt(run_name+"precision_g_minus_2_fav_high.dat",g_muon_fav_high_tab)
	#np.savetxt(run_name+"direct_det.dat",direct_det_tab)
	np.savetxt(run_name+"babar.dat",babar_tab)
	np.savetxt(run_name+"relic_density.dat",relic_tab)
	np.savetxt(run_name+"rare_decay.dat",rare_tab)
	np.savetxt(run_name+"monojet.dat",monojet_tab)
	np.savetxt(run_name+"kpipinvisk.dat",k_Vpi_tab)
	np.savetxt(run_name+"invispion.dat",invispion_tab)
	np.savetxt(run_name+"zprime.dat",zprime_tab)


#Make an array of masses!
marr=[[mv/1000.0,mx/1000.0] for mv in range(10,1000) for mx in range(1,mv/2,1)]

#marr=[[mv/1000.0,mx/1000.0] for mv in range(10,100) for mx in range(1,mv/2,1)]


make_sure_path_exists("output/")

#Masses are quite large, so this will take awhile.
table_of_limits(marr,run_name="output/")