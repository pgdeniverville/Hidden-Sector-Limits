#!/usr/bin/env python

from Hidden_Sec_Utilities import *
from Hidden_Sec_Physics import *

import itertools

"""
I use kappa and epsilon interchangeably. They mean the same thing.
"""

#Default value of alpha_p to use
_alpha_p_set = 0.1

#Returns an array of the V mass, DM mass and kappa^4*alpha_p
def k4al(mv,mx,alpha_p,kappa):
	return [mv,mx,kappa**4*alpha_p]

#Takes an array of masses mass_arr and generates some experimental limits for kinetically mixed hidden sector dark matter. These limits are written to text files.
def table_of_limits(mass_arr,alpha_p=_alpha_p_set,run_name="",fill_val=1000,func=k4al):
	
	#Relic Density, using the fast option.
	print("Generating epsilons to reproduce relic density")
	relic_tab=[k4al(mv,mx,alpha_p,gen_relic_dm_fast(mv,mx,alpha_p)) for mv,mx in mass_arr]
	
	#Best limits of muon and electron g-2
	print("Generating g-2 epsilon limits")
	g_minus_2_tab =[k4al(mv,mx,alpha_p,min(kappa_muon_lim(mv),kappa_electron_lim(mv))) for mv,mx in mass_arr]

	np.savetxt(run_name+"relic_density.dat",relic_tab)
	np.savetxt(run_name+"precision_g_minus_2.dat",g_minus_2_tab)


#Make an array of masses!
marr=[[mv/1000.0,mx/1000.0] for mv in range(10,1000) for mx in range(1,mv/2,1)]

make_sure_path_exists("output/")

#Masses are quite large, so this will take awhile.
table_of_limits(marr,run_name="output/")