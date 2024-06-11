import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import scipy.ndimage

import copy
from scipy.ndimage import zoom
from solver import Solver_FK_2c 

def main(Diffusion_param, Proliferation_param, x_coord, y_coord, z_coord, **kwargs):

    #################################### start here settings ####################################
	# Create binary segmentation masks
	gm_data = nib.load('./GM.nii.gz').get_fdata()
	wm_data = nib.load('./WM.nii.gz').get_fdata()
	affine = nib.load('./GM.nii.gz').affine

	# Set up parameters
	parameters = {
		'Dw': Diffusion_param, # Diffusion coefficient for the white matter
		'rho': Proliferation_param, # Proliferation rate
		'lambda_np': 0.35, # Transition rate between proli and necrotic cells
		'sigma_np': 0.5, #Transition threshols between proli and necrotic given nutrient field
		'D_s': 1.3,      # Diffusion coefficient for the nutrient field
		'lambda_s': 0.05, # Proli cells nutrients consumption rate
		'RatioDw_Dg': 100,  # Ratio of diffusion coefficients in white and grey matter
		'Nt_multiplier': 8,
		'gm': gm_data,      # Grey matter data
		'wm': wm_data,      # White matter data
		'NxT1_pct': x_coord,    # tumor position [%]
		'NyT1_pct': y_coord,
		'NzT1_pct': z_coord,
		'init_scale': 1., #scale of the initial gaussian
		'resolution_factor': 0.5, #resultion scaling for calculations
		'th_matter': 0.1, #when to stop diffusing: at th_matter > gm+wm
		'verbose': True, #printing timesteps 
		'time_series_solution_Nt': 8, # number of timesteps in the output
		'stopping_volume': 10000
	}

	###################################### end here settings ####################################


	# Run the FK_solver and plot the results
	print("llalalalal") 
	fk_solver = Solver_FK_2c(parameters)
	result = fk_solver.solve()
	#save results
	tumor_path = './tumor_final.nii.gz'
	nib.save(nib.Nifti1Image(result['final_state']['P'], affine), tumor_path)
	#tumor = nib.Nifti1Image(result['final_state']['P'], affine) 
	
	return tumor_path