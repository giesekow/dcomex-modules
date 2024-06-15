import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import scipy.ndimage

import copy
from scipy.ndimage import zoom
from solver import Solver_FK_2c 

def update_stats(volume, count, mean, M2):
    """
    Update mean and M2 (sum of squares of differences from the mean) for Welford's algorithm.
    
    Parameters:
    volume (numpy.ndarray): The new volume to include in the statistics.
    count (int): The current count of volumes processed.
    mean (numpy.ndarray): The current mean of volumes.
    M2 (numpy.ndarray): The current sum of squares of differences from the mean.
    
    Returns:
    count (int): The updated count.
    mean (numpy.ndarray): The updated mean.
    M2 (numpy.ndarray): The updated M2.
    """
    count += 1
    delta = volume - mean
    mean += delta / count
    delta2 = volume - mean
    M2 += delta * delta2
    return count, mean, M2

def finalize_variance(count, mean, M2):
    """
    Finalize the variance computation.
    
    Parameters:
    count (int): The total count of volumes.
    mean (numpy.ndarray): The final mean of volumes.
    M2 (numpy.ndarray): The final sum of squares of differences from the mean.
    
    Returns:
    variance (numpy.ndarray): The variance of the volumes.
    """
    if count < 2:
        return np.zeros_like(mean)  # Variance is zero if less than 2 volumes
    variance = M2 / (count - 1)
    return variance


def main(Diffusion_param, Proliferation_param, x_coord, y_coord, z_coord, uncertainty, **kwargs):

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
	if uncertainty=="yes":
		number_of_simulations = 5
	else:
		number_of_simulations = 1
    
	count = 0
	mean = None
	variance = None
	M2 = None
	for i in range(number_of_simulations):
		

		if uncertainty=="yes":
			binary_gm = (gm_data > 0.05*i).astype(np.uint8)
			binary_wm = (wm_data > 0.05*i).astype(np.uint8)
			parameters["gm"] = binary_gm
			parameters["wm"] = binary_wm


			fk_solver = Solver_FK_2c(parameters)
			volume = fk_solver.solve()
		
			volume = volume['final_state']['P']
			if mean is None:
				mean = np.zeros_like(volume, dtype=np.float64)
				M2 = np.zeros_like(volume, dtype=np.float64)

			count, mean, M2 = update_stats(volume, count, mean, M2)
		else:
			fk_solver = Solver_FK_2c(parameters)
			volume = fk_solver.solve()
		
    
	#save results
	if uncertainty=="yes":
		variance = finalize_variance(count, mean, M2)
		tumor_path = './tumor_final_{}.nii.gz'.format(i)
		nib.save(nib.Nifti1Image(volume, affine), tumor_path)

		variance_path = './tumor_variance.nii.gz'
		nib.save(nib.Nifti1Image(variance, affine), variance_path)
		return tumor_path, variance_path
	else:
		tumor_path = './tumor_final_{}.nii.gz'.format(i)
		nib.save(nib.Nifti1Image(volume['final_state']['P'], affine), tumor_path)
		return tumor_path


	
	