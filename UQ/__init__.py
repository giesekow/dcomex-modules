import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import scipy.ndimage
from scipy.stats import median_abs_deviation

import os
import copy
from scipy.ndimage import zoom

def compute_elementwise_statistics(data_array):
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    median = np.median(data_array, axis=0)
    mad = median_abs_deviation(data_array, axis=0)
    return mean, std, median, mad

def aggregate_data(folder_path):
    aggregated_data = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            file_path = os.path.join(folder_path, filename)
            img = nib.load(file_path)
            data = img.get_fdata()
            aggregated_data.append(data)
            
    if not aggregated_data:
        raise ValueError("No NIfTI files found in the folder.")
    data_shape = aggregated_data[0].shape
    
    for data in aggregated_data:
        if data.shape != data_shape:
            raise ValueError("All NIfTI files must have the same shape.")
        
    aggregated_data = np.array(aggregated_data)
    affine = img.affine

    return aggregated_data, affine

def main(mean="yes", std="yes", median="yes", mad="yes", **kwargs):
	print("1")
	folder_path = "./image_volumes/"
	aggregated_data, affine = aggregate_data(folder_path)
	print("2")
	mean_image, std_image, median_image, mad_image = compute_elementwise_statistics(aggregated_data)
	print(mean_image.shape)
	mean_path = "mean.nii.gz"
	std_path = "std.nii.gz"
	median_path = "median.nii.gz"
	mad_path = "mad.nii.gz"
      
	if mean=="yes":
		
		mean_img = nib.Nifti1Image(mean_image, affine)
		nib.save(mean_img, mean_path)
	if std=="yes":
		
		std_img = nib.Nifti1Image(std_image, affine)
		nib.save(std_img, std_path)
	if median=="yes":
		
		median_img = nib.Nifti1Image(median_image, affine)
		nib.save(median_img, median_path)
	if mad=="yes":
		
		mad_img = nib.Nifti1Image(mad_image, affine)
		nib.save(mad_img, mad_path)

	return mean_path, std_path, median_path, mad_path, 
	



	
	