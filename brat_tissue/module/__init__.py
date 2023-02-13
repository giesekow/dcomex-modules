import os, time, tempfile, shutil
from .skull import SkullStripper
import nibabel as nib
import numpy as np
from nilearn.image import math_img

def main(t1, threshold, outputDir, **kwargs):
  want_tissues = True
  want_atlas = False
  input_path = t1
  output_path = tempfile.TemporaryDirectory().name

  if not os.path.exists(output_path):
      print("The selected output folder doesn't exist, so I am making it \n")
      os.makedirs(output_path)

  start = time.time()
  skull_stripper = SkullStripper(input_path, output_path, want_tissues, want_atlas)
  skull_stripper.strip_skull()
  print('Done (' + str((time.time() - start) / 60.) + ' min)')
  
  # Copy tissue outputs
  fname = os.path.splitext(os.path.splitext(os.path.basename(input_path))[0])[0]
  wm_path = os.path.join(output_path, fname + "_wm.nii.gz")
  gm_path = os.path.join(output_path, fname + "_gm.nii.gz")
  csf_path = os.path.join(output_path, fname + "_csf.nii.gz")
  msk = os.path.join(output_path, fname + "_mask.nii.gz")

  wm = nib.load(wm_path)
  gm = nib.load(gm_path)
  csf = nib.load(csf_path)

  eqn = f"msk + (wm > {threshold})*2 + (gm > {threshold})*1 + (csf > {threshold})*3"
  full_img = math_img(eqn, msk=msk, wm=wm, gm=gm, csf=csf)

  out_file = os.path.join(outputDir, fname + "_classes.nii.gz")
  out_wm_path = os.path.join(outputDir, fname + "_wm.nii.gz")
  out_gm_path = os.path.join(outputDir, fname + "_gm.nii.gz")
  out_csf_path = os.path.join(outputDir, fname + "_csf.nii.gz")
  
  nib.save(full_img, out_file)
  shutil.copy(wm_path, out_wm_path)
  shutil.copy(gm_path, out_gm_path)
  shutil.copy(csf_path, out_csf_path)
  
  return {"classes": out_file, "probs": [out_gm_path, out_wm_path, out_csf_path]}