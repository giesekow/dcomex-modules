from .module.lib import brats_preprocessing
import os

def main(t1, t1c, t2, fla, outputDir, **kwargs):
  input_dict = dict(t1=t1, t1c=t1c, t2=t2, fla=fla)
  
  output_dict = dict(
    t1=os.path.join(outputDir, 't1.nii.gz'),
    t1c=os.path.join(outputDir, 't1c.nii.gz'),
    t2=os.path.join(outputDir, 't2.nii.gz'),
    fla=os.path.join(outputDir, 'fla.nii.gz')
  )

  brats_preprocessing(input_dict=input_dict, output_dict=output_dict, options_dict=kwargs)
  return output_dict