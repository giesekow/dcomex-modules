import os
from .lib import infer

def main(t1, t1c, t2, fla, outputDir, **kwargs):
  input_dict = dict(t1=t1, t1c=t1c, t2=t2, fla=fla)

  output_dict = dict(
    segmentation=os.path.join(outputDir, 'segmentation.nii.gz')
  )

  infer(input_dict=input_dict, output_dict=output_dict, options_dict=kwargs)