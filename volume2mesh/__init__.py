import os
import numpy as np
import pygalmesh
import meshio
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
from scipy import stats

def main(input_file, outputDir, zoom_value=1, r_start=None, r_end=None, data_files=None, **kwargs):
  mask_file = input_file
  label_files = {}

  if not data_files is None:
    for data_file in data_files:
      dn = str(os.path.basename(data_file)).split(".")[0]
      label_files[dn] = data_file

  # read mask
  mask_img = sitk.ReadImage(mask_file)
  mask = sitk.GetArrayFromImage(mask_img)
  mask = np.asarray(mask, dtype="uint8")

  if not r_start:
    mask = np.asarray(mask >= r_start, dtype="uint8")

  if not r_end:
    mask = np.asarray(mask <= r_end, dtype="uint8")

  # read labels
  labels = {}
  for k in label_files:
    label_image = sitk.ReadImage(label_files[k])
    label = sitk.GetArrayFromImage(label_image)
    labels[k] = label


  if zoom_value != 1:
    mask = zoom(mask, zoom=zoom_value, order=0, mode='nearest')

    for k in labels:
      if str(labels[k].dtype).lower().find('int') >= 0:
        labels[k] = zoom(labels[k], zoom=zoom_value, order=0, mode='nearest')
      else:
        labels[k] = zoom(labels[k], zoom=zoom_value, order=3, mode='nearest')

  voxel_size = (0.1, 0.1, 0.1)

  mesh = generate_mesh(mask, voxel_size, labels)
  
  msk_name = str(os.path.basename(input_file)).split(".")[0]
  msk_path = os.path.join(outputDir, f"{msk_name}.vtk")
  mesh.write(msk_path, file_format='vtk42')

  return {"mesh": msk_path}

def generate_mesh(data, voxel_size, labels={}):
  dd = data.astype(dtype=np.uint16)

  cell_sizes_map = {}
  cell_sizes_map['default'] = 0.5
      
  mesh = pygalmesh.generate_from_array(dd, voxel_size, max_cell_circumradius=cell_sizes_map, max_facet_distance=.5*voxel_size[0], verbose=True)

  cells = mesh.get_cells_type('tetra')
  points = mesh.points * (1 / np.asarray(voxel_size))
  
  mesh.cells = [meshio.CellBlock('tetra', cells)] 
  indexes = np.asarray(points, dtype=np.uint16).T
  
  for k in labels:
    label_values = labels[k]
    index_values = label_values[indexes[0], indexes[1], indexes[2]]
    label = index_values[cells]

    dtype = str(label.dtype).lower().find('int')
    if  dtype >= 0:
      label = stats.mode(label, axis=-1, keepdims=False).mode
    else:
      label = np.mean(label, axis=-1)

    mesh.cell_data[k] = [label]

  return mesh