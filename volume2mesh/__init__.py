import vtk, os

def main(input_file, iso_value, outputDir):
  out_file = os.path.join(outputDir, 'out.vtk')
  
  print("Reading volume data...")
  reader = vtk.vtkNIFTIImageReader()
  reader.SetFileName(input_file)
  reader.Update()

  print("Converting volume data to mesh...")
  contour=vtk.vtkMarchingCubes()  
  contour.SetInputData(reader.GetOutput())
  contour.ComputeNormalsOn()
  contour.ComputeGradientsOn()
  contour.SetValue(0, iso_value)
  contour.Update()

  print("Writing mesh data...")
  writer = vtk.vtkPolyDataWriter()
  writer.SetFileVersion(42)
  writer.SetInputData(contour.GetOutput())
  writer.SetFileName(out_file)
  writer.Write()

  print("Done Processing!")

  return {"mesh": out_file}