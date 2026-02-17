# read_vtk.py
#  contains function to read in a VTK file and apply any necessary transformations:
# - read_vtk: basic function for reading in a VTK mesh file. This setup is specific to the dataset used in KRASL and will need to be adjusted for other datasets depending on the orientation of the meshes and what transformations are necessary to match right hand orientation.
import vtk


def read_vtk(vtk_path, hand) -> tuple[vtk.vtkPolyData, vtk.vtkPoints, vtk.vtkPointData]:
    """This function reads a VTK file and returns the polydata, points, and point data.

    Args:
        vtk_path (str): The path to the VTK file.

    Returns:
        polydata (vtk.vtkPolyData): The VTK polydata.
        points (vtk.vtkPoints): The VTK points.
        point_data (vtk.vtkPointData): The VTK point data.
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    if hand == "L": # This is done to rotate all left hand meshes to match right hand orientation. It should be noted that this is specific to the dataset used in KRASL and may not be applicable to other datasets.
        transform = vtk.vtkTransform()
        transform.RotateY(180) # These may need to be adjusted for other datasets depending on what orientation needs to be flipped to match right hand orientation
        transform.Scale(1, 1, -1)
        transform.Translate(-80, 0, 0)
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(reader.GetOutput())
        transform_filter.SetTransform(transform)
        transform_filter.Update()
        polydata = transform_filter.GetOutput() 
        points = polydata.GetPoints()
        point_data = polydata.GetPointData()
    else:
        polydata = reader.GetOutput()
        points = polydata.GetPoints()
        point_data = polydata.GetPointData()
    return polydata, points, point_data

