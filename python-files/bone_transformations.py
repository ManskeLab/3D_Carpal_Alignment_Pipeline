#   bone_transformations.py
# contains basic functions to perform transformations of the radius, in cases that the radius needs to be re-aligned or re-oriented.
# - crop_meshes: This function will crop the longer of the two meshes (the radius and the secondary mesh that is being transformed) to match the length of the shorter mesh. This is necessary to ensure that the landmark-based alignment and ICP-based alignment are not affected by differences in length between the two meshes.
# - landmark_transform: This function will perform landmark-based alignment between the two meshes. The landmarks are the radial styloid, the highest point, and the widest point. This is just an initial estimate to get the meshes roughly aligned before performing ICP.
# - icp_transform: This function will perform an ICP-based transformation between the two meshes to refine the alignment after landmark-based alignment.
# - transform_uncropped_mesh: This function will apply the combined landmark and ICP transformations to the original, uncropped mesh. This is necessary because the landmark and ICP transformations are performed on the cropped meshes, so this function will apply those transformations to the original mesh to get the final aligned mesh.

import vtk
import numpy as np
import pyvista as pv

RADIUS_SHAFT_DIRECTION = "Y"  # Options are "X", "Y", or "Z"
RADIAL_ULNAR_DIRECTION = "X" 
PALMAR_DORSAL_DIRECTION = "Z"

def crop_meshes(neutral_mesh, target_mesh):
    radius_bounds = neutral_mesh.GetBounds()
    target_bounds = target_mesh.GetBounds()

    # Determine the axis that the shaft of the radius runs along, and crop the longer mesh to match the length of the shorter mesh
    if RADIUS_SHAFT_DIRECTION == "X":
        rad_length = radius_bounds[1] - radius_bounds[0]
        target_length = target_bounds[1] - target_bounds[0]
    elif RADIUS_SHAFT_DIRECTION == "Y":
        rad_length = radius_bounds[3] - radius_bounds[2]
        target_length = target_bounds[3] - target_bounds[2]
    elif RADIUS_SHAFT_DIRECTION == "Z":
        rad_length = radius_bounds[5] - radius_bounds[4]
        target_length = target_bounds[5] - target_bounds[4]

    radius_polydata = neutral_mesh
    target_polydata = target_mesh
    

    buffer = 1 # Adding a buffer of 1 mm to ensure that the cropped mesh is slightly shorter than the target mesh to avoid cases where the shaft of the radius is not aligned perfectly with one of the global axes
    if rad_length > target_length: # That is, the neutral mesh is longer than the secondary (transforming) mesh that is being transformed
        print("Original mesh is longer than the target mesh.")
        radius_crop_box = vtk.vtkBox()
        radius_crop_box.SetBounds(radius_bounds[0], radius_bounds[1], radius_bounds[2], radius_bounds[3] - target_length, radius_bounds[4], radius_bounds[5])
        radius_geometry = vtk.vtkExtractPolyDataGeometry()
        radius_geometry.SetImplicitFunction(radius_crop_box)
        radius_geometry.SetInputData(neutral_mesh)
        radius_geometry.ExtractInsideOn()
        radius_geometry.Update()
        radius_polydata = radius_geometry.GetOutput()
    else: # That is, the mesh that is transforming is longer than the neutral mesh
        print("Target mesh is longer than the original mesh.")
        target_crop_box = vtk.vtkBox()
        target_crop_box.SetBounds(target_bounds[0], target_bounds[1], target_bounds[2], target_bounds[2] + rad_length - buffer, target_bounds[4], target_bounds[5])
        target_geometry = vtk.vtkExtractPolyDataGeometry()
        target_geometry.SetImplicitFunction(target_crop_box)
        target_geometry.SetInputData(target_mesh)
        target_geometry.ExtractInsideOn()
        target_geometry.Update()
        target_polydata = target_geometry.GetOutput()
    return radius_polydata, target_polydata

def landmark_transform(radius_polydata, cropped_transforming_polydata):
    """This function will perform landmark-based alignment between two meshes. The landmarks are the radial styloid, the highest point, and the widest point.
    This is just an initial estimate to get the meshes roughly aligned before performing ICP.

    Args:
        radius_polydata (pv.PolyData): The radius mesh that will be used as the reference for alignment.
        transformed_polydata (pv.PolyData): The secondary (transforming) mesh that will be aligned to the radius mesh.

    Returns:
        transformed_polydata (pv.PolyData): The secondary (transforming) mesh after landmark-based alignment.
        transformed_mesh (pv.PolyData): The secondary (transforming) mesh after landmark-based alignment, in PyVista format.
        landmark_tfm (vtk.vtkTransform): The landmark-based transformation applied to the secondary mesh.
    """
    radius_points = np.array([radius_polydata.GetPoint(i) for i in range(radius_polydata.GetNumberOfPoints())])
    transformed_points = np.array([cropped_transforming_polydata.GetPoint(i) for i in range(cropped_transforming_polydata.GetNumberOfPoints())])

    shaft_direction = None
    if RADIUS_SHAFT_DIRECTION == "X": shaft_direction = 0
    elif RADIUS_SHAFT_DIRECTION == "Y": shaft_direction = 1
    elif RADIUS_SHAFT_DIRECTION == "Z": shaft_direction = 2
    else: print("Error: Invalid RADIUS_SHAFT_DIRECTION value. Must be 'X', 'Y', or 'Z'.")

    radial_ulnar_direction = None
    if RADIAL_ULNAR_DIRECTION == "X": radial_ulnar_direction = 0
    elif RADIAL_ULNAR_DIRECTION == "Y": radial_ulnar_direction = 1
    elif RADIAL_ULNAR_DIRECTION == "Z": radial_ulnar_direction = 2
    else: print("Error: Invalid RADIAL_ULNAR_DIRECTION value. Must be 'X', 'Y', or 'Z'.")

    # Setting landmarks that will used to perform the initial alignment
    source_radial_styloid = radius_points[np.argmax(radius_points[:, radial_ulnar_direction])]
    source_highest_point = radius_points[np.argmax(radius_points[:, shaft_direction])]
    source_widest_point = radius_points[np.argmin(radius_points[:, radial_ulnar_direction])]
    target_radial_styloid = transformed_points[np.argmin(transformed_points[:, radial_ulnar_direction])]
    target_highest_point = transformed_points[np.argmin(transformed_points[:, shaft_direction])]
    target_widest_point = transformed_points[np.argmax(transformed_points[:, radial_ulnar_direction])]

    # Creating VTK points for the landmarks for both bones
    radius_vtk_landmark_points = vtk.vtkPoints()
    radius_vtk_landmark_points.InsertNextPoint(np.array(source_radial_styloid))
    radius_vtk_landmark_points.InsertNextPoint(np.array(source_highest_point))
    radius_vtk_landmark_points.InsertNextPoint(np.array(source_widest_point))
    transformed_vtk_landmark_points = vtk.vtkPoints()
    transformed_vtk_landmark_points.InsertNextPoint(np.array(target_radial_styloid))
    transformed_vtk_landmark_points.InsertNextPoint(np.array(target_highest_point))
    transformed_vtk_landmark_points.InsertNextPoint(np.array(target_widest_point))

    # Defining the transform between the two sets of landmarks
    landmark_transform = vtk.vtkLandmarkTransform()
    landmark_transform.SetSourceLandmarks(transformed_vtk_landmark_points)
    landmark_transform.SetTargetLandmarks(radius_vtk_landmark_points)
    landmark_transform.SetModeToRigidBody()
    landmark_transform.Update()

    landmark_tfm = vtk.vtkTransform() # This transform will be used to apply the landmark-based transformation to the entire mesh
    landmark_tfm.SetMatrix(landmark_transform.GetMatrix())

    transform_filter = vtk.vtkTransformPolyDataFilter() # Applying the landmark-based transformation to the mesh that is transforming position
    transform_filter.SetInputData(cropped_transforming_polydata)
    transform_filter.SetTransform(landmark_tfm)
    transform_filter.Update()
    transformed_polydata = transform_filter.GetOutput()
    transformed_mesh = pv.wrap(transformed_polydata)

    return transformed_polydata, transformed_mesh, landmark_tfm

def icp_transform(radius_polydata, cropped_transforming_polydata):
    """This function will perform an ICP-based transformation between two meshes to refine the alignment after landmark-based alignment.

    Args:
        radius_polydata (pv.PolyData): The radius mesh that will be used as the reference for alignment.
        cropped_transforming_polydata (pv.PolyData): The mesh that will be aligned (that is, transforming) to the radius mesh.

    Returns:
        transforming_polydata (pv.PolyData): The transforming mesh after ICP-based alignment.
        transformed_mesh (pv.PolyData): The secondary mesh after ICP-based alignment, in PyVista format.
        icp_tfm (vtk.vtkTransform): The ICP-based transformation applied to the secondary mesh.
    """
    # Performing ICP-based alignment between the two meshes using standard VTK parameters
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(cropped_transforming_polydata)
    icp.SetTarget(radius_polydata)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(100)
    icp.SetMeanDistanceModeToRMS()
    icp.Update()

    # Applying the ICP transformation to the mesh that is being aligned
    icp_tfm = vtk.vtkTransform()
    icp_tfm.SetMatrix(icp.GetMatrix())
    
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(cropped_transforming_polydata)
    transform_filter.SetTransform(icp_tfm)
    transform_filter.Update()

    # Getting the new polydata and mesh after ICP transformation
    transforming_polydata = transform_filter.GetOutput()
    transformed_mesh = pv.wrap(cropped_transforming_polydata)

    return transforming_polydata, transformed_mesh, icp_tfm

def transform_uncropped_mesh(original_transforming_mesh, landmark_tfm, icp_tfm):
    """This function will apply the combined landmark and ICP transformations to the original, uncropped mesh.

    Args:
        original_transforming_mesh (pv.PolyData): The original mesh (considered secondary) that will be aligned to the radius mesh.
        landmark_tfm (vtk.vtkTransform): The landmark-based transformation.
        icp_tfm (vtk.vtkTransform): The ICP-based transformation.

    Returns:
        transforming_polydata (pv.PolyData): The secondary mesh after combined landmark and ICP-based alignment.
        transforming_mesh (pv.PolyData): The secondary mesh after combined landmark and ICP-based alignment, in PyVista format.
        final_tfm (vtk.vtkTransform): The combined transformation applied to the secondary mesh.
    """
    final_tfm = vtk.vtkTransform()
    final_tfm.PostMultiply()
    final_tfm.Concatenate(landmark_tfm)
    final_tfm.Concatenate(icp_tfm)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(original_transforming_mesh)
    transform_filter.SetTransform(final_tfm)
    transform_filter.Update()
    transforming_polydata = transform_filter.GetOutput()
    transforming_mesh = pv.wrap(transforming_polydata)

    return transforming_polydata, transforming_mesh, final_tfm
