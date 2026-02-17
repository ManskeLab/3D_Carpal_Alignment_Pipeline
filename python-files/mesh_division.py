# mesh_division.py
# contains functions to divide meshes based on PCA or principal axes:
# - half_mesh: divides the mesh into two parts based on the sign of the projection along a given direction.
# - divide_mesh_by_pca_into_thirds: divides the mesh into three parts based on projection along a principal component direction, assigning specific colors to each third.
# - divide_mesh_into_sections: splits a mesh into multiple divisions along a specified principal axis, allowing for an arbitrary number of divisions.
# - compute_bone_centroid: determines the centroid of a bone mesh using two methods: the mean of all points and the vtk.vtkCenterOfMass class.
# - split_bone_into_quadrants: splits a bone into quadrants based on the principal inertial axes, with splits including (+Y, +Z), (+Y, -Z), (-Y, -Z), (-Y, +Z).
# - split_bone_into_quadrants_using_vectors: splits a bone into quadrants based on two arbitrary axes provided by the user, with splits including (+axis1, +axis2), (+axis1, -axis2), (-axis1, -axis2), (-axis1, +axis2).

import numpy as np
import vtk
from mesh_processing import create_point

def half_mesh(mesh, mean, direction, is_pca) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function divides the mesh into two parts based on the sign of the projection along a given direction.

    Args:
        mesh (vtk.vtkPolyData): The mesh that is being divided.
        mean (numpy array): The mean point for projection.
        direction (numpy array): The direction vector for projection.

    Returns:
        indices_above_mean (numpy array): The indices of the points above the mean.
        indices_below_mean (numpy array): The indices of the points below the mean.
        points_above_mean (numpy array): The points above the mean.
        points_below_mean (numpy array): The points below the mean.
        colours (numpy array): The colours of the points based on their projection.
    """
    if is_pca:
        projections = np.dot(mesh.points - mean, direction)
    else:
        projections = np.dot(mesh.points - mean, direction.T)

    indices_above_mean = np.where(projections >= 0)[0]
    indices_below_mean = np.where(projections < 0)[0]

    points_above_mean = mesh.points[indices_above_mean]
    points_below_mean = mesh.points[indices_below_mean]

    # Creating separate colours for the positive and negative projections
    colours = np.zeros((mesh.n_points, 3))
    colours[projections >= 0] = [1, 0, 1] # Magenta for above mean
    colours[projections < 0] = [0, 1, 1] # Cyan for below mean

    return indices_above_mean, indices_below_mean, points_above_mean, points_below_mean, colours

def divide_mesh_by_pca_into_thirds(mesh, pca_mean, pca_component) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Divides the mesh into three parts based on projection along a principal component direction,
    assigning specific colors to each third.

    Args:
        mesh (vtk.vtkPolyData): The mesh that is being divided.
        pca_mean (numpy array): The mean of the principal component.
        pca_component (numpy array): The principal component.
        
    Returns:
        split_indices (list of numpy arrays): List of indices for each division.
        split_points (list of numpy arrays): List of points for each division.
        split_colours (numpy array): Colours for visualization of each division.
    """

    projections = np.dot(mesh.points - pca_mean, pca_component)
    min_proj, max_proj = np.min(projections), np.max(projections) # Getting the minimum and maximum points along the direction of the principal component
    division_points = np.linspace(min_proj, max_proj, 4)  # Three divisions require four boundary points

    # Define specific colors for each third
    colors = [
        [1.0, 0.0, 0.0], # Red
        [1.0, 1.0, 0.0], # Yellow
        [0.0, 1.0, 0.0] # Green
    ]

    split_indices = []
    split_points = []
    split_colours = np.zeros((mesh.n_points, 3))

    # Dividing the mesh into three sections and assigning colors to each section
    for i in range(3):
        # Find indices within the current division range
        indices = np.where((projections >= division_points[i]) & (projections < division_points[i + 1]))[0]
        split_indices.append(indices)
        split_points.append(mesh.points[indices])

        # Assign the specific color for the current third
        split_colours[indices] = colors[i]

    return split_indices, split_points, split_colours

def divide_mesh_into_sections(mesh, centroid, axis, divisions) -> tuple[list, list, np.ndarray]:
    """
    This function splits a mesh into multiple divisions along a specified principal axis.
    This is similar to dividing the mesh by PCA into thirds, but allows for an arbitrary number of divisions. (Could adjust the previous function to call this one with divisions = 3)

    Args:
        mesh (vtk.vtkPolyData): The mesh being divided.
        centroid (numpy array): The centroid of the mesh.
        principal_axis (numpy array): The principal axis along which the mesh is divided.
        divisions (int): Number of divisions.

    Returns:
        division_indices (list): Indices of points in each division.
        division_points (list): Points in each division.
        division_labels (numpy array): Integer labels for visualization.
    """ 
    projections = np.dot(mesh.points - centroid, axis)
    min_dist, max_dist = np.min(projections), np.max(projections)
    division_points = np.linspace(min_dist, max_dist, divisions + 1) 

    division_labels = np.zeros(mesh.n_points) 

    divided_indices = []
    divided_points = []

    for i in range(divisions):
        indices = np.where((projections >= division_points[i]) & (projections < division_points[i + 1]))[0]
        divided_indices.append(indices)
        divided_points.append(mesh.points[indices])
        division_labels[indices] = i 


    return divided_indices, divided_points, division_labels

def compute_bone_centroid(mesh) -> tuple[vtk.vtkSphereSource, vtk.vtkSphereSource]:
    """This function will determine the centroid of a bone mesh. That is, the center of mass assuming uniform density.
    This will be done with two methods: the first will be the mean of all points, and the second will use the vtk.vtkCenterOfMass class.

    Args:
        mesh (vtk.vtkPolyData): The input bone surface mesh.

    Returns:
        com_point (vtk.vtkSphereSource): The center of mass point.
        com_point_vtk (vtk.vtkSphereSource): The center of mass point using vtk.vtkCenterOfMass.
    """
    # Center of mass calculation 1
    centroid = mesh.points.mean(axis=0)
    centroid_point = create_point(centroid, 0.75)

    # Center of mass calculation 2
    centroid_filter = vtk.vtkCenterOfMass()
    centroid_filter.SetInputData(mesh)
    centroid_filter.SetUseScalarsAsWeights(False)
    centroid_filter.Update()
    centroid_vtk = centroid_filter.GetCenter()
    centroid_point_vtk = create_point(centroid_vtk, 0.75)

    return centroid_point, centroid_point_vtk

def split_bone_into_quadrants(bone, inertial_axes, origin, axis_division_1, axis_division_2) -> tuple[list, list, list]:
    """This function will split a bone into quadrants based on the principal inertial axes.
    This split will occur based on the two principal inertial axes. (E.g. second and third principal axes = the Y and Z axes)
    Splits include: (+Y, +Z), (+Y, -Z), (-Y, -Z), (-Y, +Z)

    Args:
        bone (dict): The dictionary containing the bone information.
        inertial_axes (numpy array): The principal inertial axes of the bone.
        origin (numpy array): The origin of the bone.

    Returns:
        split_indices (list): The indices of the points in each quadrant.
        split_points (list): The points in each quadrant.
        split_colours (list): The colours of the points in each quadrant.
    """
    projections = np.dot(bone["Mesh"].points - origin, inertial_axes)
    x_projections = np.dot(bone["Mesh"].points - origin, inertial_axes[0])
    y_projections = np.dot(bone["Mesh"].points - origin, inertial_axes[1])
    z_projections = np.dot(bone["Mesh"].points - origin, inertial_axes[2])

    # Choosing from any of the three axes based on user input
    if axis_division_1 == "X" and axis_division_2 == "Y":
        projection_axis_1 = x_projections
        projection_axis_2 = y_projections
    elif axis_division_1 == "X" and axis_division_2 == "Z":
        projection_axis_1 = x_projections
        projection_axis_2 = z_projections
    elif axis_division_1 == "Y" and axis_division_2 == "Z":
        projection_axis_1 = y_projections
        projection_axis_2 = z_projections
    else:
        raise ValueError("Invalid axis division specified. Please specify the axes to divide the bone into quadrants.\n Options: two of X, Y, Z") # Raising an error in case of invalid input

    colours = ["red", "blue", "green", "yellow"] # Colours of the four quadrants
    split_indices, split_points, split_colours = [], [], []
    for i in range(4):
        # Quadrant 1: +axis1, +axis2 - e.g. +Y, +Z will be red
        if i == 0:
            indices = np.where((projection_axis_1 > 0) & (projection_axis_2 > 0))
        # Quadrant 2: +axis1, -axis2
        elif i == 1:
            indices = np.where((projection_axis_1 > 0) & (projection_axis_2 < 0))
        # Quadrant 3: -axis1, +axis2
        elif i == 2:
            indices = np.where((projection_axis_1 < 0) & (projection_axis_2 > 0))
        # Quadrant 4: -axis1, -axis2
        elif i == 3:
            indices = np.where((projection_axis_1 < 0) & (projection_axis_2 < 0))

        split_indices.append(indices)
        split_points.append(bone["Mesh"].points[indices])
        split_colours.append(colours[i])

    return split_indices, split_points, split_colours

def split_bone_into_quadrants_using_vectors(bone, centroid, axis_division_1, axis_division_2) -> tuple[list, list, list]:
    """This function will split a bone into quadrants based on two arbitrary axes.
    This split will occur based on the two axes provided by the user.
    Splits include: (+axis1, +axis2), (+axis1, -axis2), (-axis1, -axis2), (-axis1, +axis2)
    This is similar to the previous function but allows for arbitrary axes to be used instead of just the principal inertial axes.

    Args:
        bone (dict): The dictionary containing the bone information.
        centroid (numpy array): The centroid of the bone.
        axis_division_1 (numpy array): The first axis to divide the bone.
        axis_division_2 (numpy array): The second axis to divide the bone.

    Returns:
        split_indices (list): The indices of the points in each quadrant.
        split_points (list): The points in each quadrant.
        split_colours (list): The colours of the points in each quadrant.
    """ 
    projections_1 = np.dot(bone["Mesh"].points - centroid, axis_division_1)
    projections_2 = np.dot(bone["Mesh"].points - centroid, axis_division_2)

    colours = ["red", "blue", "green", "yellow"]
    split_indices, split_points, split_colours = [], [], []
    for i in range(4):
        # Quadrant 1: +axis1, +axis2
        if i == 0:
            indices = np.where((projections_1 > 0) & (projections_2 > 0))
        # Quadrant 2: +axis1, -axis2
        elif i == 1:
            indices = np.where((projections_1 > 0) & (projections_2 < 0))
        # Quadrant 3: -axis1, +axis2
        elif i == 2:
            indices = np.where((projections_1 < 0) & (projections_2 > 0))
        # Quadrant 4: -axis1, -axis2
        elif i == 3:
            indices = np.where((projections_1 < 0) & (projections_2 < 0))

        split_indices.append(indices)
        split_points.append(bone["Mesh"].points[indices])
        split_colours.append(colours[i])

    return split_indices, split_points, split_colours
