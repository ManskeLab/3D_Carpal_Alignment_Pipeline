# This file contains functions to calculate angles between vectors (3D and 2D) and distances between 3D meshes using different algorithms.
# This includes:
# - calculate_3D_angle: Calculates the angle between two 3D vectors.
# - calculate_2D_angle: Calculates the angle between two 2D vectors.
# - ray_tracing_distance: Calculates the minimum distance between two meshes using ray tracing.
# - nearest_neighbour_distance: Calculates the minimum distance between two meshes using nearest neighbour search.
# - pv_closest_points_distance: Calculates the minimum distance between two meshes using PyVista's find_closest_point function.

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

def calculate_3D_angle(sc_vector, lu_vector):
    """This function calculates the angle between two vectors in 3D space.

    Args:
        sc_vector (numpy array): The vector representing the axis of one bone.
        lu_vector (numpy array): The vector representing the axis of another bone.

    Returns:
        angle (float): The angle between the two vectors in degrees.
    """
    sc_vector = sc_vector / np.linalg.norm(sc_vector)
    lu_vector = lu_vector / np.linalg.norm(lu_vector)
    sc_lun_dot_product = np.dot(sc_vector, lu_vector)
    angle_radians = np.arccos(sc_lun_dot_product)
    angle = np.degrees(angle_radians)
    return angle

def calculate_2D_angle(sc_2d_vector, lu_2d_vector):
    """This function calculates the angle between two vectors in 2D space.

    Args:
        sc_2d_vector (numpy array): The vector representing an axis projected into the 2D plane.
        lu_2d_vector (numpy array): The vector representing an axis projected into the 2D plane.

    Returns:
        angle (float): The angle between the two vectors in degrees.
    """
    sc_vector = sc_2d_vector / np.linalg.norm(sc_2d_vector)
    lu_vector = lu_2d_vector / np.linalg.norm(lu_vector)
    sc_lun_dot_product = np.dot(sc_vector, lu_vector)
    angle_radians = np.arccos(sc_lun_dot_product)
    angle = np.degrees(angle_radians)
    return angle

def ray_tracing_distance(mesh_1, mesh_2):    
    """This function calculates the minimum distance between two meshes using a ray tracing algorithm. 
    This takes into account the surface normals of the first mesh to calculate the distance to the second mesh.
    This is one of three ways to measure distances between two meshes. It takes the longest to compute but generally does not change much based on different mesh parameters.

    Args:
        mesh_1 (pv.PolyData): This is the first mesh that is passed in for distance calculation.
        mesh_2 (pv.PolyData): This is the second mesh that is passed in for distance calculation.

    Returns:
        min_distance (float): This function returns the minimum distance between the two meshes.
    """
    normal_vecs = mesh_1.compute_normals(point_normals = True, cell_normals = False, auto_orient_normals = True)

    normal_vecs["Intervals"] = np.empty(normal_vecs.n_points)
    for i in range(normal_vecs.n_points):
        p = mesh_1.points[i]
        vector = normal_vecs["Normals"][i] * normal_vecs.length
        p0 = p - vector
        p1 = p + vector
        ip, ic = mesh_2.ray_trace(p0, p1, first_point = True)
        dist = np.sqrt(np.sum((ip - p) ** 2))
        normal_vecs["Intervals"][i] = dist
    
    # Replace zeros with NaNs
    mask = normal_vecs["Intervals"] == 0
    normal_vecs["Intervals"][mask] = np.nan
    np.nanmean(normal_vecs["Intervals"])

    # Printing the minimum distance between the two meshes
    min_distance = np.nanmin(normal_vecs["Intervals"])

    # Creating a line between the two points of minimum distance - starting and ending on the two meshes
    min_distance_index = np.nanargmin(normal_vecs["Intervals"])
    print(min_distance_index)
    p = mesh_1.points[min_distance_index]
    vector = normal_vecs["Normals"][min_distance_index] * normal_vecs.length
    p0 = p - vector
    p1 = p + vector
    min_distance_line = pv.Line(p0, p1)


    p = pv.Plotter()
    p.add_mesh(normal_vecs, scalars = "Intervals", smooth_shading = True)
    p.add_mesh(mesh_2, color = True, opacity = 0.5, smooth_shading = True)
    p.add_mesh(min_distance_line, color = "red")
    p.show()

    return min_distance, min_distance_line

def nearest_neighbour_distance(mesh_1, mesh_2):
    """This function calculates the minimum distance between two meshes using a nearest neighbour algorithm
    and displays points within 10% of the minimum distance on the mesh.
    This is one of three ways to measure distances between two meshes. It is the fastest to compute but can be affected by mesh parameters.
    
    Args:
        mesh_1 (pv.PolyData): The first mesh for distance calculation.
        mesh_2 (pv.PolyData): The second mesh for distance calculation.

    Returns:
        min_distance (float): The minimum distance between the two meshes.
        min_distance_line (pv.Line): Line between the closest points on the two meshes.
        meshes_list (list): List of the two meshes, the line, and the points within 10% of the minimum distance.
    """
    # Perform nearest neighbor query
    tree = KDTree(mesh_2.points)
    dist, index = tree.query(mesh_1.points)
    mesh_1["Distances"] = dist
    min_distance = np.min(dist)

    # Define the 1% threshold
    # This serves as a simple way to visualise points that are close to the minimum distance
    threshold = min_distance * 1.01
    mask = (dist >= min_distance) & (dist <= threshold)  # Mask for distances within the threshold

    # Extract points within the threshold
    close_points = mesh_1.points[mask]
    masked_mesh = pv.PolyData(close_points)

    # Identify and create a line for the minimum distance
    min_distance_index = np.argmin(dist)
    p0 = mesh_1.points[min_distance_index]
    p1 = mesh_2.points[index[min_distance_index]]
    min_distance_line = pv.Line(p0, p1)
    min_distance_tube = pv.Line(p0, p1).tube(radius = 0.1)

    meshes_list = [mesh_1, mesh_2, masked_mesh] # The mesh where distances were calculated, the target mesh, and the points within 1% of the minimum distance
    return min_distance, min_distance_tube, meshes_list

def pv_closest_points_distance(mesh_1, mesh_2):
    """This function calculates the minimum distance between two meshes using the find_closest_point function in PyVista
    The documentation for this function is provided here: https://docs.pyvista.org/examples/01-filter/distance_between_surfaces.html -- it relies on the pv.find_closest_cell function.

    Args:
        mesh_1 (pv.PolyData): This is the first mesh that is passed in for distance calculation.
        mesh_2 (pv.PolyData): This is the second mesh that is passed in for distance calculation.

    Returns:
        _min_distance (float): This function returns the minimum distance between the two meshes.
    """
    closest_cells, closest_points = mesh_2.find_closest_cell(mesh_1.points, return_closest_point = True)
    dist_exact = np.linalg.norm(mesh_1.points - closest_points, axis = 1)
    mesh_1["Distances"] = dist_exact
    mean = np.mean(dist_exact)
    min_distance = np.min(dist_exact)

    p = pv.Plotter()
    p.add_mesh(mesh_1, scalars = "Distances", smooth_shading = True)
    p.add_mesh(mesh_2, color = True, opacity = 0.5, smooth_shading = True)
    p.show()

    return min_distance