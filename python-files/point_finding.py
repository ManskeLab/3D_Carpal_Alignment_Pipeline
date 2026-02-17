# point_finding.py
# contains functions to find specific points on a mesh:
# - find_max_point_in_axis: finds the point with the maximum projection along a certain axis (x, y, or z)
# - find_largest_euclidean_distance: finds the point with the largest Euclidean distance from the mean (centroid) of the mesh
# - find_intersection_point_along_pc: finds the point on the surface that is closest to the projection of a principal component axis
# - find_max_point_along_axis: finds the point with the maximum projection along a certain principal component direction
# - find_max_percent_of_points_along_axis: finds the points with the maximum projection along a certain principal component direction, considering only a certain percentage of points
# - find_max_point_along_axis_in_search_radius: finds the point with the maximum projection along a certain principal component direction, considering only points within a certain search radius
# - find_max_point_along_pc_in_search_radius: finds the point with the maximum projection along a principal component direction, considering only points within a certain search radius of an original point
# - compute_bone_centroid: determines the centroid of a bone mesh using two methods: the mean of all points and the vtk.vtkCenterOfMass class
# - find_closest_point_on_mesh: finds the closest point on a mesh to a given point

import numpy as np
import vtk
from scipy.spatial import KDTree
from mesh_processing import create_point, build_mesh_adjacency_list, find_points_within_k_steps

def find_max_point_in_axis(points, axis) -> tuple[int, np.ndarray]:
    """This function will find the maximum point along the x-, y-, or z- axis.

    Args:
        points (vtk.vtkPoints): The points of the mesh (can be the entire mesh or a subset of points, like the points above or below the mean in a certain principal component direction).
        axis (int): The axis along which the maximum point is being found (0 for x, 1 for y, 2 for z).

    Returns:
        max_point_index (int): The index of the maximum point.
    """

    max_point_index = np.argmax(points[:, axis])
    max_point = points[max_point_index]

    return max_point_index, max_point

def find_largest_euclidean_distance(points, mean) -> tuple[int, np.ndarray]:
    """This function finds the point on the mesh that has the largest Euclidean distance from the mean (centroid) of the mesh.

    Args:
        points (numpy array): The points of the mesh.
        mean (numpy array): The mean of the mesh.

    Returns:
        max_distance_index (int): The index of the point with the largest distance.
        max_distance_point (numpy array): The point with the largest distance.
    """

    distances = np.linalg.norm(points - mean, axis = 1)
    max_distance_index = np.argmax(distances)
    max_distance_point = points[max_distance_index]

    return max_distance_index, max_distance_point

def find_intersection_point_along_pc(points, pca_mean, pca_component) -> tuple[np.ndarray, int]:
    """
    This function finds the point on the surface that is closest to the projection of a principal component axis.

    Args:
        points (numpy array): The points of the mesh.
        pca_mean (numpy array): The mean of the principal component.
        pca_component (numpy array): The principal component direction.

    Returns:
        closest_point (numpy array): The point on the surface that is closest to the principal component axis.
        closest_point_index (int): The index of the closest point.
    """
    # Normalize the principal component vector (PCA axis)
    pca_axis = pca_component / np.linalg.norm(pca_component)

    # Function to calculate the distance from a point to the PCA line
    def distance_to_pca_line(point, pca_mean, pca_axis):
        # The distance formula for a point to a line
        vec_to_point = point - pca_mean
        cross_product = np.cross(vec_to_point, pca_axis)
        distance = np.linalg.norm(cross_product) / np.linalg.norm(pca_axis)
        return distance

    # Find the point with the minimum distance to the PCA axis
    min_distance = float('inf')
    closest_point_index = -1

    for i, point in enumerate(points):
        distance = distance_to_pca_line(point, pca_mean, pca_axis)
        if distance < min_distance:
            min_distance = distance
            closest_point_index = i

    closest_point = points[closest_point_index]
    return closest_point, closest_point_index

def find_max_point_along_axis(points, centroid, axis) -> np.ndarray:
    """This function finds the point with the maximum projection along a certain principal component direction.

    Args:
        points (vtk.vtkPoints): The points of the mesh.
        centroid (numpy array): The centroid of the mesh. This essentially acts as the starting point of the search.
        axis (numpy array): The principal component direction along which the search is being conducted. 

    Returns:
        max_point (numpy array): The point with the maximum projection along the principal component direction.
    """
    points = np.array(points)
    centroid = np.array(centroid)
    axis = np.array(axis)
    
    projections = np.dot(points - centroid, axis)
    max_projection_index = np.argmax(projections)
    max_point = points[max_projection_index]
    return max_point

def find_max_percent_of_points_along_axis(points, centroid, axis, percent) -> list:
    """This function finds the point with the maximum projection along a certain principal component direction,
    considering only a certain percentage of points.

    Args:
        points (vtk.vtkPoints): The points of the mesh.
        centroid (numpy array): The centroid of the mesh. This essentially acts as the starting point of the search.
        axis (numpy array): The principal component direction along which the search is being conducted.
        percent (float): The percentage of points to consider for the maximum projection.

    Returns:
        max_points (list): A list of the maximum points along a given axis
    """
    points = np.array(points)
    centroid = np.array(centroid)
    axis = np.array(axis)

    projections = np.dot(points - centroid, axis)
    num_points_to_consider = int(len(points) * percent)
    
    # Get indices of the top 'percent' points based on projections
    top_indices = np.argsort(projections)[-num_points_to_consider:]
    max_points = points[top_indices]
    return max_points

def find_max_point_along_axis_in_search_radius(mesh, original_point, search_radius, principal_inertial_axis) -> np.ndarray:
    """Similar to find_max_point_along_axis, but only considers points within a certain search radius.

    Args:
        points (vtk.vtkPoints): The points of the mesh. 
        original_point (numpy array): The point around which the search is being conducted. This is intended to be near the radial side of the lunate horns
        search_radius (int): The radius within which the search is being conducted.
        principal_inertial_axis (numpy array): The principal inertial axis of the mesh.
    
    Returns:
        max_point (numpy array): The point with the maximum projection along the principal inertial axis within the search radius.
        """
    adjacency = build_mesh_adjacency_list(mesh)
    indices_within_steps = find_points_within_k_steps(adjacency, original_point, search_radius)
    
    num_points = mesh.GetNumberOfPoints()
    valid_indices = [i for i in indices_within_steps if 0 <= i < num_points]

    # # Debug: Check if any invalid indices were filtered
    # if len(valid_indices) != len(indices_within_steps):
    #     print(f"Filtered {len(indices_within_steps) - len(valid_indices)} invalid indices.")

    # Extract coordinates of valid neighboring points
    neighbours = np.array([np.array(mesh.GetPoint(i)) for i in valid_indices])

    # Get the coordinates of the original point
    original_point_coords = np.array(mesh.GetPoint(original_point))

    # Compute projections of neighbors onto the principal inertial axis
    projections = np.dot(neighbours - original_point_coords, principal_inertial_axis)

    # Find the neighbor with the maximum projection
    max_projection_index = np.argmax(projections)
    max_point = neighbours[max_projection_index]

    return max_point

def find_max_point_along_pc_in_search_radius(mesh, original_point, search_radius, pca_mean, pca_component) -> np.ndarray:
    """This function finds the point with the maximum projection along a principal component direction,
    considering only points within a certain search radius of an original point.
    Args:
        mesh (vtk.vtkPolyData): The mesh that is being analyzed.
        original_point (numpy array): The point around which the search is being conducted.
        search_radius (int): The radius within which the search is being conducted.
        pca_mean (numpy array): The mean of the principal component.
        pca_component (numpy array): The principal component direction. 
    Returns:
        max_point (numpy array): The point with the maximum projection along the principal component direction within the search radius.
    """
    tree = KDTree(mesh.points)
    indices = tree.query_ball_point(original_point, search_radius)
    points = mesh.points[indices]
    max_point = find_max_point_along_axis(points, pca_mean, pca_component)
    return max_point

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

def find_closest_point_on_mesh(search_mesh,  search_point) -> tuple[int, np.ndarray]:
    """This function finds the closest point on a mesh to a given point.    
    Args:
        search_mesh (vtk.vtkPolyData): The mesh that is being searched.
        search_point (numpy array): The point that is being searched for.   
    Returns:
        closest_point_id (int): The index of the closest point on the mesh.
        closest_point (numpy array): The coordinates of the closest point on the mesh.
    """
    # Needed? Convert the numpy array to a vtk array
    # search_point = numpy_to_vtk(search_point)
    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(search_mesh)
    point_locator.BuildLocator()

    closest_point_id = point_locator.FindClosestPoint(search_point)
    closest_point = search_mesh.GetPoint(closest_point_id)
    return closest_point_id, np.array(closest_point)