# curvature.py
# contains functions to calculate curvatures of a mesh:
# - calculate_curvatures: Calculates the signed principal curvatures, mean curvature, and Gaussian curvature of a mesh using the second fundamental form. This uses a neighborhood defined by adjacency steps to estimate the curvatures at each vertex. This approach helps to account for potential noise in the mesh data.
# - normalize_curvatures: Normalizes the curvatures to a range of 0 to 1. This is done to ensure that the curvatures are on the same scale for further analysis.


import numpy as np
import vtk
from scipy.spatial import KDTree    
from mesh_processing import build_mesh_adjacency_list, find_points_within_k_steps

def calculate_curvatures(mesh, normals, search_steps) -> np.ndarray:
    """Calculates the signed principal curvatures, mean curvature, and Gaussian curvature of a mesh using the second fundamental form.
    This uses a neighborhood defined by adjacency steps to estimate the curvatures at each vertex. This approach helps to account for potential noise in the mesh data.
    This curvature calculation utilized the following papers/documentation as references:
        - "Estimating Curvatures and Their Derivatives on Triangle Meshes" by Szymon Rusinkiewicz (2004) (https://gfx.cs.princeton.edu/pubs/Rusinkiewicz_2004_ECA/curvpaper.pdf)
        - "Surface Reconstruction from Unorganized Points" by Hoppe et al. (1992) (https://graphics.pixar.com/library/Reconstruction/paper.pdf)
        - vtkCurvatures Class Reference (https://vtk.org/doc/nightly/html/classvtkCurvatures.html#details)

    Args:
        mesh (vtk.vtkPolyData): The mesh to analyze.
        normals (numpy array): Array of shape (n_points, 3) representing the normal vectors at each point.
        adjacency_list (dict): Adjacency list of the mesh (keys are point indices, values are neighbors).
        search_steps (int): The number of adjacency steps to search for neighbors.

    Returns:
        curvatures (numpy array): Array of shape (n_points, 4) containing: k1, k2, mean curvature, and Gaussian curvature. (k1 and k2 are the principal curvatures)
    """
    print("Calculating the curvatures...")

    # Build adjacency list for the mesh (getting the closest vertices)
    adjacency_list = build_mesh_adjacency_list(mesh)
    num_points = mesh.GetNumberOfPoints()
    curvatures = np.zeros((num_points, 4))  # Setting up array to hold k1, k2, mean curvature, Gaussian curvature

    for i in range(num_points): # Iterate through each point in the mesh and calculate the curvature
        neighbor_indices = find_points_within_k_steps(adjacency_list, i, search_steps) # Find the closest neighbors within the specified number of adjacency steps (k = search_steps)

        if len(neighbor_indices) < 5: # Accounts for when there is not enough neighbours around a point (particularly helpful for edges of an unclosed mesh like the radius)
            continue

        # Step 1: Gather neighborhood points
        # The neighbour points within the specified number of adjacency steps are calculated here, and they are all expressed in terms of the center point
        neighbours = np.array([mesh.GetPoint(idx) for idx in neighbor_indices])
        center_point = np.array(mesh.GetPoint(i))
        centered_neighbours = neighbours - center_point

        # Step 2: Compute local frame using PCA - this gives the tangent plane at a given point which helps define the curvature at a given point
        cov_matrix = np.cov(centered_neighbours.T) # This helps to explain how the points are distributed around the center point during principal component analysis
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # This is not always needed but kept for clarity -- the normal is in the direction perpendicular to the tangent plane
        normal = eigenvectors[:, 0]  
        if np.dot(normal, normals[i]) < 0: # Ensure the eigenvector direction matches the provided normal
            normal = -normal
        tangent_u, tangent_v = eigenvectors[:, 1], eigenvectors[:, 2]  # The other two eigenvectors form the local tangent plane

        # Step 3: Estimate the second fundamental form (II) coefficients
        II = np.zeros((2, 2)) # This matrix defines how the normal changes along the tangent directions (tangent_u and tangent_v)
        for idx in neighbor_indices:
            if idx == i: # Skip the center point itself
                continue
            neighbor_point = np.array(mesh.GetPoint(idx))
            displacement = neighbor_point - center_point # The displacement vector from the center point to the neighbor point

            # Getting the displacement between the center point and the neighbor point in terms of the tangent directions
            u_proj = np.dot(displacement, tangent_u) 
            v_proj = np.dot(displacement, tangent_v)

            # Calculating the difference in normals between the center point and the neighbor point - and subsequently projecting that onto the tangent directions to get the curvatures
            normals_change = normals[idx] - normals[i]
            u_curvature = np.dot(normals_change, tangent_u)
            v_curvature = np.dot(normals_change, tangent_v)

            # Update the second fundamental form matrix
            II[0, 0] += u_proj * u_curvature # Change along the u direction due to curvature in the u direction
            II[0, 1] += u_proj * v_curvature 
            II[1, 0] += v_proj * u_curvature
            II[1, 1] += v_proj * v_curvature # Change along the v direction due to curvature in the v direction

        II /= len(neighbor_indices) # Averaging over the number of neighbors

        # Calculate the eigenvalues of II for principal curvatures
        principal_curvatures, _ = np.linalg.eigh(II)
        k1, k2 = principal_curvatures

        # Populating the curvatures array with the principal curvatures, mean curvature, and Gaussian curvature
        curvatures[i, 0] = k1 # First principal curvature
        curvatures[i, 1] = k2 # Second principal curvature
        curvatures[i, 2] = (k1 + k2) / 2  # Mean curvature
        curvatures[i, 3] = k1 * k2 # Gaussian curvature

    return curvatures

def normalize_curvatures(curvatures) -> np.ndarray:
    '''This function normalizes the curvatures to a range of 0 to 1.
    This is done to ensure that the curvatures are on the same scale for further analysis.

    Args:
        curvatures (numpy array): The array of curvature values calculated using the calculate_curvatures function.
    Returns:
        normalized_curvatures (numpy array): The normalized curvatures.
    '''
    # Getting the maximum and minimum of k1 and k2
    min_k1 = np.min(curvatures[:, 0])
    max_k1 = np.max(curvatures[:, 0])
    min_k2 = np.min(curvatures[:, 1])
    max_k2 = np.max(curvatures[:, 1])

    min_mean = np.min(curvatures[:, 2])
    max_mean = np.max(curvatures[:, 2])
    min_gaussian = np.min(curvatures[:, 3])
    max_gaussian = np.max(curvatures[:, 3])

    # Normalizing the curvatures using the minimum and maximum values
    normalized_curvatures = np.zeros_like(curvatures)
    normalized_curvatures[:, 0] = (curvatures[:, 0] - min_k1) / (max_k1 - min_k1)
    normalized_curvatures[:, 1] = (curvatures[:, 1] - min_k2) / (max_k2 - min_k2)
    normalized_curvatures[:, 2] = (curvatures[:, 2] - min_mean) / (max_mean - min_mean)
    normalized_curvatures[:, 3] = (curvatures[:, 3] - min_gaussian) / (max_gaussian - min_gaussian)

    return normalized_curvatures
