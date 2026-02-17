# mesh_processing.py
# contains various basic mesh processing functions that are used for larger components of the project:
# - create_point: creates a point at a specific location (used for visualization purposes and for having the proper format in certain setups).
# - principal_component_analysis: performs principal component analysis on a set of data to help define 3D shape of a bone mesh.
# - compute_derivatives: computes the first and second derivatives of a numpy array (used for curvature calculations).
# - extract_mesh_from_vtk: extracts the vertices and faces from a VTK polydata object (used for converting VTK meshes into a format that can be used for certain calculations).
# - calculate_vertex_normals: calculates the vertex normals of a mesh (used for several purposes including ray tracing distance measurements and curvature calculations).
# - smooth_mesh: smooths a mesh using the Laplacian smoothing algorithm (used to smooth meshes after decimation and to smooth curvature maps for better region growing).
# - build_mesh_adjacency_list: converts a vtkPolyData mesh into an adjacency list to help traverse the mesh for various applications such as region growing (finding the closest neighbours to a vertex).
# - find_points_within_k_steps: finds all points within "k" steps of a starting point in an adjacency list using a breadth-first searching approach (useful for searching the mesh for neighbouring points and is used in curvature calculations).
# - grow_region: grows a region starting from a point, only including points in a set of valid indices (used for growing regions of negative curvature in curvature maps).
# - compute_mass_moment_of_inertia: computes the mass moment of inertia tensor and principal axes for a given VTK polydata (used to find the principal axes of a bone mesh which can be used for alignment and angle calculations).


import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from sklearn.decomposition import PCA
import pyvista as pv
from collections import defaultdict

def create_point(centre, radius) -> vtk.vtkSphereSource:
    """This creates a point at a specific location (used for visualization purposes).

    Args:
        centre (numpy array): The coordinates of the centre of the point.
        radius (float): The radius of the point.

    Returns:
        point (vtk.vtkSphereSource): The VTK sphere source.
    """
    point = vtk.vtkSphereSource()
    point.SetCenter(centre)
    point.SetRadius(radius)
    point.SetPhiResolution(32)
    point.SetThetaResolution(32)
    point.Update()

    return point

def principal_component_analysis(np_data) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """This function will perform principal component analysis on a set of data.
    This helps to define 3D shape of a bone mesh.

    Args:
        np_data (numpy array): The numpy array of the data.

    Returns:
        pca_components (numpy array): The principal components of the data.
        pca_means (numpy array): The means of the data.
        pca_variance (numpy array): The variance of the data.
        component_colours (list): The colours of the principal components.
    """
    pca_components = []
    pca_means = []
    pca_variance = []
    for img in np_data:
        pca = PCA(n_components = 3)
        pca.fit(img)
        pca_components.append(pca.components_)
        pca_means.append(pca.mean_)
        pca_variance.append(pca.explained_variance_ratio_)
        
    component_colours = ['red', 'green', 'blue']
    return pca_components, pca_means, pca_variance, component_colours

def compute_derivatives(np_data) -> tuple[np.ndarray, np.ndarray]:
    """This function computes the first and second derivatives of a numpy array.   
    Args:   
        np_data (numpy array): The numpy array of the data.                     
    Returns:
        first_derivatives (numpy array): The first derivatives of the data.
        second_derivatives (numpy array): The second derivatives of the data.
    """
    first_derivatives = np.gradient(np_data, axis = 0)
    second_derivatives = np.gradient(first_derivatives, axis = 0)
    return first_derivatives, second_derivatives    

def extract_mesh_from_vtk(polydata) -> tuple[np.ndarray, np.ndarray]:
    """This function extracts the vertices and faces from a VTK polydata object.
    Args:
        polydata (vtk.vtkPolyData): The VTK polydata object to extract from.
    Returns:
        vertices (numpy array): The vertices of the mesh.
        faces (numpy array): The faces of the mesh.
    """

    points = polydata.GetPoints()
    vertices = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])

    faces = []
    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        ids = cell.GetPointIds()
        faces.append([ids.GetId(j) for j in range(ids.GetNumberOfIds())])

    face_list = []
    for face in faces:
        face_list.append(len(face))
        face_list.extend(face)
    face_array = np.array(face_list, dtype=np.int32)

    return vertices, face_array       

def calculate_vertex_normals(mesh) -> np.ndarray:
    """This function calculates the vertex normals of a mesh.
    The normals are used for several purposes including ray tracing distance measurements and curvature calculations.
    
    Args:
        mesh (vtk.vtkPolyData): The mesh to calculate normals for.

    Returns:
        vertex_normals (numpy array): The vertex normals of the mesh.
    """
    mesh = pv.wrap(mesh).clean(tolerance = 1e-8) #this was added to clean the mesh and remove any duplicate points that may cause issues with normal calculation
    face_normals = mesh.face_normals
    vertex_normals = np.zeros_like(mesh.points)
    for i, cell in enumerate(mesh.faces.reshape(-1, 4)):
        for j in cell[1:]:
            vertex_normals[j] += face_normals[i]
    norm = np.linalg.norm(vertex_normals, axis = 1)
    vertex_normals = (vertex_normals.T / norm).T # Normalizing the vertex normals

    return vertex_normals

def smooth_mesh(mesh, iterations, relaxation_factor) -> pv.PolyData:
    """This function will smooth a mesh using the Laplacian smoothing algorithm.

    Args:
        mesh (vtk.vtkPolyData): The mesh that is being smoothed.
        iterations (int): The number of iterations for which the mesh is being smoothed.

    Returns:
        _type_: _description_
    """
    mesh = pv.wrap(mesh).clean(tolerance = 1e-8)
    mesh = mesh.smooth(n_iter = iterations, relaxation_factor = relaxation_factor, feature_smoothing = False, boundary_smoothing = True)

    return mesh

def decimate_mesh(mesh, target_reduction) -> pv.PolyData:
    """This function will decimate a mesh using the vtkDecimatePro algorithm.

    Args:
        mesh (vtk.vtkPolyData): The mesh that is being decimated.
        target_reduction (float): The target reduction for the decimation (between 0 and 1).

    Returns:
        decimated_mesh (vtk.vtkPolyData): The decimated mesh.
    """
    decimated_mesh = mesh.decimate(target_reduction = target_reduction)
    return decimated_mesh

def build_mesh_adjacency_list(mesh) -> dict:
    """This function converts a vtkPolyData mesh into an adjacency list.
    This helps to traverse the mesh for various applications such as region growing (finding the closest neighbours to a vertex)

    Args:
        mesh (vtk.vtkPolyData): The mesh that is being converted.

    Returns:
        adjacency_list (dict): The adjacency list of the mesh (keys are point indices and values are lists of neighboring point indices).
    """
    adjacency_list = defaultdict(list)
    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        for j in range(cell.GetNumberOfPoints()):
            p1 = cell.GetPointId(j)
            p2 = cell.GetPointId((j + 1) % cell.GetNumberOfPoints())
            adjacency_list[p1].append(p2)
            adjacency_list[p2].append(p1)

    # Debug: Print adjacency list size and a small sample
    # print("Adjacency list size:", len(adjacency_list))
    # print("Sample adjacency entries:", dict(list(adjacency_list.items())[:5]))

    return adjacency_list

def find_points_within_k_steps(adjacency_list, start_point_index, k) -> set:
    """This function finds all points within "k" steps of a starting point in an adjacency list. 
    A breadth-first searching approach is used.
    This is useful for searching the mesh for neighbouring points and is used in curvature calculations.

    Args:
        adjacency_list (dict): The adjacency list of the mesh.
        start_point_index (int): The index of the starting point.
        k (int): The number of steps to search.

    Returns:
        points_within_k_steps (list): The list of points within "k" steps of the starting point.
    """
    visited = set()
    queue = [(start_point_index, 0)]
    points_within_k_steps = set()

    while queue:
        current, depth = queue.pop(0)
        if current in visited or depth > k:
            continue
        visited.add(current)
        points_within_k_steps.add(current)
        for neighbour in adjacency_list[current]:
            if neighbour not in visited:
                queue.append((neighbour, depth + 1))

    return points_within_k_steps

def grow_region(adjacency_list, start_point, valid_indices) -> set:
    """
    Grows a region starting from `start_point`, only including points in `valid_indices`.

    Args:
        adjacency_list (dict): Adjacency list of the mesh.
        start_point (int): Index of the starting point.
        valid_indices (set): Set of valid indices (negative curvature points).

    Returns:
        region (set): Set of indices that form the connected negative curvature region.
    """
    region = set()
    queue = [start_point]

    while queue:
        current = queue.pop(0)
        if current in region:
            continue
        region.add(current)

        # Only add neighbors that are also in the negative curvature set
        for neighbor in adjacency_list[current]:
            if neighbor in valid_indices and neighbor not in region:
                queue.append(neighbor)

    return region

def compute_mass_moment_of_inertia(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the mass moment of inertia tensor and principal axes for a given VTK polydata.

    Args:
        polydata (vtk.vtkPolyData): The VTK mesh data.
        density (float): Density of the material (assumed uniform).

    Returns:
        inertia_tensor (np.ndarray): 3x3 mass moment of inertia tensor.
        eigenvalues (np.ndarray): Principal moments of inertia.
        eigenvectors (np.ndarray): Principal axes (eigenvectors of the tensor).
    """
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    n_points = points.shape[0]

    # Calculate the center of mass
    center_of_mass = np.mean(points, axis=0)

    # Translate points relative to the center of mass
    r = points - center_of_mass

    # Initialize inertia tensor
    inertia_tensor = np.zeros((3, 3))

    for i in range(n_points):
        x, y, z = r[i]
        inertia_tensor[0, 0] += (y**2 + z**2)  # I_xx, equal to the sum of all (y^2 + z^2) terms
        inertia_tensor[1, 1] += (x**2 + z**2)  # I_yy, equal to the sum of all (x^2 + z^2) terms
        inertia_tensor[2, 2] += (x**2 + y**2)  # Izz, equal to the sum of all (x^2 + y^2) terms

        inertia_tensor[0, 1] -= x * y  # I_xy, equal to the sum of all (x*y) terms; negative to account for negative in inertia tensor formula
        inertia_tensor[0, 2] -= x * z  # I_xz, equal to the sum of all (x*z) terms; negative to account for negative in inertia tensor formula
        inertia_tensor[1, 2] -= y * z  # I_yz, equal to the sum of all (y*z) terms; negative to account for negative in inertia tensor formula

    # Symmetry in the inertia tensor
    inertia_tensor[1, 0] = inertia_tensor[0, 1] #I_xy = I_yx
    inertia_tensor[2, 0] = inertia_tensor[0, 2] #I_xz = I_zx
    inertia_tensor[2, 1] = inertia_tensor[1, 2] #I_yz = I_zy

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)

    return inertia_tensor, eigenvalues, eigenvectors
