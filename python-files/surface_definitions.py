#surfaces_def.py
# contains functions are typically used to define articulating surfaces between two meshes, particularly relevant to measuring the scapholunate interval at multiple locations:
# - compute_oriented_bounding_box: This function computes the oriented bounding box (OBB) for a given set of points. Open3D was used for this as it provides a straightforward method to compute the OBB; other libraries were not consistent in this measurement.
# - create_box_lines: This function creates the edges of the oriented bounding box from the given vertices. The edge names are just for the SL model as done for the lunate, but can be applied readily to any bounding box.
# - define_articulating_surface: This function is very similar to the define_lunate_articulating_surface function, but it is more general and can be applied to any two meshes. The articulating surface is defined as the points on the first mesh that are within a certain threshold distance of the second mesh. This follows a similar technique to the methodology outlined by Teule et al. (2024) for measuring positional scapholunate intervals.
# - define_lunate_articulating_surface: This function defines the lunate articluating surface and returns its oriented bounding box, midpoints, and distances to the scaphoid. The articulating surface is defined as the points on the lunate mesh that are within a certain threshold distance of the scaphoid mesh, and where the angle between the normal vectors of the two meshes is within a certain range. This follows a similar technique to the methodology outlined by Teule et al. (2024) for measuring positional scapholunate intervals.
# - create_plane_and_normal: This function creates a plane from the given vertices and calculates the normal vector of the plane. This is used for visualizing the orientation of the bounding box and articulating surface.

import pyvista as pv
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

# These functions are typically used to define articulating surfaces between two meshes, particularly relevant to measuring the scapholunate interval at multiple locations.

def create_box_lines(box_vertices) -> tuple[list[pv.PolyData], list[pv.PolyData], np.ndarray]:
    """
    Create the edges of the oriented bounding box from the given vertices.
    
    Args:
        box_vertices (np.ndarray): The vertices of the oriented bounding box.
        
    Returns:
        box_lines (pv.PolyData): A PolyData object representing the edges of the box.
    """
    # Define the edges by connecting the vertices to form the box
    # The edge names are just for the SL model as done for the lunate, but can be applied readily to any bounding box
    ulnar_proximal_edge = [2, 7]
    ulnar_palmar_edge = [7, 1]
    ulnar_distal_edge = [1, 0]
    ulnar_dorsal_edge = [0, 2]

    radial_proximal_edge = [5, 4]
    radial_palmar_edge = [4, 6]
    radial_distal_edge = [6, 3]
    radial_dorsal_edge = [3, 5]

    proximal_dorsal_edge = [2, 5]
    proximal_palmar_edge = [7, 4]
    distal_dorsal_edge = [1, 6]
    distal_palmar_edge = [0, 3]
    
    edges = [
        ulnar_proximal_edge, ulnar_palmar_edge, ulnar_distal_edge, ulnar_dorsal_edge, # The edges that make up the ulnar face
        radial_proximal_edge, radial_palmar_edge, radial_distal_edge, radial_dorsal_edge, # The edges that make up the radial face
        proximal_dorsal_edge, proximal_palmar_edge, distal_dorsal_edge, distal_palmar_edge # The edges that connect the ulnar and radial faces
    ]
    
    # Create the PolyData object for the lines
    lines = []
    for edge in edges:
        lines.append([2, edge[0], edge[1]])  # 2 means two points define a line
    lines = np.array(lines)

    # Create separate lines for the ulnar, radial, and connecting edges
    ulnar_lines = lines[0:4]
    radial_lines = lines[4:8]
    connecting_lines = lines[8:12]
    ulnar_box_lines = pv.PolyData(box_vertices)
    ulnar_box_lines.lines = ulnar_lines
    radial_box_lines = pv.PolyData(box_vertices)
    radial_box_lines.lines = radial_lines
    connecting_box_lines = pv.PolyData(box_vertices)
    connecting_box_lines.lines = connecting_lines

    # Calculating the midpoints of the edges
    midpoints = []
    for edge in edges:
        midpoint = (box_vertices[edge[0]] + box_vertices[edge[1]]) / 2
        midpoints.append(midpoint)
    midpoints_list = np.array(midpoints)

    ulnar_midpoints = midpoints[0:4]
    radial_midpoints = midpoints[4:8]
    connecting_midpoints = midpoints[8:12]

    ulnar_midpoints = pv.PolyData(ulnar_midpoints)
    radial_midpoints = pv.PolyData(radial_midpoints)
    connecting_midpoints = pv.PolyData(connecting_midpoints)

    box_lines = [ulnar_box_lines, radial_box_lines, connecting_box_lines]
    edge_midpoints = [ulnar_midpoints, radial_midpoints, connecting_midpoints]
    
    return box_lines, edge_midpoints, midpoints_list

def define_articulating_surface(bone_1, bone_2, articulation_threshold, min_angle, max_angle) -> tuple[pv.PolyData, np.ndarray, pv.PolyData, np.ndarray, pv.PolyData, pv.PolyData, pv.PolyData, pv.PolyData, pv.PolyData, list, np.ndarray, list, int]:
    """This function is very similar to the define_lunate_articulating_surface function, but it is more general and can be applied to any two meshes.
    The articulating surface is defined as the points on the first mesh that are within a certain threshold distance of the second mesh.
    This follows a similar technique to the methodology outlined by Teule et al. (2024) for measuring positional scapholunate intervals.

    Args:
        bone_1 (vtk.vtkPolyData): The mesh where the articulating surface will be defined.
        bone_2 (vtk.vtkPolyData): The mesh that will be used as a reference for the articulating surface.
        articulation_threshold (float): The threshold distance that will be used to define the articulating surface.
        min_angle (float): The minimum angle between the normal vectors of the two meshes.
        max_angle (float): The maximum angle between the normal vectors of the two meshes.
        box_selection (int): This will be 0, 1, or 2, and will determine which box to select for the articulating surface. 0 = X-axis, 1 = Y-axis, 2 = Z-axis.

    Returns:
        original_surface_mesh (pv.PolyData): The original surface mesh of the articulating surface.
        articulating_surface (np.ndarray): The points on the articulating surface.
        surface_mesh (pv.PolyData): The surface mesh of the articulating surface.
        box_vertices (np.ndarray): The vertices of the oriented bounding box.
        box_lines (pv.PolyData): The lines of the oriented bounding box.
        edge_midpoints (pv.PolyData): The midpoints of the edges of the bounding box.
        central_point (pv.PolyData): The central point of the bounding box.
        midpoints (pv.PolyData): The midpoints between the edge midpoints and the central point.
        closest_points (pv.PolyData): The closest points on the articulating surface mesh to the midpoints.
        min_distance_lines (list): The lines connecting the midpoints to the closest points.
        distances (np.ndarray): The distances between the midpoints and the closest points.
        positions (list): The positions of the midpoints.
    """
    # Creating a KDTree for one mesh to determine points that are within the articulation threshold of the other mesh
    tree = KDTree(bone_2["Mesh"].points)
    dist, index = tree.query(bone_1["Mesh"].points)
    mask = (dist <= articulation_threshold) # Create a mask for points within the articulation threshold (e.g. 2.5 mm for the scapholunate articulation)

    # Extract the points within the threshold
    articulating_surface = bone_1["Mesh"].points[mask]
    original_surface_mesh = pv.PolyData(articulating_surface)

    # Calculate the normal vectors at the points on the articulating surface and the closest point on the other bone
    bone_1_normals = bone_1["Mesh"].point_normals[mask]
    bone_2_normals = bone_2["Mesh"].point_normals[index[mask]]
    dot_products = np.sum(bone_1_normals * bone_2_normals, axis = 1)
    angles = np.arccos(dot_products)
    mask = (angles >= np.radians(min_angle)) & (angles <= np.radians(max_angle))
    articulating_surface = articulating_surface[mask]
    surface_mesh = pv.PolyData(articulating_surface)

    # The oriented bounding box is a 3D box that takes up the minimum volume around a set of points (the points within the articulating surface)
    box_vertices = compute_oriented_bounding_box(articulating_surface)
    box_lines, edge_midpoints, midpoints_list = create_box_lines(box_vertices)

    # Get the central point of the each group of 4 in the midpoints list (0-3, 4-7, 8-11)
    # This is done by taking the mean of the points in each group
    central_points = []
    for i in range(3):
        temp_center_point = np.mean(edge_midpoints[i].points, axis = 0)
        central_points.append(temp_center_point)

    # Get the distance between each point in central points and the centroid of the bone_1 mesh
    centroid_to_central_point_distance = 0
    box_index = 0
    for i in range(3):
        distance = np.linalg.norm(central_points[i] - bone_1["Mesh"].points.mean(axis = 0))
        if distance > centroid_to_central_point_distance:
            centroid_to_central_point_distance = distance
            box_index = i
    
    # The central point of the box to be used for the articulating surface measurements
    central_point = np.mean(edge_midpoints[box_index].points, axis = 0)
    central_point = pv.PolyData(central_point.reshape(1, -1))

    # Calculating the midpoints of the edges
    midpoints = []
    for edge in edge_midpoints[box_index].points:
        midpoint = (edge + central_point.points[0]) / 2
        midpoints.append(midpoint)

    midpoints = np.array(midpoints)
    midpoints = pv.PolyData(midpoints)

    closest_points = []
    for midpoint in midpoints.points:
        closest_point = surface_mesh.points[surface_mesh.find_closest_point(midpoint)]
        closest_points.append(closest_point)

    # The new central point will be the average of the proximal and distal points. That is, the mean position between closest_points[0] and closest_points[2]
    # The combats the issue where there is more of a slope on the lunate surface, causing the central point overlapping with the distal point
    # This is an adjustment from Teule et al.'s (2024) methodology
    adjusted_centre_point = np.mean([closest_points[0], closest_points[2]], axis = 0)
    closest_point_to_centre = surface_mesh.points[surface_mesh.find_closest_point(adjusted_centre_point)]
    closest_points.append(closest_point_to_centre)
    closest_points_np = np.array(closest_points)
    closest_points = pv.PolyData(closest_points_np)

    adjusted_midpoints = []
    for edge in edge_midpoints[box_index].points:
        adjusted_midpoint = (edge + closest_point_to_centre) / 2
        adjusted_midpoints.append(adjusted_midpoint)
    adjusted_midpoints = np.array(adjusted_midpoints)
    adjusted_midpoints = pv.PolyData(adjusted_midpoints)
    
    adjusted_closest_points = []
    for adjusted_midpoint in adjusted_midpoints.points:
        adjusted_closest_point = surface_mesh.points[surface_mesh.find_closest_point(adjusted_midpoint)]
        adjusted_closest_points.append(adjusted_closest_point)
    adjusted_closest_points.append(closest_point_to_centre)
    adjusted_closest_points_np = np.array(adjusted_closest_points)
    adjusted_closest_points = pv.PolyData(adjusted_closest_points)

    positions = ["Proximal", "Palmar", "Distal", "Dorsal", "Central"] # This is just for the SL model, the only position that is correct for any surface is the central point. Need to fix this.

    tree = KDTree(bone_2["Mesh"].points)
    distances, indices = tree.query(closest_points.points)
    closest_points["Distance to Scaphoid"] = distances

    min_distance_lines = []
    for i in range(len(adjusted_closest_points.points)):
        min_distance_line = pv.Line(adjusted_closest_points.points[i], bone_2["Mesh"].points[indices[i]]).tube(radius = 0.1)
        min_distance_lines.append(min_distance_line)

    return original_surface_mesh, articulating_surface, surface_mesh, box_vertices, box_lines, edge_midpoints, central_point, midpoints, adjusted_closest_points_np, min_distance_lines, distances, positions, midpoints_list, box_index

def define_lunate_articulating_surface(lunate, scaphoid, articulation_threshold, min_angle, max_angle) -> tuple[pv.PolyData, np.ndarray, pv.PolyData, np.ndarray, pv.PolyData, pv.PolyData, pv.PolyData, pv.PolyData, pv.PolyData, list, np.ndarray, list]:
    """This function defines the lunate articluating surface and returns its oriented bounding box, midpoints, and distances to the scaphoid.

    Args:
        lunate (dict): The lunate mesh and associated data.
        scaphoid (dict): The scaphoid mesh and associated data.
        articulation_threshold (float): The threshold distance to define the articulating surface.
        min_angle (int): The minimum angle between the normal vectors of the two meshes.
        max_angle (int): The maximum angle between the normal vectors of the two meshes.

    Returns:
        original_surface_mesh (pv.PolyData): The original surface mesh of the lunate articulating surface.
        lunate_articulating_surface (np.ndarray): The points on the lunate articulating surface.
        surface_mesh (pv.PolyData): The surface mesh of the lunate articulating surface.
        box_vertices (np.ndarray): The vertices of the oriented bounding box.
        box_lines (pv.PolyData): The lines of the oriented bounding box.
        edge_midpoints (pv.PolyData): The midpoints of the edges of the bounding box.
        central_point (pv.PolyData): The central point of the bounding box.
        midpoints (pv.PolyData): The midpoints between the edge midpoints and the central point.
        closest_points (pv.PolyData): The closest points on the lunate articulating surface mesh to the midpoints.
        min_distance_lines (list): The lines connecting the midpoints to the closest points.
        distances (np.ndarray): The distances between the midpoints and the closest points.
        positions (list): The positions of the midpoints.
    """
    # Calculate the nearest neighbour distances between the scaphoid and lunate
    tree = KDTree(scaphoid["Mesh"].points)
    dist, index = tree.query(lunate["Mesh"].points)
    mask = (dist <= articulation_threshold)

    # Extract the points within the threshold
    lunate_articulating_surface = lunate["Mesh"].points[mask]
    original_surface_mesh = pv.PolyData(lunate_articulating_surface)

    # Calculate the normal vectors at the points on the lunate articulating surface and the closest point on the scaphoid
    lunate_normals = lunate["Mesh"].point_normals[mask]
    scaphoid_normals = scaphoid["Mesh"].point_normals[index[mask]]
    dot_products = np.sum(lunate_normals * scaphoid_normals, axis = 1)
    angles = np.arccos(dot_products)
    mask = (angles >= np.radians(min_angle)) & (angles <= np.radians(max_angle))
    lunate_articulating_surface = lunate_articulating_surface[mask]
    surface_mesh = pv.PolyData(lunate_articulating_surface)
    
    box_vertices = compute_oriented_bounding_box(lunate_articulating_surface, lunate)
    box_lines, edge_midpoints = create_box_lines(box_vertices)

    # Getting the central point between all of the edge midpoints in edge_midpoints[1]
    central_point = np.mean(edge_midpoints[1].points, axis = 0)
    central_point = pv.PolyData(central_point.reshape(1, -1))

    # Getting the four midpoints between the edge midpoints and the central point
    midpoints = []
    for edge in edge_midpoints[1].points:
        midpoint = (edge + central_point.points[0]) / 2
        midpoints.append(midpoint)

    midpoints = np.array(midpoints)
    midpoints = pv.PolyData(midpoints)

    # Getting the closest points on the lunate articulating surface mesh to the midpoints
    closest_points = []
    for midpoint in midpoints.points:
        closest_point = surface_mesh.points[surface_mesh.find_closest_point(midpoint)]
        closest_points.append(closest_point)
    closest_point_to_centre = surface_mesh.points[surface_mesh.find_closest_point(central_point.points[0])]
    closest_points.append(closest_point_to_centre)
    closest_points = np.array(closest_points)
    closest_points = pv.PolyData(closest_points)

    positions = ["Proximal", "Palmar", "Distal", "Dorsal", "Central"]

    # For each point in closest_points, calculate the shortest distance to the scaphoid mesh using the KDTree
    tree = KDTree(scaphoid["Mesh"].points)
    distances, indices = tree.query(closest_points.points)
    closest_points["Distance to Scaphoid"] = distances
    
    min_distance_lines = []
    for i in range(len(closest_points.points)):
        min_distance_line = pv.Line(closest_points.points[i], scaphoid["Mesh"].points[indices[i]]).tube(radius = 0.1)
        min_distance_lines.append(min_distance_line)
        

    return original_surface_mesh, lunate_articulating_surface, surface_mesh, box_vertices, box_lines, edge_midpoints, central_point, midpoints, closest_points, min_distance_lines, distances, positions

def create_plane_and_normal(plane_vertices) -> tuple[pv.PolyData, pv.PolyData, np.ndarray, np.ndarray, pv.PolyData]:
    edges = [[0, 1], [1, 3], [3, 2], [2, 0]]
    lines = []
    for edge in edges:
        lines.append([2, edge[0], edge[1]])
    lines = np.array(lines)
    box = pv.PolyData(plane_vertices, lines = lines)
    faces = np.array([4, 0, 1, 3, 2])
    plane = pv.PolyData(plane_vertices, faces = faces)
    plane_centre = plane.center
    edge_1 = plane_vertices[1] - plane_vertices[0]
    edge_2 = plane_vertices[3] - plane_vertices[0]
    plane_normal = np.cross(edge_1, edge_2)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    plane_normal_arrow = pv.Arrow(start = plane_centre, direction = plane_normal, scale = 2.5)
    return box, plane, plane_centre, plane_normal, plane_normal_arrow

def compute_oriented_bounding_box(points) -> np.ndarray:
    """
    Computes the oriented bounding box (OBB) for a given set of points.
    Open3D was used for this as it provides a straightforward method to compute the OBB; other libraries were not consistent in this measurement.

    Args:
        points (np.ndarray): The 3D coordinates of the points.

    Returns:
        box_vertices (np.ndarray): The vertices of the oriented bounding box.
        box_edges (np.ndarray): The edges of the bounding box for visualization.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    obb = point_cloud.get_oriented_bounding_box()
    obb_vertices = np.asarray(obb.get_box_points())

    return obb_vertices