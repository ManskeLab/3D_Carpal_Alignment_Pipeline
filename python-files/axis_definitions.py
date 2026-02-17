#axes_def.py
# contains functions to define axes for the radius, scaphoid, and lunate bones. 
# These include:
# - get_radial_axes: Defines the radial coordinate system based on anatomical landmarks and PCA of the radius mesh.
# - get_scaphoid_axis: Generates an axis for the scaphoid bone based on its proximal and distal poles.
# - get_lunate_axis: Constructs a guide axis for the lunate bone using its articulating surface and principal inertial axes.
# - get_capitate_axis: Defines the capitate bone axis based on its principal inertial axes and curvature analysis.
# - get_mc3_axis: Defines the third metacarpal bone axis using its principal inertial axes.
# - adjust_scaphoid_principal_axes: Adjusts the scaphoid's principal axes to ensure correct anatomical orientation.
# - adjust_lunate_principal_axes: Adjusts the lunate's principal axes to ensure correct anatomical orientation.
# - adjust_capitate_principal_axes: Adjusts the capitate's principal axes to ensure correct anatomical orientation.
# - adjust_mc3_principal_axes: Adjusts the third metacarpal's principal axes to ensure correct anatomical orientation.

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA

from mesh_processing import create_point
from calc import calculate_3D_angle
from point_finding import find_max_point_along_axis, find_max_percent_of_points_along_axis, find_max_point_along_axis_in_search_radius, find_max_point_along_pc_in_search_radius, compute_bone_centroid, find_closest_point_on_mesh
from mesh_lines import get_mesh_lines, count_line_mesh_interactions, adjust_line
from mesh_division import split_bone_into_quadrants
from mesh_processing import compute_mass_moment_of_inertia, grow_region, build_mesh_adjacency_list
from surface_definitions import define_articulating_surface, create_box_lines, compute_oriented_bounding_box, create_plane_and_normal, define_lunate_articulating_surface
SCAPHOID_LENGTH_MULTIPLIER = 1.5
LUNATE_LENGTH_MULTIPLIER = 1.25

def get_radial_axes(radius, mesh, ulna, shaft_coordinate, radial_ulnar_coordinate, palmar_dorsal_coordinate, visualize) -> tuple[pv.Line, np.ndarray, pv.Line, np.ndarray, pv.Line, np.ndarray]:
    """The three axes will be defined based on Coburn's 2007 paper / Kobayashi's 1997 paper on the coordinate systems for the carpal bones.
    X-axis: A line fit through the centroids of the radial shaft. The axis coincides with the radial long axis. Pronation corresponds to positive rotation.
    Y-axis: Directed through the radial styloid and perpendicular to the X-axis. Positive values are along the radial side of the wrist, with positive rotation corresponding to (palmar) flexion.
    Z-axis: Directed palmarly and the cross product of the X- and Y- axes, with the positive direction in the palmar direction. Positive rotation corresponds to ulnar deviation.
    Origin: The intersection of the X-axis and the distal radial articular surface.

    Args:
        radius (dict): The dictionary containing the radius information. This includes the mesh of the radius, the PCA components, and the PCA means (and really everything else).
        mesh (pv.PolyData): The mesh of the radius.
        ulna (dict): The dictionary containing the ulna information. This includes the mesh of the ulna, the PCA components, and the PCA means (and really everything else).
        shaft_coordinate (str): The coordinate to use as an initial placement for the shaft length ("X", "Y", or "Z").
        radial_ulnar_coordinate (str): The coordinate to use as an estimate for the radial-ulnar direction ("X", "Y", or "Z").
        palmar_dorsal_coordinate (str): The coordinate to use as an estimate for the palmar-dorsal direction ("X", "Y", or "Z").
        visualize (bool): Whether to visualize the axes and points.

    Returns:
        x_axis_line (pv.Line): The line representing the X-axis.
        x_axis_vector (numpy array): The normalized vector representing the +X-axis.
        y_axis_line (pv.Line): The line representing the Y-axis.
        y_axis_vector (numpy array): The normalized vector representing the +Y-axis.
        z_axis_line (pv.Line): The line representing the Z-axis.
        z_axis_vector (numpy array): The normalized vector representing the +Z-axis.
    """
    #for debugging - set radial_styloid_point to None early
    radial_styloid_point = None


    # Creating a line fit through the centroids (best-fit line/principal component axis) of the radial shaft
    # This requires a definition of the direction that the shaft runs along such that there is a reference for the X-axis of the radius
    # If longer lengths of the radius are available, this could be replaced by a principal component analysis in future studies.
    if shaft_coordinate == "X": coord = 0
    elif shaft_coordinate == "Y": coord = 1
    elif shaft_coordinate == "Z": coord = 2
    shaft_length_coordinates = radius["Mesh"].points[:, coord]
    sorted_shaft_length_coordinates = np.sort(shaft_length_coordinates)
    num_slices = 100 # Splitting the radius into 100 unique slices
    shaft_length_min, shaft_length_max = np.min(sorted_shaft_length_coordinates), np.max(sorted_shaft_length_coordinates) # Setting the minimum and maximum slices for the shaft points to be evaluated for the radial (X) axis
    slice_height = np.linspace(shaft_length_min, shaft_length_max, num_slices) # Slicing the shaft into 100 slices
    centroids = []
    # Take the centroid of points within the 10-75% of the shaft length to avoid shifts in the centroid due to the distal end of the radius
    for idx in slice_height[int(0.1 * num_slices):int(0.75 * num_slices)]:
        points = radius["Mesh"].points[np.abs(radius["Mesh"].points[:, coord] - idx) < 1]
        centroids.append(np.mean(points, axis = 0))
    centroids = np.array(centroids)

    # # Display the centroids using PyVista
    # centroid_points = pv.PolyData(centroids)
    # p = pv.Plotter()
    # p.add_mesh(radius["Mesh"], color = "white", opacity = 0.5, smooth_shading = True)
    # p.add_mesh(centroid_points, color = "red", point_size = 10)
    # p.show_grid()
    # p.show()

    # Running a principal component analysis through the centroids (to get the best fit line through the centroids of the radial shaft)
    pca = PCA(n_components=1) # The PCA library from sklearn is used to get the principal component axis
    pca.fit(centroids)
    radius_x_vector = -pca.components_[0] # The first principal component is the direction of the radial shaft (X-axis)
    radius_x_vector /= np.linalg.norm(radius_x_vector) # Normalizing the X-axis vector

    projections = np.dot(centroids - centroids.mean(axis = 0), radius_x_vector) # Projecting the centroids onto the X-axis vector
    sorted_indices = np.argsort(projections)
    sorted_centroids = centroids[sorted_indices]
    middle_index = int(0.5 * len(sorted_centroids)) # Getting the middle index of the sorted centroids - this is just used as a seed point for creating the mesh of the line
    middle_point = sorted_centroids[middle_index]

    # Extending the line fit through the radius shaft from the middle point by 100 mm in both directions
    extended_start_point = middle_point - radius_x_vector * 100
    extended_end_point = middle_point + radius_x_vector * 100
    # Creating a point at the intersection of the X-axis and the distal radial articular surface using ray tracing
    start_point = np.array(extended_start_point)
    end_point = np.array(extended_end_point)
    points, intersected_cells = mesh.ray_trace(start_point, end_point)
    if len(points) > 0:
        intersection_point = points[0]
    else: # Raising an error in the case that the x-axis line does not intersect the radius mesh
        raise ValueError("No intersection point found. Please adjust the line fit through the radius shaft.")
    
    # Create a point at the origin of the radius coordinate system -- the intersection of the X-axis and the distal radial articular surface
    origin_point = create_point(intersection_point, 0.5)
    origin = origin_point.GetCenter()
    # The line below can be uncommented to output the location of the origin point
    # print("Origin point:", origin)
    
    # The X-axis will be defined along the radius shaft, running from distal to proximal - it will start at the origin and move proximal along the x-vector (down the radius shaft)
    x_axis_start_point = origin_point.GetCenter()
    x_axis_arrow = pv.Arrow(start = x_axis_start_point, direction = radius_x_vector, scale = 10)
    x_axis_line = pv.Line(x_axis_start_point + radius_x_vector * 40, x_axis_start_point - radius_x_vector * 25)
    x_axis_tube = x_axis_line.tube(radius = 0.1)

    # The Y-axis will be directed through the radial styloid and perpendicular to the X-axis
    # The radial styloid is the point that occurs furthest radially at the section of the radius that has the largest width in the x-direction
    # Note: Right now, this code utilizes global coordinates, which is not ideal. In future codes, it can be modified to use local coordinates. For now, this was found to be the best way to identify the radial styloid.
    if radial_ulnar_coordinate == "X": ru_coord = 0
    elif radial_ulnar_coordinate == "Y": ru_coord = 1
    elif radial_ulnar_coordinate == "Z": ru_coord = 2
    radius_ulna_distance = np.array(ulna["PCA_Means"]) - origin
    radial_ulnar_coordinates = radius["Mesh"].points[:, ru_coord]
    sorted_radial_ulnar_indices = np.argsort(radial_ulnar_coordinates)
    sorted_radial_ulnar_coordinates = radial_ulnar_coordinates[sorted_radial_ulnar_indices]

    if palmar_dorsal_coordinate == "X": pd_coord = 0
    elif palmar_dorsal_coordinate == "Y": pd_coord = 1
    elif palmar_dorsal_coordinate == "Z": pd_coord = 2
    pd_coordinates = radius["Mesh"].points[:, pd_coord]
    sorted_pd_indices = np.argsort(pd_coordinates)
    sorted_pd_coordinates = pd_coordinates[sorted_pd_indices]

    # Getting all points of the radius mesh and then determining the radial styloid point based on the following criteria:
    #   - The point must be the furthest from the X-axis line (maximum distance from the projection onto the X-axis)
    #   - The point must be in the radial direction (that is, in the opposite direction as the radius-ulna coordinate)
    #   - The point must be near the centre of the bone in the palmar-dorsal direction (such that it is actually on the radial styloid and not shifted too far dorsal or palmar)
    radius_points = radius["Mesh"].points
    max_distance = 0
    diff = 0
    for point in radius_points:
        vector_to_point = point - origin
        projection_length = np.dot(vector_to_point, radius_x_vector)
        projection_point = origin + projection_length * radius_x_vector
        distance = np.linalg.norm(point - projection_point)

        angle = calculate_3D_angle(radius_x_vector, vector_to_point)
        angle_diff = abs(angle - 90)

        # Replace if the distance is greater than the max distance AND the point is radial (that is, in the same direction as the radius-scaphoid x-distance)
        # The vector to the point should also be at a right angle to the radius x-vector, which is indicated by a dot product of close to 0
        # The mean curvature should also be less than 0, as the radial styloid is a local peak
        # The point should also be in the middle 33% of the z-width of the radius to ensure that the y-axis is not shifted too dorsal or palmar
        if distance > max_distance and np.dot(vector_to_point, radius_ulna_distance) < 0 and angle_diff < 1 and point[0] > sorted_radial_ulnar_coordinates[int(90 * len(sorted_radial_ulnar_coordinates)/100)] and point[2] > sorted_pd_coordinates[int(33 * len(sorted_pd_coordinates)/100)] and point[2] < sorted_pd_coordinates[int(67 * len(sorted_pd_coordinates)/100)]:
            max_distance = distance
            radial_styloid_point = point
            diff = angle_diff

    # The code below can be uncommented to output the coordinates of the radial styloid point
    # print("Radial styloid point:", radial_styloid_point)

    # Creating a line and an arrow for the Y-axis
    radial_styloid_point = create_point(radial_styloid_point, 0.5)
    radial_styloid_vector = np.array(radial_styloid_point.GetCenter()) - np.array(origin_point.GetCenter()) # This is the vector from the origin to the radial styloid point

    # This is the projection of the radial styloid vector onto the plane perpendicular to the X-axis
    # Subtracting the radial styloid vector's projection onto the X-axis from the radial styloid vector gives us the vector in the direction of the Y-axis
    # This ensures that the y-axis is orthogonal to the x-axis
    styloid_vector_projection = radial_styloid_vector - np.dot(radial_styloid_vector, radius_x_vector) * radius_x_vector 
    radius_y_vector = styloid_vector_projection / np.linalg.norm(styloid_vector_projection) # # This is the normalized vector in the direction of the Y-axis
    y_axis_start_point = origin
    y_axis_arrow = pv.Arrow(start = y_axis_start_point, direction = radius_y_vector, scale = 10)

    # The Z-axis will be directed palmarly and the cross product of the X- and Y- axes
    radius_z_vector = np.cross(radius_x_vector, radius_y_vector)
    radius_z_vector = radius_z_vector / np.linalg.norm(radius_z_vector)
    z_axis_arrow = pv.Arrow(start = origin, direction = radius_z_vector, scale = 10)


    if visualize == True:
        p = pv.Plotter(window_size=(600,600))
        p.add_mesh(radius["Mesh"], color = "white", opacity = 0.8, smooth_shading = True)
        p.add_mesh(x_axis_tube, color = "red", line_width = 5)
        p.add_mesh(y_axis_arrow, color = "green", line_width = 5)
        p.add_mesh(z_axis_arrow, color = "blue", line_width = 5)
        p.add_mesh(origin_point, color = "black", point_size = 10)
        p.add_mesh(radial_styloid_point, color = "yellow", point_size = 10)
        p.add_mesh(x_axis_arrow, color = "red", line_width = 5)
        p.show_grid()
        camera_position = "xy"
        p.camera_position = camera_position
        p.show()

    return origin_point, radius_x_vector, radius_y_vector, radius_z_vector, x_axis_tube

    # return origin_point, radius_x_vector, x_axis_arrow, radial_styloid_point, radius_y_vector, y_axis_arrow, radius_z_vector, z_axis_arrow, x_axis_line, x_axis_tube

def get_scaphoid_axis(scaphoid, visualize) -> tuple[pv.Line, pv.Tube, np.ndarray, np.ndarray, pv.Arrow, list]:
    """This function generates an axis for the scaphoid. This line should run along the palmar aspects of the proximal and distal poles of the scaphoid.

    Args:
        scaphoid (dict): The dictionary containing the scaphoid information. 

    Returns:
        scaphoid_line (pv.Line): The line representing the scaphoid axis.
        scaphoid_tube (pv.Tube): The tube representing the scaphoid axis.
        scaphoid_vector (numpy array): The vector representing the scaphoid axis.
        scaphoid_normalized_vector (numpy array): The normalized vector representing the scaphoid axis.
        scaphoid_arrow (pv.Arrow): The arrow representing the scaphoid axis.
        scaphoid_intersection_points (list): The list of intersection points between the scaphoid axis and the scaphoid mesh.
    """
    # Get initial points along the distal and proximal poles of the scaphoid
    # The (negative) second principal axis will always generally point towards the palmar side of the scaphoid, and the scaphoid has already been split into three components (proximal pole, waist, and distal pole)
    scaphoid_distal_point = find_max_point_along_axis(scaphoid["Split_Points"][2], scaphoid["Centroid"].GetCenter(), -scaphoid["Principal_Axis_2"])
    scaphoid_proximal_point = find_max_point_along_axis(scaphoid["Split_Points"][0], scaphoid["Centroid"].GetCenter(), -scaphoid["Principal_Axis_2"])
    scaphoid_distal_point_id = scaphoid["Mesh"].find_closest_point(scaphoid_distal_point)
    scaphoid_proximal_point_id = scaphoid["Mesh"].find_closest_point(scaphoid_proximal_point)

    scaphoid_distal_point = create_point(scaphoid_distal_point, 0.5)
    scaphoid_proximal_point = create_point(scaphoid_proximal_point, 0.5)

    # # The two scaphoid points must be checked for the following two conditions: the mean curvature must be below 0, and the Gaussian curvature must be greater than 0 (-Mean Curvature, +Gaussian Curvature)
    # # This is because the two points are on the distal and proximal poles respectively, and these lie on local peaks of the bone
    # if scaphoid["Mesh"].point_data["Mean_Curvature"][scaphoid["Mesh"].find_closest_point(scaphoid_distal_point)] < 0 and scaphoid["Mesh"].point_data["Gaussian_Curvature"][scaphoid["Mesh"].find_closest_point(scaphoid_distal_point)] > 0:
    #     print("Original placement of the distal point is accurate.")
    #     scaphoid_distal_point = create_point(scaphoid_distal_point, 0.5)
    # else: # Search around the initial placement of the distal point to find a point that satisfies the curvature conditions
    #     print("Position of distal point is not accurate. Refinement is underway.")
    #     k = 2
    #     while not ( # While the curvature conditions are not met, keep searching
    #         scaphoid["Mesh"].point_data["Mean_Curvature"][scaphoid["Mesh"].find_closest_point(scaphoid_distal_point)] < 0 and
    #         scaphoid["Mesh"].point_data["Gaussian_Curvature"][scaphoid["Mesh"].find_closest_point(scaphoid_distal_point)] > 0
    #     ):
    #         if k == 10: # This breaks the loop in the case that the search radius deviates too far from the original point -- the algorithm will generally find this point within k = 3
    #             break
    #         print("Search radius: ", k) # Sign post to indicate the search radius
    #         # Get the scaphoid distal point within the search radius that has the maximum value along the principal axis 3
    #         scaphoid_distal_point = find_max_point_along_axis_in_search_radius(scaphoid["Mesh"], scaphoid_distal_point_id, k, scaphoid["Principal_Axis_3"])
    #         scaphoid_distal_point_id = scaphoid["Mesh"].find_closest_point(scaphoid_distal_point)
    #         k = k + 1
    #     scaphoid_distal_point = create_point(scaphoid_distal_point, 0.5)
    #     # k = k + 1  
    # if scaphoid["Mesh"].point_data["Mean_Curvature"][scaphoid["Mesh"].find_closest_point(scaphoid_proximal_point)] < 0 and scaphoid["Mesh"].point_data["Gaussian_Curvature"][scaphoid["Mesh"].find_closest_point(scaphoid_proximal_point)] > 0:
    #     print("Original placement of the proximal point is accurate.")
    #     scaphoid_proximal_point = create_point(scaphoid_proximal_point, 0.5)
    # else: # Search around the initial placement of the proximal point to find a point that satisfies the curvature conditions
    #     print("Position of proximal point is not accurate. Refinement is underway.")
    #     k = 2
    #     while not (
    #         scaphoid["Mesh"].point_data["Mean_Curvature"][scaphoid["Mesh"].find_closest_point(scaphoid_proximal_point)] < 0 and
    #         scaphoid["Mesh"].point_data["Gaussian_Curvature"][scaphoid["Mesh"].find_closest_point(scaphoid_proximal_point)] > 0
    #     ):
    #         if k == 10:
    #             break
    #         print("Search radius: ", k)
    #         scaphoid_proximal_point = find_max_point_along_axis_in_search_radius(scaphoid["Mesh"], scaphoid_proximal_point_id, k, -scaphoid["Principal_Axis_3"])
    #         scaphoid_proximal_point_id = scaphoid["Mesh"].find_closest_point(scaphoid_proximal_point)
    #         k = k + 1
    #     scaphoid_proximal_point = create_point(scaphoid_proximal_point, 0.5)

    
    # The scaphoid axis will be a line that runs between the distal and proximal poles of the scaphoid -- the points will be extended to see how many times the scaphoid line intersects the scaphoid mesh
    scaphoid_line, scaphoid_tube, scaphoid_vector, scaphoid_normalized_vector, scaphoid_arrow = get_mesh_lines(scaphoid_distal_point.GetCenter(), scaphoid_proximal_point.GetCenter(), SCAPHOID_LENGTH_MULTIPLIER)
    scaphoid_distal_point_extended = create_point(scaphoid_distal_point.GetCenter() - scaphoid_vector * 10, 0.5)
    scaphoid_proximal_point_extended = create_point(scaphoid_proximal_point.GetCenter() + scaphoid_vector * 10, 0.5)

    # If there are more than two intersections between the line and the mesh, adjustments may need to be made to the scaphoid line
    # The code block below adjusts the scaphoid line until there are only two intersections between the line and the mesh (indicating a tangential line)
    scaphoid_num_intersections, scaphoid_intersection_points, scaphoid_intersection_meshes = count_line_mesh_interactions(scaphoid_distal_point_extended.GetCenter(), scaphoid_proximal_point_extended.GetCenter(), scaphoid["Mesh"])
    # print("Pre-adjustment number of interactions with the scaphoid line: ", scaphoid_num_intersections)
    k = 2
    while scaphoid_num_intersections != 2:
        if k == 10:
            print("Could not get tangential line, exiting loop.") # The original line definition will be used if a tangential line cannot be found within 10 iterations
            break
        sc_line_start, sc_line_end = adjust_line(scaphoid["Mesh"], scaphoid_distal_point.GetCenter(), scaphoid_proximal_point.GetCenter(), k)
        scaphoid_line, scaphoid_tube, scaphoid_vector, scaphoid_normalized_vector, scaphoid_arrow = get_mesh_lines(sc_line_start, sc_line_end, SCAPHOID_LENGTH_MULTIPLIER)
        scaphoid_num_intersections, scaphoid_intersection_points, scaphoid_intersection_meshes = count_line_mesh_interactions(sc_line_start, sc_line_end, scaphoid["Mesh"])
        # print("Intersections: ", scaphoid_num_intersections)
        k = k + 1
    # print("Post-adjustment number of interactions with the scaphoid line: ", scaphoid_num_intersections) # This will equal 2 if the line adjustment was successful

    if visualize == True:
        p = pv.Plotter(window_size=(600,600))
        p.add_mesh(scaphoid["Mesh"], color = "white", opacity = 0.8, smooth_shading = True)
        p.add_mesh(scaphoid_tube, color = "red", line_width = 5)
        p.add_mesh(scaphoid_arrow, color = "red", line_width = 5)
        p.add_mesh(scaphoid_distal_point, color = "blue", point_size = 10)
        p.add_mesh(scaphoid_proximal_point, color = "green", point_size = 10)
        p.show_grid()
        camera_position = "xy"
        p.camera_position = camera_position
        p.show()

    return scaphoid_line, scaphoid_tube, scaphoid_vector, scaphoid_normalized_vector, scaphoid_arrow, scaphoid_intersection_points

def get_lunate_axis(lunate, radial_axes, visualize) -> tuple[pv.Line, pv.Tube, np.ndarray, np.ndarray, pv.Arrow]:
    """This function defines the lunate axis, which will run through the centre of the lunate and in a perpendicular direction to the lunate guide axis.
    This function will define the lunate guide axis based on the articulating surface between the lunate and capitate bones.
    The articulating surface is defined using curvature-based methods, and the maximum points in the direction of the articulating surface average normal on the palmar and dorsal sides of the lunate are used to define the lunate guide axis.


    Args:
        lunate (dict): The dictionary containing the lunate information.
        radial_axes (numpy array): The radial axes for reference.
        

    Returns:
        lunate_line, lunate, tube, lunate_vector, lunate_normalized_vector, lunate_arrow (vtk.vtkPolyData, vtk.vtkPolyData, numpy array, numpy array, vtk.vtkPolyData): The line, tube, vector, normalized vector, and arrow for the lunate guide axis.
    """
    # Getting the centroid of the lunate mesh
    lunate_centroid, lunate_centroid_vtk = compute_bone_centroid(lunate["Mesh"])

    # Getting the principal inertial axes of the lunate
    lunate_principal_axes = compute_mass_moment_of_inertia(lunate["Mesh"])
    lunate_principal_axes = adjust_lunate_principal_axes(lunate, radial_axes)
    lunate_split_indices, lunate_split_points, lunate_split_colours = split_bone_into_quadrants(lunate, lunate_principal_axes, lunate_centroid.GetCenter(), "Y", "Z")
    lunate_adjacency_list = build_mesh_adjacency_list(lunate["Mesh"]) # Building the adjacency list of the lunate mesh - will be used when building the lunate-capitate articulating surface

    # Get the centroid of the Quadrant 1 points - this is the seed point for the articulating surface of the lunate
    lunate_centroid_quadrant_1 = np.mean(lunate_split_points[0], axis = 0)
    quad_1_closest_point_id, quad_1_closest_point = find_closest_point_on_mesh(lunate["Mesh"], lunate_centroid_quadrant_1)

    # This obtains all points on the lunate mesh with a negative mean curvature 
    negative_curvature_indices = set()
    for i in range(len(lunate["Mesh"].point_data["Mean_Curvature"])):
        if lunate["Mesh"].point_data["Mean_Curvature"][i] <= 0:
            negative_curvature_indices.add(i)
    negative_curvature_points = np.array([lunate["Mesh"].points[i] for i in negative_curvature_indices])
    negative_curvature_polydata = pv.PolyData(negative_curvature_points)

    # Grow a region of negative curvature points around the quad_1_closest_point - these negative curvature components are meant to represent the articulating surface of the lunate
    closest_neg_curvature_index_in_subset = negative_curvature_polydata.find_closest_point(quad_1_closest_point)
    closest_neg_curvature_index = list(negative_curvature_indices)[closest_neg_curvature_index_in_subset]
    closest_neg_curvature_point = lunate["Mesh"].points[closest_neg_curvature_index]
    # The closest point in the negative curvature region is used as a seed point to grow the articulating surface region - the largest region around this point is defined as the articulating surface of the lunate
    connected_negative_curvature_region = grow_region(lunate_adjacency_list, closest_neg_curvature_index, negative_curvature_indices)
    articulating_surface_points = np.array([lunate["Mesh"].points[i] for i in connected_negative_curvature_region])
    articulating_surface_polydata = pv.PolyData(articulating_surface_points)
    articulating_surface_normals = np.array([lunate["Mesh"].point_normals[i] for i in connected_negative_curvature_region])

    # Create a bounding box around the articulating surface points
    articulating_surface_bounding_box = articulating_surface_polydata.bounds
    box_mesh = pv.Box(bounds = articulating_surface_bounding_box)

    # Get the mean normal vector of the articulating surface
    # This will act as the line through the lunate - it is intended to be perpendicular to the articulating surface -- the hope is that this helps to understand the radial-ulnar deviation of the lunate from an AP view
    mean_distal_normal_vector = np.mean(articulating_surface_normals, axis = 0)
    mean_distal_normal_vector = mean_distal_normal_vector / np.linalg.norm(mean_distal_normal_vector)
    mean_distal_normal_line = pv.Line(quad_1_closest_point - mean_distal_normal_vector * 30, quad_1_closest_point + mean_distal_normal_vector * 30)

    # Dividing the lunate into palmar and dorsal points based on the articulating surface bounding box as defined by the negative curvature points
    lunate_articulating_surfaces_vertices = compute_oriented_bounding_box(articulating_surface_points)
    lunate_box_lines, lunate_edge_midpoints, lunate_midpoints_list = create_box_lines(lunate_articulating_surfaces_vertices)
    plane_vertices = [lunate_midpoints_list[0], lunate_midpoints_list[2], lunate_midpoints_list[4], lunate_midpoints_list[6]] # NEED TO CHECK THIS
    palmar_dorsal_box, palmar_dorsal_plane, plane_center, palmar_dorsal_plane_normal_vector, palmar_dorsal_plane_normal_arrow = create_plane_and_normal(plane_vertices)
    lunate_palmar_points = []
    lunate_dorsal_points = []


    # # The articulating surface of the lunate with the capitate is used to separate the lunate into palmar and dorsal points
    # # This is done as a foolproof way to separate the two sides -- the principal axes could have been used but this controls for any misalignment of the principal axes
    # original_LC_surface_mesh, LC_surface, LC_surface_mesh, LC_bounding_box_vertices, LC_bounding_box_lines, LC_bounding_box_midpoints, LC_central_point, LC_midpoints, LC_closest_points, LC_min_distance_lines, LC_distances, LC_positions, LC_midpoints_list, LC_box_index = define_articulating_surface(lunate, capitate, LC_values[0], LC_values[1], LC_values[2])    
    # plane_vertices = [LC_midpoints_list[0], LC_midpoints_list[2], LC_midpoints_list[4], LC_midpoints_list[6]]
    # palmar_dorsal_box, palmar_dorsal_plane, plane_center, palmar_dorsal_plane_normal_vector, palmar_dorsal_plane_normal_arrow = create_plane_and_normal(plane_vertices)
    # # Dividing the lunate into palmar and dorsal points based on the plane created using the capitate articulating surface
    # lunate_palmar_points = []
    # lunate_dorsal_points = []




    for point in lunate["Mesh"].points:
        vector_to_point = point - quad_1_closest_point # quad_1_closest point is intended to be close to the centre of the articulating surface of the lunate
        if np.dot(vector_to_point, palmar_dorsal_plane_normal_vector) > 0:
            lunate_dorsal_points.append(point)
        else:
            lunate_palmar_points.append(point)
    lunate_palmar_points = np.array(lunate_palmar_points)
    lunate_dorsal_points = np.array(lunate_dorsal_points)

    # Initial estimate of the maximum palmar and dorsal points along the vector that is normal to the distal articulating surface of the lunate
    lunate_max_palmar_point = find_max_point_along_axis(lunate_palmar_points, quad_1_closest_point, mean_distal_normal_vector)
    lunate_max_dorsal_point = find_max_point_along_axis(lunate_dorsal_points, quad_1_closest_point, mean_distal_normal_vector)

    # Get the normal at the lunate_max_palmar_point and lunate_max_dorsal_point - these will be used to find the maximum palmar and dorsal points along the normal vector (to essentially mimic the two lunate horns)
    lunate_max_palmar_point_normal = lunate["Mesh"].point_normals[lunate["Mesh"].find_closest_point(lunate_max_palmar_point)]
    lunate_max_palmar_point_normal = lunate_max_palmar_point_normal / np.linalg.norm(lunate_max_palmar_point_normal)
    lunate_max_dorsal_point_normal = lunate["Mesh"].point_normals[lunate["Mesh"].find_closest_point(lunate_max_dorsal_point)]
    lunate_max_dorsal_point_normal = lunate_max_dorsal_point_normal / np.linalg.norm(lunate_max_dorsal_point_normal)

    # Representing an estimate of the lunate horns by taking the top 5% of points along the normal vector from the maximum palmar and dorsal points
    # This step ensures that the lunate line is not biased towards extreme points on the lunate horns, which may be more curved in some participants
    lunate_max_palmar_points = find_max_percent_of_points_along_axis(lunate_palmar_points, quad_1_closest_point, lunate_max_palmar_point_normal, 0.05)
    lunate_max_dorsal_points = find_max_percent_of_points_along_axis(lunate_dorsal_points, quad_1_closest_point, lunate_max_dorsal_point_normal, 0.05)
    lunate_max_palmar_points = np.array(lunate_max_palmar_points)
    lunate_max_dorsal_points = np.array(lunate_max_dorsal_points)

    # Find the closest point to the average of the maximum palmar and dorsal points - this is intended to bring the lunate line closer to the centre of the lunate horn plateaus rather than being biased towards the edges
    # This controls for participants that have different morphologies of the lunate horns, where some will be more curved while others are flatter
    dorsal_id, lunate_max_dorsal_point = find_closest_point_on_mesh(lunate["Mesh"], np.mean(lunate_max_dorsal_points, axis = 0))
    palmar_id, lunate_max_palmar_point = find_closest_point_on_mesh(lunate["Mesh"], np.mean(lunate_max_palmar_points, axis = 0))
    lunate_palmar_point = create_point(lunate_max_palmar_point, 0.5)
    lunate_dorsal_point = create_point(lunate_max_dorsal_point, 0.5)

    # Creating a midpoint between the palmar and dorsal points -- this is where the lunate line will go through
    midpoint = (np.array(lunate_palmar_point.GetCenter()) + np.array(lunate_dorsal_point.GetCenter())) / 2
    midpoint = create_point(midpoint, 0.5)

    # Creating the lunate guide line that runs along the lunate horns
    lunate_guide_line, lunate_guide_tube, lunate_guide_vector, lunate_normalized_guide_vector, lunate_guide_arrow = get_mesh_lines(lunate_palmar_point.GetCenter(), lunate_dorsal_point.GetCenter(), LUNATE_LENGTH_MULTIPLIER)

    # Extend the lunate dorsal and palmar points to be further along the vector from the midpoint to the palmar and dorsal points
    # This was done to create a line that runs across the lunate and can identify when there are more than two intersections along the lunate mesh (which ensures that it is tangent to the lunate mesh)
    lunate_dorsal_point_extended = lunate_dorsal_point.GetCenter() + lunate_normalized_guide_vector * 2
    lunate_dorsal_point_extended = create_point(lunate_dorsal_point_extended, 0.5)
    lunate_palmar_point_extended = lunate_palmar_point.GetCenter() - lunate_normalized_guide_vector * 2
    lunate_palmar_point_extended = create_point(lunate_palmar_point_extended, 0.5)

    # Count the number of intersections between the lunate guide line and the lunate mesh
    lunate_num_intersections, lunate_intersection_points, lunate_intersection_point_meshes = count_line_mesh_interactions(lunate_dorsal_point_extended.GetCenter(), lunate_palmar_point_extended.GetCenter(), lunate["Mesh"])
    # print("Pre-adjustment number of interactions with the lunate line: ", lunate_num_intersections)
    k = 1
    # Adjust the lunate guide line until it intersects the lunate mesh at exactly two points -- ensuring a tangent line like what is done in 2D measurements
    while lunate_num_intersections != 2:
        if k == 10: # This just ensures that the loop is not an infinite loop
            break
        lun_line_start, lun_line_end = adjust_line(lunate["Mesh"], lunate_dorsal_point.GetCenter(), lunate_palmar_point.GetCenter(), k)
        lunate_guide_line, lunate_guide_tube, lunate_guide_vector, lunate_normalized_guide_vector, lunate_guide_arrow = get_mesh_lines(lun_line_start, lun_line_end, LUNATE_LENGTH_MULTIPLIER)
        lunate_num_intersections, lunate_intersection_points, lunate_intersection_point_meshes = count_line_mesh_interactions(lun_line_start, lun_line_end, lunate["Mesh"])
        # print("Intersections: ", lunate_num_intersections)
        k = k + 1
    if lunate_num_intersections != 2:
        print("Lunate line does not intersect at exactly two points. Adjustments required.")
    # print("Post-adjustment number of interactions with the lunate line: ", lunate_num_intersections)


    # Create two new points based on the line adjustments made where tangency was achieved
    lunate_palmar_point, lunate_dorsal_point = lunate_intersection_point_meshes[0], lunate_intersection_point_meshes[1]
    lunate_palmar_point = create_point(lunate_palmar_point.GetCenter(), 0.5)
    lunate_dorsal_point = create_point(lunate_dorsal_point.GetCenter(), 0.5)

    # Don't really need this code here but keeping here now for reference
    lunate_palmar_point_normal = lunate["Mesh"].point_normals[lunate["Mesh"].find_closest_point(lunate_palmar_point.GetCenter())]
    lunate_palmar_point_normal = lunate_palmar_point_normal / np.linalg.norm(lunate_palmar_point_normal)
    lunate_dorsal_point_normal = lunate["Mesh"].point_normals[lunate["Mesh"].find_closest_point(lunate_dorsal_point.GetCenter())]
    lunate_dorsal_point_normal = lunate_dorsal_point_normal / np.linalg.norm(lunate_dorsal_point_normal)
    average_normal = (lunate_palmar_point_normal + lunate_dorsal_point_normal) / 2

    midpoint_np = (np.array(lunate_palmar_point.GetCenter()) + np.array(lunate_dorsal_point.GetCenter())) / 2
    midpoint_np = create_point(midpoint_np, 0.1)
    # Draw a line from the midpoint in the direction of the average normal vector to create the lunate line
    lunate_line = pv.Line(midpoint_np.GetCenter() + mean_distal_normal_vector * 30, midpoint_np.GetCenter() - mean_distal_normal_vector * 30)
    lunate_vector = lunate_line.points[0] - lunate_line.points[-1]
    lunate_vector = lunate_vector / np.linalg.norm(lunate_vector)

    # Get the intersection points between the mean normal line and the lunate mesh
    lunate_line_intersections, lunate_line_intersection_points, _ = count_line_mesh_interactions(lunate_line.points[-1], lunate_line.points[0], lunate["Mesh"])

    # Get the intersection point that is furthest from the midpoint -- this corresponds to the proximal point on the lunate
    # The initial creation of the average normal vector was just to get the furthest point from the midpoint, but is not used as the actual vector for the lunate line
    max_distance = 0
    max_point = None
    for i in range(lunate_line_intersection_points.GetNumberOfPoints()):
        point = lunate_line_intersection_points.GetPoint(i)
        distance = np.linalg.norm(np.array(point) - midpoint_np.GetCenter())
        if distance > max_distance:
            max_distance = distance
            max_point = point
    lunate_proximal_point = create_point(max_point, 0.5)

    lunate_guide_line_vector = np.array(lunate_palmar_point.GetCenter()) - np.array(lunate_dorsal_point.GetCenter())
    lunate_guide_line_vector = lunate_guide_line_vector / np.linalg.norm(lunate_guide_line_vector)
    lunate_line_vector = np.array(lunate_proximal_point.GetCenter()) - np.array(midpoint.GetCenter())
    lunate_line_vector = lunate_line_vector / np.linalg.norm(lunate_line_vector)

    # *****
    # projection = np.dot(lunate_line_vector, lunate_guide_line_vector) * lunate_guide_line_vector
    # lunate_line_vector = lunate_line_vector - projection
    # lunate_line_vector = lunate_line_vector / np.linalg.norm(lunate_line_vector)

    # lunate_line = pv.Line(midpoint_np.GetCenter() - lunate_line_vector * 30, midpoint_np.GetCenter() + lunate_line_vector * 30)
    # *****

    # The lunate vector must be in the opposite direction of the third principal axis of the lunate. If it is not, it must be flipped. (This case rarely occurs but is here as a control)
    if np.dot(lunate_line_vector, lunate["Principal_Axis_3"]) > 0:
        lunate_line_vector = -lunate_line_vector
        lunate_line.points = np.array([lunate_line.points[-1], lunate_line.points[0]])
        lunate_line = pv.Line(midpoint_np.GetCenter() - mean_distal_normal_vector * 30, midpoint_np.GetCenter() + mean_distal_normal_vector * 30)

    lunate_tube = lunate_line.tube(radius = 0.1)

    articulating_surface_angle = calculate_3D_angle(lunate_guide_line_vector, lunate_line_vector)
    # print("Angle between the lunate guide line and the lunate line: ", articulating_surface_angle)

    if visualize == True:
        p = pv.Plotter(window_size =(600,600))
        p.add_mesh(lunate["Mesh"], color = "white", opacity = 1, smooth_shading = True)
        p.add_mesh(lunate_centroid_quadrant_1, color = "black", point_size = 10, render_points_as_spheres = True, smooth_shading = True)
        p.add_mesh(quad_1_closest_point, color = "purple", point_size = 10, render_points_as_spheres = True, smooth_shading = True)
        p.add_mesh(midpoint_np, color = "black")
        p.add_mesh(lunate_guide_line, color = "purple")
        p.add_mesh(lunate_intersection_point_meshes[0], color = "red")
        p.add_mesh(lunate_intersection_point_meshes[1], color = "blue")
        p.add_mesh(lunate_line, color = "red")
        p.show_grid()
        p.show()

    return lunate_guide_line, lunate_guide_vector, lunate_tube, lunate_line_vector

def get_capitate_axis(capitate, visualize) -> tuple[pv.Tube, np.ndarray]:
    """This function defines how to get the capitate axis. It was found that the best way to define this was as follows:
    Find the maximum points along the second principal axis along both the radial and ulnar sides of the distal component of the capitate. And then create a midpoint between these two points.
    Then, find the maximum points along the second principal axis along the ulnar side of the capitate and the centroid of the radial-proximal component of the capitate.  
    This creates a first iteration of the capitate axis. From here, adjust the proximal component of the capitate axis to be the centroid of the lowest 15% of proximal points along the capitate axis. 
    The new capitate axis is then defined as the line between the centroid of the distal points and the centroid of the proximal points.

    Args:
        capitate (dict): The dictionary containing the capitate information.

    Returns:
        capitate_tube (vtk.vtkPolyData): The tube representing the capitate axis.
        capitate_vector (numpy array): The vector representing the capitate axis.
    """
    capitate_principal_axes = [capitate["Principal_Axis_1"], capitate["Principal_Axis_2"], capitate["Principal_Axis_3"]]
    capitate_centroid = capitate["Centroid"].GetCenter()
    capitate_quadrant_indices, capitate_quadrant_points, capitate_quadrant_colours = split_bone_into_quadrants(capitate, capitate_principal_axes, capitate_centroid, "X", "Y") # Splitting the capitate into quadrants based on the first two principal axes
    # Four points on the capitate to be used initially:
    # 1) Max point in the principal direction 2 in Quad 1 (Distal & Radial)
    # 2) Max point in -principal direction 2 in Quad 2 (Distal & Ulnar)
    # 3) Max point in -principal direction 2 in Quad 3 (Proximal & Ulnar)
    # 4) The closest point to the centroid in Quad 4 (Proximal & Radial)
    dist_rad_point = find_max_point_along_axis(capitate_quadrant_points[0], capitate_centroid, capitate["Principal_Axis_2"])
    dist_rad_point = create_point(dist_rad_point, 0.5)
    dist_uln_point = find_max_point_along_axis(capitate_quadrant_points[1], capitate_centroid, -capitate["Principal_Axis_2"])
    dist_uln_point = create_point(dist_uln_point, 0.5)
    prox_uln_point = find_max_point_along_axis(capitate_quadrant_points[3], capitate_centroid, -capitate["Principal_Axis_2"])
    prox_uln_point = create_point(prox_uln_point, 0.5)
    # The proximal radial point is not necessarily always a local maximum, so the point closest to the centroid in that quadrant is used
    prox_rad_centroid = np.mean(capitate_quadrant_points[2], axis = 0)
    closest_index = np.argmin(np.linalg.norm(capitate_quadrant_points[2] - prox_rad_centroid, axis = 1))
    prox_rad_point = capitate_quadrant_points[2][closest_index]
    prox_rad_point = create_point(prox_rad_point, 0.5)

    # A point will be created on the distal and proximal poles of the capitate -- these are the means of the distal and proximal points created above
    capitate_distal_point = np.mean([dist_rad_point.GetCenter(), dist_uln_point.GetCenter()], axis = 0)
    capitate_distal_point_vtk = create_point(capitate_distal_point, 0.5)
    capitate_proximal_point = np.mean([prox_uln_point.GetCenter(), prox_rad_point.GetCenter()], axis = 0)
    capitate_proximal_point = create_point(capitate_proximal_point, 0.5)

    # Create an initial capitate axis line between the distal and proximal points
    capitate_vector = np.array(capitate_proximal_point.GetCenter()) - np.array(capitate_distal_point_vtk.GetCenter())
    # The capitate vector must be in the same direction as the first principal axis of the capitate, which is the vector that points from the proximal pole to the distal pole
    if np.dot(capitate_vector, capitate["Principal_Axis_1"]) > 0:
        capitate_vector = -capitate_vector
    capitate_vector = capitate_vector / np.linalg.norm(capitate_vector)
    capitate_line = pv.Line(capitate_centroid + capitate_vector * 1.5, capitate_centroid - capitate_vector * 1.5)
    capitate_tube = pv.Line(capitate_centroid + capitate_vector * 1.5, capitate_centroid - capitate_vector * 1.5).tube(radius = 0.25)
    capitate_arrow = pv.Arrow(start = capitate_centroid, direction = capitate_vector, scale = 10)

    # Find the top 15% of the points in the direction of the capitate vector - these will be used to redefine the proximal point of the capitate axis
    capitate_prox_points = find_max_percent_of_points_along_axis(capitate["Mesh"].points, capitate_centroid, capitate_vector, 0.15)
    capitate_prox_points = np.array(capitate_prox_points)
    capitate_prox_points_centroid = np.mean(capitate_prox_points, axis = 0)

    # Create a line between the two centroids
    capitate_new_line = pv.Line(capitate_prox_points_centroid, np.array(capitate_distal_point))
    capitate_vector = capitate_prox_points_centroid - capitate_distal_point
    capitate_vector = capitate_vector / np.linalg.norm(capitate_vector)
    capitate_tube = pv.Line(capitate_prox_points_centroid + capitate_vector * 10, capitate_distal_point - capitate_vector * 10).tube(radius = 0.1)
    capitate_arrow = pv.Arrow(start = capitate_prox_points_centroid, direction = capitate_vector, scale = 10)

    capitate_prox_centroid = create_point(capitate_prox_points_centroid, 0.5)

    if visualize == True:
        p = pv.Plotter(window_size = (600, 600))
        p.add_mesh(capitate["Mesh"], color = "purple", opacity = 0.4, smooth_shading = True)
        p.add_mesh(pv.Arrow(start = capitate_centroid, direction = capitate["Principal_Axis_1"], scale = 4), color = "red")
        p.add_mesh(pv.Arrow(start = capitate_centroid, direction = capitate["Principal_Axis_2"], scale = 4), color = "green")
        p.add_mesh(pv.Arrow(start = capitate_centroid, direction = capitate["Principal_Axis_3"], scale = 4), color = "blue")
        p.add_mesh(capitate_tube, color = "red", line_width = 1)
        p.add_mesh(dist_rad_point, color = "blue")
        p.add_mesh(dist_uln_point, color = "blue")
        p.add_mesh(prox_uln_point, color = "blue")
        p.add_mesh(prox_rad_point, color = "blue")
        p.add_mesh(capitate_distal_point_vtk, color = "red")
        p.add_mesh(capitate_proximal_point, color = "red", opacity = 0.75)
        p.add_mesh(capitate_prox_centroid, color = "black")
        position = "xy"
        p.camera_position = position
        p.show()

    return capitate_tube, capitate_vector

def get_mc3_axis(mc3, visualize) -> tuple[pv.Tube, np.ndarray]:
    """The third metacarpal axis will pass through the middle of the third metacarpal.
    For now, a range of the middle 80% was selected--this was found to best represent an axis along the third metacarpal, even when the top of the bone was not in the field of view

    Args:
        mc3 (dict): The dictionary containing the third metacarpal information.
    
    Returns:
        mc3_tube (vtk.vtkPolyData): The tube representing the third metacarpal axis.
        mc3_vector (numpy array): The vector representing the third metacarpal axis.
        mc3_normalized_vector (numpy array): The normalized vector representing the third metacarpal axis.
        mc3_arrow (vtk.vtkPolyData): The arrow representing the third metacarpal axis.
    """
    mc3_points = mc3["Mesh"].points
    mc3_centroid = np.array(mc3["Centroid"].GetCenter())

    # Create a best fit line through the middle 80% of the centroids along the third metacarpal
    # Not using the principal axis directly as this was found to be less accurate in some cases where the top of the third metacarpal was not in the field of view
    # Cutting out the end 10% helps to prevent the line from being biased by the shape of the distal and proximal ends of the third metacarpal
    projections = np.dot(mc3_points, mc3["Principal_Axis_1"])
    sorted_indices = np.argsort(projections)
    sorted_mc3_points = mc3_points[sorted_indices]
    projection_10_percentile = np.percentile(projections, 10)
    projection_90_percentile = np.percentile(projections, 90)
    slice_positions = np.linspace(projection_10_percentile, projection_90_percentile, num=50)
    centroids = []
    for pos in slice_positions:
        points = sorted_mc3_points[np.abs(projections - pos) < 0.01]
        if len(points) > 0:
            centroids.append(np.mean(points, axis=0))
    centroids = np.array(centroids)
    pca = PCA(n_components=1)
    pca.fit(centroids)
    mc3_vector = pca.components_[0]

    # The MC3 vector must always be in the opposite direction as the first principal axis of the third metacarpal, which is the vector that points from the proximal pole to the distal pole
    # This is a minor change but helps to ensure that the axis measurement is not flipped by 180° (e.g. in Radius-MC3 angle measurements)
    if np.dot(mc3_vector, mc3["Principal_Axis_1"]) > 0: # Flip the vector if it is in the wrong direction
        mc3_vector = -mc3_vector
    mc3_normalized_vector = mc3_vector / np.linalg.norm(mc3_vector)
    mc3_line = pv.Line(mc3_centroid + mc3_vector * 50, mc3_centroid - mc3_vector * 50)
    mc3_tube = pv.Line(mc3_centroid + mc3_vector * 50, mc3_centroid - mc3_vector * 50).tube(radius = 0.25)
    mc3_arrow = pv.Arrow(start = mc3_centroid, direction = mc3_vector, scale = 10)

    if visualize == True:
        p = pv.Plotter(window_size = (600, 600))
        p.add_mesh(mc3["Mesh"], color = "orange", opacity = 0.5, smooth_shading = True)
        p.add_mesh(pv.Arrow(start = mc3_centroid, direction = mc3["Principal_Axis_1"], scale = 6), color = "red")
        p.add_mesh(pv.Arrow(start = mc3_centroid, direction = mc3["Principal_Axis_2"], scale = 6), color = "green")
        p.add_mesh(pv.Arrow(start = mc3_centroid, direction = mc3["Principal_Axis_3"], scale = 6), color = "blue")
        p.add_mesh(mc3_tube, color = "red", line_width = 5)
        p.show()

    return mc3_tube, mc3_vector

def adjust_scaphoid_principal_axes(scaphoid, radial_axes) -> np.ndarray:
    """This function will adjust the sense of the scaphoid principal inertial axes to ensure they are in a consistent orientation with respect to the radial axes.
    This ensures that the sense of the axes are in the same direction regardless of slight changes in bone morphology.
    Specifically, the first principal axis will run from the centroid in the direction of the distal pole (from the proximal pole) to the scaphoid. This will be directed radially.
    The second principal axis must be directed distally, to ensure that the third principal axis points through the capitate facet.

    Args:
        scaphoid (vtk.vtkPolyData): The input scaphoid surface mesh.
        radial_axes (numpy array): The radial axes of the position.

    Returns:
        new_principal_axes (numpy array): The adjusted principal axes of the scaphoid.
    """
    inertia_tensor, eigenvalues, eigenvectors = compute_mass_moment_of_inertia(scaphoid["Mesh"])
    x_axis = eigenvectors[:, 0]
    y_axis = eigenvectors[:, 1]

    RCS_x_axis = radial_axes[0]
    RCS_y_axis = radial_axes[1]
    RCS_z_axis = radial_axes[2]

    # Calculate the dot product of the first principal axis and the y-axis of the radius
    # This block ensures that the first principal axis of inertia is directed radially, as slight changes in shape may cause this axis to be pointed ulnarly
    dot_product = np.dot(x_axis, RCS_y_axis)
    if dot_product < 0:
        x_axis = -x_axis

    # Calculate the dot product of the second principal axis and the z-axis of the radius
    # This block ensures that the second principal axis of inertia is directed distally, as slight changes in shape may cause this axis to be pointed dorsally
    dot_product = np.dot(y_axis, RCS_z_axis)
    if dot_product > 0:
        y_axis = -y_axis

    z_axis = np.cross(x_axis, y_axis)

    new_principal_axes = np.array([x_axis, y_axis, z_axis])

    return new_principal_axes

def adjust_lunate_principal_axes(lunate, radial_axes) -> np.ndarray:
    """This function will adjust the sense of the principal inertial axes of a bone mesh based on the radial axes.
    Specifically, it will ensure that the first principal axis (that is, the X-axis) points is directed radially and palmarly.
    This can be determined by calculating the dot product of the first principal axis and the y-axis of the radius. And if the dot product is negative, the first principal axis will be reversed.
    Similarly, the dot product of the first principal axis and the z-axis of the radius will be calculated. If the dot product is negative, the first principal axis will be reversed.
    The second principal axis (that is, the Y-axis) should also be directed radially. This is calculated by calculating the dot product of the second principal axis and the z-axis of the radius. If the dot product is negative, the second principal axis will be reversed.
    The third principal axis is still the cross product of the first two axes.

    Args:
        mesh (vtk.vtkPolyData): The input bone surface mesh.
        radial_axes (numpy array): The radial axes of the bone.

    Returns:
        new_principal_axes (numpy array): The adjusted principal axes of the bone.
    """
    inertia_tensor, eigenvalues, eigenvectors = compute_mass_moment_of_inertia(lunate["Mesh"])

    x_axis = eigenvectors[:, 0]
    y_axis = eigenvectors[:, 1]

    RCS_x_axis = radial_axes[0]
    RCS_y_axis = radial_axes[1]
    RCS_z_axis = radial_axes[2]

    # Calculate the dot product of the first principal axis and the y-axis of the radius
    # This block ensures that the first principal axis of inertia is directed radially, as slight changes in shape may cause this axis to be pointed ulnarly
    dot_product = np.dot(x_axis, RCS_y_axis)
    if dot_product < 0:
        x_axis = -x_axis

    # Calculate the dot product of the first principal axis and the z-axis of the radius
    # This block ensures that the first principal axis of inertia is directed palmarly, as slight changes in shape may cause this axis to be pointed dorsally
    dot_product = np.dot(x_axis, RCS_z_axis)
    if dot_product < 0:
        x_axis = -x_axis

    # Calculate the dot product of the second principal axis and the y-axis of the radius - just in the case of the 2D XY components of the vectors
    # This block ensures that the second principal axis of inertia is directed radially, as slight changes in shape may cause this axis to be pointed ulnarly
    y_axis_2D = np.array([y_axis[0], y_axis[1], 0])
    RCS_y_axis_2D = np.array([RCS_y_axis[0], RCS_y_axis[1], 0])
    dot_product = np.dot(y_axis_2D, RCS_y_axis_2D)
    if dot_product < 0:
        y_axis = -y_axis

    # Re-calculating the third principal axis based on the adjusted first and second axes
    z_axis = np.cross(x_axis, y_axis)
    new_principal_axes = np.array([x_axis, y_axis, z_axis])

    return new_principal_axes

def adjust_capitate_principal_axes(capitate, mc3) -> np.ndarray:
    """This function just ensures that the capitate principal inertial axes are in the correct direction.
    The first principal axis should be in the same direction as the first principal axis of the third metacarpal.

    Args:
        capitate (dict): The dictionary containing the capitate information.
        mc3 (dict): The dictionary containing the third metacarpal information.
    Returns:
        new_principal_axes (numpy array): The adjusted principal axes of the capitate.
    """
    inertia_tensor, eigenvalues, eigenvectors = compute_mass_moment_of_inertia(capitate["Mesh"])
    x_axis = eigenvectors[:, 0]
    y_axis = eigenvectors[:, 1]

    # The first principal axis of the capitate should be in the same direction as the first principal axis of the third metacarpal
    mc3_x_axis = mc3["Principal_Axis_1"]
    dot_product = np.dot(x_axis, mc3_x_axis)
    if dot_product < 0:
        x_axis = -x_axis

    z_axis = np.cross(x_axis, y_axis)
    new_principal_axes = np.array([x_axis, y_axis, z_axis])

    return new_principal_axes

def adjust_mc3_principal_axes(mc3, radial_axes, radius, position, pa2_flip) -> np.ndarray:
    """This function ensures that the first principal axis of the third metacarpal is directed distally (in a neutral scan).
    To accomplish this, the first principal axis must be directed towards the distal side of the hand (that is, the -X-axis of the radius).
    The dor product of the first principal axis and the -X-axis of the radius will be calculated, and if it is positive (pointing in the same direction), the first principal axis will be reversed.
    The first principal axis will be re-calculated based on this criteria. 

    Args:
        mc3 (dict): The dictionary containing the third metacarpal information.
        radial_axes (numpy array): The radial axes of the position.
        sagittal_plane (numpy array): The sagittal plane of the position.

    Returns:
        new_principal_axes (numpy array): The adjusted principal axes of the third metacarpal.
    """
    inertia_tensor, eigenvalues, eigenvectors = compute_mass_moment_of_inertia(mc3["Mesh"])
    x_axis = eigenvectors[:, 0]
    y_axis = eigenvectors[:, 1]
    z_axis = eigenvectors[:, 2]

    RCS_x_axis = radial_axes[0]
    RCS_y_axis = radial_axes[1]
    RCS_z_axis = radial_axes[2]

    # The first principal axis must be directed in the opposite direction of the vector from the MC3 centroid to the radius centroid
    mc3_radius_direction_vector = np.array(radius["Origin"].GetCenter()) - np.array(mc3["Centroid"].GetCenter())
    mc3_radius_direction_vector = mc3_radius_direction_vector / np.linalg.norm(mc3_radius_direction_vector)

    dot_product = np.dot(x_axis, mc3_radius_direction_vector)
    # print("RADIUS-MC3 DOT PRODUCT: ", dot_product)
    if dot_product > 0:
        x_axis = -x_axis

    # Need to add case here where second principal axis is not in the correct direction
    # NEUTRAL: If the second principal axis is not facing the palmar side of the hand (Radius Principal Axis 3), direction will be reversed -- based on projections onto the sagittal plane
    # PLANK OR PUSHUP: If the second principal axis is not facing the distal side of the hand (Radius Principal Axis 1), direction will be reversed -- based on projections onto the transverse plane
        
    # The second principal axis being in the correct direction is very reliant on both the position of the hand as well as the entire third metacarpal being in the field of view
    # Therefore, adjustments can be made manually to ensure it is in the same direction for multiple scans of the same hand in multiple positions
    if pa2_flip == True:
        y_axis = -y_axis
    # sagittal_plane_normal = radial_axes[2]
    # sagittal_plane_normal_2D = np.array([sagittal_plane_normal[0], sagittal_plane_normal[1], 0])
    # y_axis_2D = np.array([y_axis[0], y_axis[1], 0])
    # RCS_x_axis_2D = np.array([RCS_x_axis[0], RCS_x_axis[1], 0])
    # RCS_z_axis_2D = np.array([RCS_z_axis[0], RCS_z_axis[1], 0])
    # dot_product = np.dot(y_axis_2D, sagittal_plane_normal_2D)
    # if dot_product < 0:
    #     y_axis = -y_axis
    # if position == "NEUTRAL":
    #     dot_product = np.dot(y_axis_2D, RCS_z_axis_2D)
    #     if dot_product < 0:
    #         y_axis = -y_axis
    # elif position == "PUSHUP" or position == "PLANK":
    #     dot_product = np.dot(y_axis_2D, RCS_x_axis_2D)
    #     if dot_product > 0:
    #         y_axis = -y_axis

    z_axis = np.cross(x_axis, y_axis)
    new_principal_axes = np.array([x_axis, y_axis, z_axis])

    return new_principal_axes
