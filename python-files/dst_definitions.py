#   dorsal_scaphoid_translation.py
# contains function to measure dorsal scaphoid translation relative to the radius:
# - find_max_points_in_increments_for_dst: finds the maximum point along a given search axis in increments along another axis. This is used to find the points on the dorsal rim of the radius for the dorsal tangential line method.
# - get_3D_dst_components: this function will measure the dorsal translation of the scaphoid relative to the radius. Using Chan 2019's method, the dorsal tangential line method will be used. This requires the drawing of two lines that are parallel to the longitudinal axis of the radius, which are use in future calculations.
# - measure_dtl_dst: this function will measure the dorsal tangential line method dorsal scaphoid translation. This is done by projecting the dorsal scaphoid line and the radius rim line onto a plane parallel to the sagittal plane, and then finding the intersection of the projected dorsal scaphoid line with a line drawn from the midpoint of the projected radius rim line in the direction of the projected radius z-axis. The distance from this intersection point to the midpoint of the projected radius rim line is the dorsal scaphoid translation.
# - measure_prsa_dst: this function will measure the 2D and 3D PRSA based DST. The 3D angle is measured between the dorsal RS line and the radius line, while the 2D angle is measured between the projections of these lines onto a plane parallel to the sagittal plane.

import numpy as np
import pyvista as pv
from mesh_division import divide_mesh_into_sections, split_bone_into_quadrants_using_vectors
from point_finding import find_max_point_along_axis, find_closest_point_on_mesh, compute_bone_centroid
from mesh_processing import create_point, build_mesh_adjacency_list
from calc import calculate_2D_angle, calculate_3D_angle

def find_max_points_in_increments_for_dst(radius, radius_origin, scaphoid, radial_axes, increment_divisions) -> tuple[list]:
    """Finds the maximum point along a given search axis in increments along another axis.

    Args:
        radius (dict): The radius dictionary containing the mesh and the radius values.
        radius_origin (numpy array): The origin of the radius.
        radial_axes (numpy array): The principal axes of the radius.
        increment_axis (numpy array): The axis along which the increments are being made.
        increment_size (int): The size of the increments.

    Returns:
        max_points (list of numpy arrays): The maximum points found at each increment.
    """
    radius_quad_indices, radius_quad_points, radius_quad_colours = split_bone_into_quadrants_using_vectors(radius, radius_origin, radial_axes[1], -radial_axes[2])

    # Quad 1 and Quad 3 are the two quadrants that make up the dorsal side of the radius
    quad_1_points = radius_quad_points[0]
    quad_1_mesh = pv.PolyData(quad_1_points)
    quad_1_indices, quad_1_points, quad_1_labels = divide_mesh_into_sections(quad_1_mesh, radius_origin, radial_axes[1], increment_divisions)
    quad_3_points = radius_quad_points[2]
    quad_3_mesh = pv.PolyData(quad_3_points)
    quad_3_indices, quad_3_points, quad_3_labels = divide_mesh_into_sections(quad_3_mesh, radius_origin, radial_axes[1], increment_divisions)

    # Get the most ulnar point of the scaphoid (the point with the maximum projection along the -y axis)
    max_scaphoid_point = find_max_point_along_axis(scaphoid["Mesh"].points, radius_origin, -radial_axes[1])

    max_points = []
    for i in range(len(quad_1_points)):
        max_point = find_max_point_along_axis(quad_1_points[i], radius_origin, -radial_axes[0])
        # If the max point is more radial than the origin (that is, it is more positive than the scaphoid point in the direction of the radius y-axis), do not include it
        # This ensures that the radius point will (to some extent) be in the radioscaphoid articulation
        y_projections = np.dot(max_point - radius_origin, radial_axes[1])
        scaphoid_point_projections = np.dot(max_point - max_scaphoid_point, radial_axes[1])
        if scaphoid_point_projections > 0:
            max_points.append(max_point)
    for i in range(len(quad_3_points)):
        max_point = find_max_point_along_axis(quad_3_points[i], radius_origin, -radial_axes[0])
        # If the max point is more radial than the origin (that is, it is more positive than the scaphoid point in the direction of the radius y-axis), do not include it
        # This ensures that the radius point will (to some extent) be in the radioscaphoid articulation
        y_projections = np.dot(max_point - radius_origin, radial_axes[1])
        scaphoid_point_projections = np.dot(max_point - max_scaphoid_point, radial_axes[1])
        if scaphoid_point_projections > 0:
            max_points.append(max_point)

    p = pv.Plotter()
    p.add_mesh(pv.Arrow(start = radius_origin, direction = radial_axes[0], scale = 10), color = "red")
    p.add_mesh(pv.Arrow(start = radius_origin, direction = radial_axes[1], scale = 10), color = "green")
    p.add_mesh(pv.Arrow(start = radius_origin, direction = radial_axes[2], scale = 10), color = "blue")
    # Display the four regions of the radius
    for i in range(4):
        p.add_mesh(radius_quad_points[i], color = radius_quad_colours[i], point_size = 1)
    p.show_grid()
    p.show()
        
    return max_points

def get_3D_dst_components(radius, radial_axes, scaphoid, dst_type) -> list:
    """This function will measure the dorsal translation of the scaphoid relative to the radius.
    Chan 2019: "Dorsal translation of the proximal scaphoid pole onto the rim of the distal radius is a late finding associated with chronic SLI."
    Using Chan 2019's method, the dorsal tangential line method will be used. This requires the drawing of two lines that are parallel to the longitudinal axis of the radius
    --- The first line is drawn tangential to the proximal articular surface of the scaphoid.
    --- The second line is drawn through the dorsal scaphoid facet of the distal radius.

    Args:
        radius (dict): The dictionary containing the radius information.
        radial_axes (numpy array): The radial axes of the position.
        scaphoid (dict): The dictionary containing the scaphoid information.
        dst_type (str): The type of dorsal scaphoid translation to be measured. This can either be "DTL" or "PRSA" (dorsal tangential line method or posterior radioscaphoid angle method).
    """
    # Deining the dorsal point of the scaphoid
    scaphoid_centroid, scaphoid_centroid_vtk = compute_bone_centroid(scaphoid["Mesh"])
    scaphoid_centroid = create_point(scaphoid_centroid.GetCenter(), 0.5)
    scaphoid_split_indices, scaphoid_split_points, scaphoid_split_labels = divide_mesh_into_sections(scaphoid["Mesh"], scaphoid_centroid.GetCenter(), scaphoid["Principal_Axis_1"], 3)
    scaphoid_proximal_points = np.array(scaphoid_split_points[0])
    scaphoid_dorsal_point_np = find_max_point_along_axis(scaphoid_proximal_points, scaphoid_centroid.GetCenter(), -radial_axes[2])
    # The dorsal point of the scaphoid will only be for points on the proximal pole of the scaphoid--it is of the belief that this method allows for the most reproducible analysis regardless of wrist position.
    scaphoid_dorsal_point = create_point(scaphoid_dorsal_point_np, 0.5)
    scaphoid_dorsal_line = pv.Line(scaphoid_dorsal_point.GetCenter() - 20 * radial_axes[0], scaphoid_dorsal_point.GetCenter() + 20 * radial_axes[0])
    scaphoid_dorsal_tube = scaphoid_dorsal_line.tube(radius = 0.05)

    # Dividing the radius into quadrants and thirds to facilitate finding the dorsal rim point
    radius_origin = radius["Origin"].GetCenter()

    # Split the radius into quadrants and thirds
    rad_quad_indices, rad_quad_points, rad_quad_colours = split_bone_into_quadrants_using_vectors(radius, radius_origin, radial_axes[1], -radial_axes[2])

    # Splitting the radius into four quadrants
    q1_points, q1_mesh = rad_quad_points[0], pv.PolyData(rad_quad_points[0])
    q2_points, q2_mesh = rad_quad_points[1], pv.PolyData(rad_quad_points[1])
    q3_points, q3_mesh = rad_quad_points[2], pv.PolyData(rad_quad_points[2])
    q4_points, q4_mesh = rad_quad_points[3], pv.PolyData(rad_quad_points[3])

    # Get the width of the radius along the second principal axis of the radius (that is, the Y-axis)
    radius_width = np.max(np.dot(radius["Mesh"].points - radius_origin, radial_axes[1])) - np.min(np.dot(radius["Mesh"].points - radius_origin, radial_axes[1]))

    
    # Dividing each quadrant into slices along the Y-axis
    q1_indices, q1_points, q1_labels = divide_mesh_into_sections(q1_mesh, q1_mesh.center, radial_axes[1], int(radius_width))
    q2_indices, q2_points, q2_labels = divide_mesh_into_sections(q2_mesh, q2_mesh.center, radial_axes[1], int(radius_width))
    q3_indices, q3_points, q3_labels = divide_mesh_into_sections(q3_mesh, q3_mesh.center, radial_axes[1], int(radius_width))
    q4_indices, q4_points, q4_labels = divide_mesh_into_sections(q4_mesh, q4_mesh.center, radial_axes[1], int(radius_width))


    mean_curvature_array = radius["Mesh"].point_data["Mean_Curvature"]
    radius_adjacency_list = build_mesh_adjacency_list(radius["Mesh"])

    # The blocks of code below are to define specific points based on whether the user prefers the dorsal tangential line (DTL) method or the posterior radioscaphoid angle (PRSA) method

    if dst_type == "DTL":
        max_scaphoid_point = find_max_point_along_axis(scaphoid["Mesh"].points, radius_origin, -radial_axes[1]) # Starting with the most ulnar point of the scaphoid, so that any other point would be considered the most radial point on the radius
        max_points = []
        # Search quadrants 1 and 3 to determine the highest points on the dorsal rim of the radius for each slice
        for slice in range(len(q1_points)): # q1 is the first quadrant of the radius (the dorsal-radial quadrant)
            # Break the loop if q1_points is empty - this indicates that all slices have been processed
            if len(q1_points[slice]) == 0:
                continue
            max_point = find_max_point_along_axis(q1_points[slice], radius_origin, -radial_axes[0]) # Finding the most dorsal point in a given slice
            index = radius["Mesh"].find_closest_point(max_point) # Getting the index of the max point on the radius mesh
            scaphoid_point_projections = np.dot(max_point - max_scaphoid_point, radial_axes[1]) 
            if mean_curvature_array[index] > 0 and scaphoid_point_projections > 0: # If the curvature is positive and the point is more radial than the previously determined scaphoid point, add it to the list of max points
                max_points.append(max_point)
        for slice in range(len(q3_points)): # q3 is the third quadrant of the radius (the dorsal-ulnar quadrant)
            # Break the loop if q3_points is empty
            if len(q3_points[slice]) == 0:
                continue
            max_point = find_max_point_along_axis(q3_points[slice], radius_origin, -radial_axes[0])
            index = radius["Mesh"].find_closest_point(max_point)
            scaphoid_point_projections = np.dot(max_point - max_scaphoid_point, radial_axes[1])
            if mean_curvature_array[index] > 0 and scaphoid_point_projections > 0:
                max_points.append(max_point)

        if len(max_points) < (int(radius_width) - 5): # This suggests that the radius is tilted such that the highest points are in the facet of the radius rather than the dorsal rim, therefore not enough points were found
            print("Not enough points found on the rim. Searching in the global coordinate system.")
            # To combat this, search in the global coordinate system
            for slice in range(len(q1_points)):
                # Break the loop if q1_points is empty
                if len(q1_points[slice]) == 0:
                    continue
                max_point = find_max_point_along_axis(q1_points[slice], radius_origin, [0, 1, 0])
                index = radius["Mesh"].find_closest_point(max_point)
                scaphoid_point_projections = np.dot(max_point - max_scaphoid_point, radial_axes[1])
                if mean_curvature_array[index] > 0 and scaphoid_point_projections > 0:
                    max_points.append(max_point)

        # Finding the dorsal radius point from the collected max points
        dorsal_radius_point = find_max_point_along_axis(np.array(max_points), radius_origin, -radial_axes[2])
        dorsal_radius_point = create_point(dorsal_radius_point, 0.5)
        # If there is no dorsal radius point, find the closest point on the mesh to the scaphoid dorsal point. This is a fallback mechanism.
        if dorsal_radius_point is None:
            dorsal_radius_index, dorsal_radius_point = find_closest_point_on_mesh(radius["Mesh"], scaphoid_dorsal_point.GetCenter())
            dorsal_radius_point = create_point(dorsal_radius_point, 0.5)
        radius_rim_line = pv.Line(dorsal_radius_point.GetCenter() - 20 * radial_axes[0], dorsal_radius_point.GetCenter() + 20 * radial_axes[0])
        radius_rim_tube = radius_rim_line.tube(radius = 0.05)

        required_dst_components = [scaphoid_dorsal_point, scaphoid_dorsal_line, scaphoid_dorsal_tube, dorsal_radius_point, radius_rim_line, radius_rim_tube, max_points]
    
    elif dst_type == "PRSA":
        # This method requires identification of the highest points on the dorsal rim and on the palmar rim of the radius (posterior and anterior respectively, as stated by Gondim Teixeira AJR 2016)
        # The angle will be measured between this line and the line connecting the dorsal scaphoid point to the dorsal rim point
        max_scaphoid_point = find_max_point_along_axis(scaphoid["Mesh"].points, radius_origin, -radial_axes[1])
        dorsal_max_points = []
        palmar_max_points = []
        
        # Searching the dorsal quadrants (1 and 3) for the highest points on the dorsal rim
        for slice in range(len(q1_points)):
            # Break the loop if q1_points is empty
            if len(q1_points[slice]) == 0:
                continue
            dorsal_max_point = find_max_point_along_axis(q1_points[slice], radius_origin, -radial_axes[0])
            index = radius["Mesh"].find_closest_point(dorsal_max_point)
            scaphoid_point_projections = np.dot(dorsal_max_point - max_scaphoid_point, radial_axes[1])
            if mean_curvature_array[index] > 0 and scaphoid_point_projections > 0:
                dorsal_max_points.append(dorsal_max_point)
            else: continue
        for slice in range(len(q3_points)):
            # Break the loop if q3_points is empty
            if len(q3_points[slice]) == 0:
                continue
            dorsal_max_point = find_max_point_along_axis(q3_points[slice], radius_origin, -radial_axes[0])
            index = radius["Mesh"].find_closest_point(dorsal_max_point)
            scaphoid_point_projections = np.dot(dorsal_max_point - max_scaphoid_point, radial_axes[1])
            if mean_curvature_array[index] > 0 and scaphoid_point_projections > 0:
                dorsal_max_points.append(dorsal_max_point)
            else: continue

        if len(dorsal_max_points) < (int(radius_width) - 5): # This suggests that the radius is tilted such that the highest points are in the facet of the radius rather than the dorsal rim
            print("Not enough points found on the rim. Searching in the global coordinate system.")
            # To combat this, search in the global coordinate system
            for slice in range(len(q1_points)):
                # Break the loop if q1_points is empty
                if len(q1_points[slice]) == 0:
                    continue
                max_point = find_max_point_along_axis(q1_points[slice], radius_origin, [0, 1, 0])
                index = radius["Mesh"].find_closest_point(max_point)
                scaphoid_point_projections = np.dot(max_point - max_scaphoid_point, radial_axes[1])
                if mean_curvature_array[index] > 0 and scaphoid_point_projections > 0:
                    dorsal_max_points.append(max_point)

        # Finding the dorsal radius point from the collected dorsal max points
        dorsal_radius_point = find_max_point_along_axis(np.array(dorsal_max_points), radius_origin, -radial_axes[2])
        dorsal_y_value = np.dot(dorsal_radius_point - radius_origin, radial_axes[1])
        tolerance = 0.1
        for slice in range(len(q2_points)):
            for point in q2_points[slice]:
                vector = point - radius_origin
                index = radius["Mesh"].find_closest_point(point)
                curvature = mean_curvature_array[index]
                y_proj = np.dot(vector, radial_axes[1])
                if abs(y_proj - dorsal_y_value) <= tolerance and curvature > 0:
                    palmar_max_points.append(point)
        for slice in range(len(q4_points)):
            for point in q4_points[slice]:
                vector = point - radius_origin
                index = radius["Mesh"].find_closest_point(point)
                curvature = mean_curvature_array[index]
                y_proj = np.dot(vector, radial_axes[1])
                if abs(y_proj - dorsal_y_value) <= tolerance and curvature > 0:
                    palmar_max_points.append(point)
        palmar_radius_point = find_max_point_along_axis(np.array(palmar_max_points), radius_origin, -radial_axes[0])

        dorsal_radius_point = create_point(dorsal_radius_point, 0.5)
        palmar_radius_point = create_point(palmar_radius_point, 0.5)


        # Creating the lines and vectors required for PRSA measurement
        dorsalRadius_to_dorsalScaphoid_line = pv.Line(scaphoid_dorsal_point.GetCenter(), dorsal_radius_point.GetCenter())
        dorsalRadius_to_dorsalScaphoid_tube = dorsalRadius_to_dorsalScaphoid_line.tube(radius = 0.05)
        dorsalRadius_to_dorsalScaphoid_vector = np.array(scaphoid_dorsal_point.GetCenter()) - np.array(dorsal_radius_point.GetCenter())
        dorsalRadius_to_dorsalScaphoid_vector = dorsalRadius_to_dorsalScaphoid_vector / np.linalg.norm(dorsalRadius_to_dorsalScaphoid_vector)
        palmarRadius_to_dorsalRadius_line = pv.Line(palmar_radius_point.GetCenter(), dorsal_radius_point.GetCenter())
        palmarRadius_to_dorsalRadius_tube = palmarRadius_to_dorsalRadius_line.tube(radius = 0.05)
        palmarRadius_to_dorsalRadius_vector = np.array(palmar_radius_point.GetCenter()) - np.array(dorsal_radius_point.GetCenter())
        palmarRadius_to_dorsalRadius_vector = palmarRadius_to_dorsalRadius_vector / np.linalg.norm(palmarRadius_to_dorsalRadius_vector)
        dorsal_line_extended = pv.Line(scaphoid_dorsal_point.GetCenter() - 10 * radial_axes[0], dorsal_radius_point.GetCenter())

        # Creating the sagittal slice plane for the 2D PRSA measurement
        dorsal_y = np.dot(np.array(dorsal_radius_point.GetCenter()) - np.array(radius_origin), radial_axes[1])
        palmar_y = np.dot(np.array(palmar_radius_point.GetCenter()) - np.array(radius_origin), radial_axes[1])
        disp_y = dorsal_y
        slice_origin = radius_origin + disp_y * radial_axes[1]
        plane_normal = radial_axes[1]
        plane_i = radial_axes[0]
        plane_j = radial_axes[2]
        sagittal_slice = pv.Plane(center=slice_origin, direction=plane_normal, i_size=100, j_size=100, i_resolution=500, j_resolution=500)
        radius_slice = radius["Mesh"].slice(normal=plane_normal, origin=slice_origin)
        scaphoid_slice = scaphoid["Mesh"].slice(normal=plane_normal, origin=slice_origin)

        # Project the dorsal radius-scaphoid line and radius line onto the radius slice
        dorsal_line_projected = dorsal_line_extended.project_points_to_plane(origin=slice_origin, normal=plane_normal)
        dorsal_vector_projected = dorsal_line_projected.points[-1] - dorsal_line_projected.points[0]
        dorsal_vector_projected = dorsal_vector_projected / np.linalg.norm(dorsal_vector_projected)
        radius_line_projected = palmarRadius_to_dorsalRadius_line.project_points_to_plane(origin=slice_origin, normal=plane_normal)
        radius_vector_projected = radius_line_projected.points[-1] - radius_line_projected.points[0]
        radius_vector_projected = radius_vector_projected / np.linalg.norm(radius_vector_projected)

        required_dst_components = [dorsalRadius_to_dorsalScaphoid_tube, palmarRadius_to_dorsalRadius_tube, dorsal_radius_point, palmar_radius_point, scaphoid_dorsal_point, plane_normal, slice_origin, radius_slice, scaphoid_slice, dorsal_line_projected, radius_line_projected, dorsal_vector_projected, radius_vector_projected]

    return required_dst_components

def measure_dtl_dst(scaphoid_dorsal_line, radius_rim_line, radial_axes, radius_origin, sagittal_normal) -> tuple[pv.PolyData, pv.PolyData, pv.PolyData, pv.PolyData, pv.PolyData, float]:
    #this needs docstring to be filled in
    """This function will measure the dorsal tangential line method dorsal scaphoid translation.    
    Args:
        scaphoid_dorsal_line (_type_): _description_
        radius_rim_line (_type_): _description_
        radial_axes (_type_): _description_
        radius_origin (_type_): _description_
        sagittal_normal (_type_): _description_
    """ 
    scaphoid_dorsal_line_projected = scaphoid_dorsal_line.project_points_to_plane(origin = radius_origin, normal = sagittal_normal)
    radius_rim_line_projected = radius_rim_line.project_points_to_plane(origin = radius_origin, normal = sagittal_normal)

    line_start = radius_rim_line_projected.points[0]
    line_end = radius_rim_line_projected.points[1]
    line_middle = (line_start + line_end) / 2
    line_midpoint_pv = pv.PolyData(line_middle)
    line_midpoint = create_point(line_middle, 0.25)

    z_axis_projected = pv.Line(radial_axes[2], radial_axes[2] + 20 * radial_axes[2])
    z_axis_projected = z_axis_projected.project_points_to_plane(origin = radius_origin, normal = sagittal_normal)

    axis_start = line_middle
    axis_end = line_middle - 5 * radial_axes[2]
    z_axis_projected = pv.Line(axis_start, axis_end)

    x1, y1, z1 = scaphoid_dorsal_line_projected.points[0]
    x2, y2, z2 = scaphoid_dorsal_line_projected.points[-1]
    x3, y3, z3 = z_axis_projected.points[0]
    x4, y4, z4 = z_axis_projected.points[-1]

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    intersection_point = [x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1)]
    intersection_point = create_point(intersection_point, 0.1)

    dorsal_scaphoid_translation = np.linalg.norm(np.array(intersection_point.GetCenter()) - np.array(line_middle))
    if np.array(intersection_point.GetCenter())[2] > np.array(line_middle)[2]:
        dorsal_scaphoid_translation = -dorsal_scaphoid_translation

    return scaphoid_dorsal_line_projected, radius_rim_line_projected, line_midpoint_pv, z_axis_projected, intersection_point, dorsal_scaphoid_translation

def measure_prsa_dst(radius, radial_axes, scaphoid, dorsal_rs_line, radius_line, sagittal_normal) -> tuple[float, float, pv.PolyData, pv.PolyData, np.array, np.array]:
    """This function will measure the 2D and 3D PRSA based DST.

    Args:
        radius (_type_): _description_
        radial_axes (_type_): _description_
        scaphoid (_type_): _description_
        dorsal_rs_line (_type_): _description_
        radius_line (_type_): _description_
    """
    # First get the 3D angle between the dorsal RS line and the radius line
    dorsal_rs_line_vector = np.array(dorsal_rs_line.points[-1]) - np.array(dorsal_rs_line.points[0])
    dorsal_rs_line_vector = dorsal_rs_line_vector / np.linalg.norm(dorsal_rs_line_vector)
    radius_line_vector = np.array(radius_line.points[-1]) - np.array(radius_line.points[0])
    radius_line_vector = radius_line_vector / np.linalg.norm(radius_line_vector)
    PRSA_3D_DST_angle = calculate_3D_angle(dorsal_rs_line_vector, radius_line_vector)

    dorsal_rs_line_projected = dorsal_rs_line.project_points_to_plane(origin = radius["Origin"].GetCenter(), normal = sagittal_normal)
    radius_line_projected = radius_line.project_points_to_plane(origin = radius["Origin"].GetCenter(), normal = sagittal_normal)
    dorsal_rs_line_vector_projected = np.array(dorsal_rs_line_projected.points[0]) - np.array(dorsal_rs_line_projected.points[-1])
    dorsal_rs_line_vector_projected = dorsal_rs_line_vector_projected / np.linalg.norm(dorsal_rs_line_vector_projected)
    radius_line_vector_projected = np.array(radius_line_projected.points[0]) - np.array(radius_line_projected.points[-1])
    radius_line_vector_projected = radius_line_vector_projected / np.linalg.norm(radius_line_vector_projected)
    PRSA_2D_DST_angle = calculate_2D_angle(dorsal_rs_line_vector_projected, radius_line_vector_projected)

    return PRSA_3D_DST_angle, PRSA_2D_DST_angle, dorsal_rs_line_projected, radius_line_projected, dorsal_rs_line_vector_projected, radius_line_vector_projected