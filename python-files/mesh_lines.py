# mesh_lines.py
# contains functions to create and manipulate lines in relation to meshes:
# - get_mesh_lines: creates a line, tube, vector, normalized vector, and arrow between two points
# - count_line_mesh_interactions: counts the number of times a line intersects with a mesh and returns the intersection points
# - adjust_line: adjusts a line to be tangent to a bone mesh by searching around the closest points on the mesh to the original line start and end, and finding the pair of points that creates a line with the smallest dot product with the normals at the points

import numpy as np
import pyvista as pv
import vtk

from mesh_processing import create_point, build_mesh_adjacency_list, find_points_within_k_steps, calculate_vertex_normals
from point_finding import find_closest_point_on_mesh

def get_mesh_lines(start_point, end_point, length_multiplier) -> tuple[vtk.vtkPolyData, vtk.vtkPolyData, np.ndarray, np.ndarray, vtk.vtkPolyData]:
    """This creates a line, tube, vector, normalized vector, and arrow between two points.
    This is often used when defining axes for bones which are then used in carpal alignment angles.

    Args:
        start_point (numpy array): The coordinates of the start point of the line.
        end_point (numpy array): The coordinates of the end point of the line.

    Returns:
        line (vtk.vtkPolyData): The line between the two points.
        tube (vtk.vtkPolyData): The tube around the line.
        vector (numpy array): The vector between the two points.
        normalized_vector (numpy array): The normalized vector between the two points.
        arrow (vtk.vtkPolyData): The arrow pointing from the start point to the end point.
    """

    vector = np.array(end_point) - np.array(start_point)
    normalized_vector = vector / np.linalg.norm(vector)

    length = np.linalg.norm(vector)

    arrow = pv.Arrow(start = start_point, direction = normalized_vector, scale = 2.5)

    # Extending the line in both directions in the direction of the vector
    extended_start_point = start_point - normalized_vector * length * length_multiplier
    extended_end_point = end_point + normalized_vector * length * length_multiplier * 1.25

    line = pv.Line(extended_start_point, extended_end_point, resolution = 100)
    tube = pv.Line(extended_start_point, extended_end_point, resolution = 100).tube(radius = 0.1)

    return line, tube, vector, normalized_vector, arrow

def count_line_mesh_interactions(line_start, line_end, mesh) -> tuple[int, vtk.vtkPoints, list[vtk.vtkPolyData]]:
    """This will count the number of times a line intersects with a mesh.

    Args:
        line_start (numpy array): The start point of the line.
        line_end (numpy array): The end point of the line.
        mesh (vtk.vtkPolyData): The mesh that the line is intersecting with.

    Returns:
        number_of_points (int): The number of intersection points.
        intersection_points (vtk.vtkPoints): The intersection points.
    """
    line_source = vtk.vtkLineSource()
    line_source.SetPoint1(line_start)
    line_source.SetPoint2(line_end)
    line_source.Update()

    # Creating a mesh from the line source
    line_polydata = vtk.vtkPolyData()
    line_polydata.SetPoints(line_source.GetOutput().GetPoints())
    line_polydata.SetLines(line_source.GetOutput().GetLines())

    # Creating an OBB Tree - this is used to determine the intersection points between the line and the mesh
    obb_tree = vtk.vtkOBBTree()
    obb_tree.SetDataSet(mesh)
    obb_tree.BuildLocator()

    intersection_points = vtk.vtkPoints()
    obb_tree.IntersectWithLine(line_start, line_end, intersection_points, None) # Getting all intersection points between the line and the mesh

    # If there are multiple intersection points with the same coordinates, remove the duplicates -- this does not happen often but can occur with certain mesh geometries
    unique_points = vtk.vtkPoints()
    unique_points.InsertNextPoint(intersection_points.GetPoint(0))
    for i in range(1, intersection_points.GetNumberOfPoints()):
        if not np.allclose(intersection_points.GetPoint(i), intersection_points.GetPoint(i - 1)):
            unique_points.InsertNextPoint(intersection_points.GetPoint(i))
    intersection_points = unique_points

    number_of_points = 0
    # Printing the coordinates of the intersection points
    for i in range(intersection_points.GetNumberOfPoints()):
        # print("Intersection Point: ", intersection_points.GetPoint(i))
        number_of_points += 1

    # Create a point at each intersection using the create_point function
    intersection_points_meshes = [create_point(intersection_points.GetPoint(i), 0.1) for i in range(intersection_points.GetNumberOfPoints())]

    return number_of_points, intersection_points, intersection_points_meshes

def adjust_line(mesh, line_start, line_end, search_radius) -> tuple[np.ndarray, np.ndarray]:
    """This function will adjust a line to be tangent to a bone mesh.
    It will take the start and end of a line as the original input, and then adjust the line to be tangent to the bone mesh.

    Args:
        mesh (vtk.vtkPolyData): The bone surface mesh.
        line_start (numpy array): The start point of the line.
        line_end (numpy array): The end point of the line.

    Returns:
        new_line_start (numpy array): The new start point of the line.
        new_line_end (numpy array): The new end point of the line.
    """
    line_start_point = create_point(line_start, 0.5)
    line_end_point = create_point(line_end, 0.5)
    intersections, intersection_points, intersection_meshes = count_line_mesh_interactions(line_start_point.GetCenter(), line_end_point.GetCenter(), mesh) # This should return 2 intersections if the line is properly intersecting the mesh
    if intersections == 0: # This suggests that a bone axis is not connected to the bone surface mesh at all
        print("No intersection points found.")
        return line_start, line_end
    
    # Find the closest intersection point on the mesh to the original line start and end
    closest_start_id, closest_start = find_closest_point_on_mesh(mesh, line_start)
    closest_end_id, closest_end = find_closest_point_on_mesh(mesh, line_end)

    # Search around these points and evaluate the number of intersection points until a line with exactly two intersection points is found
    adjacency_list = build_mesh_adjacency_list(mesh)

    closest_points_to_start = find_points_within_k_steps(adjacency_list, closest_start_id, search_radius)
    closest_points_to_end = find_points_within_k_steps(adjacency_list, closest_end_id, search_radius)

    # Creating lists of the sets closest_points_to_start and closest_points_to_end
    closest_points_to_start = list(closest_points_to_start)
    closest_points_to_end = list(closest_points_to_end)

    vertex_normals = calculate_vertex_normals(mesh)

    a_zero = 1
    b_zero = 1

    # Traverse through each pair of points in the lists and find a pair of points that creates a line that runs tangent to the mesh
    # This can be done by calculating the dot product between the line and the normal of the mesh at the points
    # This block of code will find the pair of points that creates a line with the smallest dot product with the normals at the points - this aims to create a tangent line
    for i in closest_points_to_start:
        for j in closest_points_to_end:
            # Get the id's stored at the i and j indices, and create points at these id's
            # Create a point at the id of the point
            point_a = create_point(mesh.GetPoint(i), 0.5)
            point_b = create_point(mesh.GetPoint(j), 0.5)
            line = pv.Line(point_a.GetCenter(), point_b.GetCenter())
            vector = np.array(point_b.GetCenter()) - np.array(point_a.GetCenter())
            normalized_vector = vector / np.linalg.norm(vector)
            # Get the normals at the points using the vertex_normals list
            normal_a = vertex_normals[i]
            normal_a = normal_a / np.linalg.norm(normal_a)
            normal_b = vertex_normals[j]
            normal_b = normal_b / np.linalg.norm(normal_b)
            # Calculate the dot product between the line and the normal at the points
            dot_product_a = np.dot(normal_a, normalized_vector)
            dot_product_b = np.dot(normal_b, normalized_vector)
            
            # If the dot product is closer to zero than the previous value of a_zero and b_zero, replace these values respectively
            if abs(dot_product_a) < a_zero:
                a_zero = abs(dot_product_a)
                new_line_start = point_a.GetCenter()
            if abs(dot_product_b) < b_zero:
                b_zero = abs(dot_product_b)
                new_line_end = point_b.GetCenter()
    return new_line_start, new_line_end