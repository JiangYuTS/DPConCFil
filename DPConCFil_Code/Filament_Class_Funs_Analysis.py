import numpy as np
from skimage import measure, morphology
from scipy import ndimage
from collections import defaultdict,deque
from scipy.interpolate import splprep, splev, RegularGridInterpolator
import networkx as nx
from scipy.spatial.distance import cdist



def Filament_Coords(origin_data, regions_data, data_wcs, clump_coords_dict, related_ids_T, CalSub=False):
    """
    Extract and organize coordinates for a filamentary structure from the data.
    
    This function collects all coordinates associated with a filament (made up of multiple clumps),
    creates a padded box containing the filament, and transfers the data into this new coordinate system.
    
    Parameters:
    -----------
    origin_data : ndarray
        The original data cube containing intensity values
    regions_data : ndarray
        Integer mask identifying different regions/clumps
    data_wcs : astropy.wcs.WCS
        World Coordinate System object for coordinate transformations
    clump_coords_dict : dict
        Dictionary mapping clump IDs to their constituent pixel coordinates
    related_ids_T : list
        List of clump IDs that make up the filament being analyzed
    CalSub : bool, optional
        Flag to calculate sub-structure information
        
    Returns:
    --------
    filament_coords : ndarray
        Array of all pixel coordinates in the filament
    filament_item : ndarray
        3D intensity data cube containing just the filament
    data_wcs_item : astropy.wcs.WCS
        Updated WCS object for the new coordinate system
    regions_data_T : ndarray
        Mask identifying regions/clumps in the new coordinate system
    start_coords : ndarray
        Offset coordinates to convert between original and new system
    filament_item_mask_2D : ndarray
        2D projection mask of the filament
    lb_area : int or None
        Area of the filament in galactic longitude-latitude plane
    """
    # Combine all coordinates from multiple clumps into a single filament
    filament_coords = clump_coords_dict[related_ids_T[0]]
    for clump_id in related_ids_T[1:]:
        filament_coords = np.r_[filament_coords, clump_coords_dict[clump_id]]
    
    # Find the bounding box of the filament
    x_min = np.array(filament_coords[:, 0]).min()
    x_max = np.array(filament_coords[:, 0]).max()
    y_min = np.array(filament_coords[:, 1]).min()
    y_max = np.array(filament_coords[:, 1]).max()
    z_min = np.array(filament_coords[:, 2]).min()
    z_max = np.array(filament_coords[:, 2]).max()
    
    # Calculate length of box with padding (10% plus 5 or 6 pixels)
    length = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    if np.int32(length * 0.1) % 2 == 0:
        length += np.int32(length * 0.1) + 5
    else:
        length += np.int32(length * 0.1) + 6
    
    # Create empty arrays for the filament data and region masks
    filament_item = np.zeros([length, length, length])
    regions_data_i = np.zeros([length, length, length], dtype=np.int32)
    
    # Calculate starting positions for centering the filament in the new array
    start_x = np.int64((length - (x_max - x_min)) / 2)
    start_y = np.int64((length - (y_max - y_min)) / 2)
    start_z = np.int64((length - (z_max - z_min)) / 2)
    
    # Transfer data from original coordinates to new centered coordinates
    filament_item[filament_coords[:, 0] - x_min + start_x, filament_coords[:, 1] - y_min + start_y, \
                  filament_coords[:, 2] - z_min + start_z] = \
        origin_data[filament_coords[:, 0], filament_coords[:, 1], filament_coords[:, 2]]
    
    # Transfer region data to new coordinates
    regions_data_i[filament_coords[:, 0] - x_min + start_x, filament_coords[:, 1] - y_min + start_y, \
                   filament_coords[:, 2] - z_min + start_z] = \
        regions_data[filament_coords[:, 0], filament_coords[:, 1], filament_coords[:, 2]]
    
    # Save coordinate transformation reference
    start_coords = np.array([x_min - start_x, y_min - start_y, z_min - start_z])

    # Create a 2D mask of the filament (summed along velocity axis)
    filament_item_mask_2D = np.zeros_like(filament_item.sum(0), dtype=np.int32)
    filament_item_mask_2D[filament_coords[:, 1] - y_min + start_y, filament_coords[:, 2] - z_min + start_z] = 1
    
    if CalSub == False:
        # Calculate area in the longitude-latitude plane
        box_region = measure.regionprops(filament_item_mask_2D)
        lb_area = box_region[0].area
        
        # Update WCS information for the new coordinate system
        data_wcs_item = data_wcs.deepcopy()
        data_wcs_item.wcs.crpix[0] -= start_coords[2]  # Adjust reference pixel in longitude
        data_wcs_item.wcs.crpix[1] -= start_coords[1]  # Adjust reference pixel in latitude
        data_wcs_item.wcs.crpix[2] -= start_coords[0]  # Adjust reference pixel in velocity
    else:
        lb_area = None
        data_wcs_item = None
    
    return filament_coords, filament_item, data_wcs_item, regions_data_i, start_coords, filament_item_mask_2D, lb_area


def Get_Line_Coords_2D(point_a, point_b):
    """
    Calculate all pixel coordinates that form a line between two 2D points.
    
    This function uses a line drawing algorithm to determine which pixels lie on a line
    between two points, ensuring proper connectivity.
    
    Parameters:
    -----------
    point_a : array-like
        Starting point coordinates [x, y]
    point_b : array-like
        Ending point coordinates [x, y]
        
    Returns:
    --------
    line_coords : ndarray
        Array of [x, y] coordinates of pixels that form the line
    """
    coords = []
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    
    # Determine which dimension has the largest change to prioritize stepping along it
    sort_index = np.argsort(np.abs(point_b - point_a))
    point_a_temp = point_a[sort_index]
    point_b_temp = point_b[sort_index]
    
    x1, y1 = point_a_temp[0], point_a_temp[1]
    x2, y2 = point_b_temp[0], point_b_temp[1]
    kx, ky = (x2 - x1), (y2 - y1)
    k_norm = np.sqrt(kx ** 2 + ky ** 2)
    
    # Iterate along the dimension with largest change, calculating corresponding points
    for y in range(min(int(round(y1)), int(round(y2))), max(int(round(y1)), int(round(y2))) + 1):
        # Calculate x coordinate using the line equation
        x = x1 + kx * (y - y1) / ky
        coords.append((int(round(x)), int(round(y))))
    
    # Reorder coordinates to match original dimensions
    index_0 = np.where(sort_index == 0)[0]
    index_1 = np.where(sort_index == 1)[0]
    line_coords = np.c_[np.array(coords)[:, index_0], np.array(coords)[:, index_1]].astype('int')
    
    return line_coords


def Get_Line_Coords_3D(point_a, point_b):
    """
    Calculate all pixel coordinates that form a line between two 3D points.
    
    Similar to Get_Line_Coords_2D but for 3D space, determining which voxels lie on a line
    between two 3D points.
    
    Parameters:
    -----------
    point_a : array-like
        Starting point coordinates [x, y, z]
    point_b : array-like
        Ending point coordinates [x, y, z]
        
    Returns:
    --------
    line_coords : ndarray
        Array of [x, y, z] coordinates of voxels that form the line
    """
    coords = []
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    
    # Determine which dimension has the largest change to prioritize stepping along it
    sort_index = np.argsort(np.abs(point_b - point_a))
    point_a_temp = point_a[sort_index]
    point_b_temp = point_b[sort_index]
    
    x1, y1, z1 = point_a_temp[0], point_a_temp[1], point_a_temp[2]
    x2, y2, z2 = point_b_temp[0], point_b_temp[1], point_b_temp[2]
    kx, ky, kz = (x2 - x1), (y2 - y1), (z2 - z1)
    k_norm = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    
    # Iterate along the dimension with largest change, calculating corresponding points
    for z in range(min(int(round(z1)), int(round(z2))), max(int(round(z1)), int(round(z2))) + 1):
        # Calculate x and y coordinates using the 3D line equation
        x = x1 + kx * (z - z1) / kz
        y = y1 + ky * (z - z1) / kz
        coords.append((int(round(x)), int(round(y)), int(round(z))))
    
    # Reorder coordinates to match original dimensions
    index_0 = np.where(sort_index == 0)[0]
    index_1 = np.where(sort_index == 1)[0]
    index_2 = np.where(sort_index == 2)[0]
    coords_xy = np.c_[np.array(coords)[:, index_0], np.array(coords)[:, index_1]]
    line_coords = np.c_[coords_xy, np.array(coords)[:, index_2]].astype('int')
    
    return line_coords


def Get_DV(box_data, box_center):
    """
    Calculate the diagonalization of the inertia tensor for a 2D region.
    
    This function computes the eigenvalues and eigenvectors of the inertia tensor
    of a 2D distribution, which can be used to determine the principal axes and
    the elongation of the structure.
    
    Parameters:
    -----------
    box_data : ndarray
        3D data with the first dimension being summed over
    box_center : array-like
        Center coordinates of the box [v, b, l]
        
    Returns:
    --------
    D : ndarray
        Eigenvalues of the inertia tensor, sorted in descending order
    V : ndarray
        Eigenvectors corresponding to the eigenvalues
    size_ratio : float
        Ratio of the major to minor axis lengths (sqrt of eigenvalue ratio)
    angle : float
        Rotation angle in degrees of the major axis relative to the coordinate axes
    """
    # Sum over the first dimension (velocity axis) to get a 2D projection
    box_data_sum = box_data.sum(0)
    # Get coordinates of non-zero points in the 2D projection
    box_region = np.where(box_data_sum != 0)
    
    # Calculate components of the inertia tensor
    # A11 = sum of (y - y_center)^2 * intensity
    A11 = np.sum((box_region[0] - box_center[1]) ** 2 * \
                 box_data_sum[box_region])
    # A12 = -sum of (y - y_center) * (x - x_center) * intensity
    A12 = -np.sum((box_region[0] - box_center[1]) * \
                  (box_region[1] - box_center[2]) * \
                  box_data_sum[box_region])
    A21 = A12  # Symmetric tensor
    # A22 = sum of (x - x_center)^2 * intensity
    A22 = np.sum((box_region[1] - box_center[2]) ** 2 * \
                 box_data_sum[box_region])
    
    # Form the inertia tensor and normalize by number of points
    A = np.array([[A11, A12], [A21, A22]]) / len(box_region[0])
    
    # Calculate eigenvalues and eigenvectors
    D, V = np.linalg.eig(A)
    
    # Sort eigenvalues in descending order and arrange eigenvectors accordingly
    if D[0] < D[1]:
        D = D[[1, 0]]
        V = V[[1, 0]]
    
    # Ensure consistent orientation of eigenvectors
    if V[1][0] < 0 and V[0][0] > 0 and V[1][1] > 0:
        V = -V
    
    # Calculate the aspect ratio and the rotation angle
    size_ratio = np.sqrt(D[0] / D[1])
    angle = np.around(np.arccos(V[0][0]) * 180 / np.pi - 90, 2)
    
    return D, V, size_ratio, angle


def Dists_Array(matrix_1, matrix_2):
    """
    Calculate the pairwise Euclidean distances between two sets of points.
    
    This function efficiently computes distances between all pairs of points from
    two matrices without explicitly looping through them.
    
    Parameters:
    -----------
    matrix_1 : ndarray
        First set of points, shape (n, d) where n is number of points and d is dimensionality
    matrix_2 : ndarray
        Second set of points, shape (m, d)
        
    Returns:
    --------
    dists : ndarray
        Distance matrix of shape (n, m) containing pairwise distances
    """
    matrix_1 = np.array(matrix_1)
    matrix_2 = np.array(matrix_2)
    
    # Compute squared distances using the formula:
    # dist^2(x,y) = ||x||^2 + ||y||^2 - 2*x·y
    # First term: -2 * dot product
    dist_1 = -2 * np.dot(matrix_1, matrix_2.T)
    # Second term: sum of squares of first matrix
    dist_2 = np.sum(np.square(matrix_1), axis=1, keepdims=True)
    # Third term: sum of squares of second matrix
    dist_3 = np.sum(np.square(matrix_2), axis=1)
    
    # Add a tiny random component to avoid identical distances
    random_dist = np.random.random(dist_1.shape) / 1000000
    
    # Combine all terms to get squared distances
    dist_temp = dist_1 + dist_2 + dist_3 + random_dist
    
    # Take square root to get Euclidean distances
    dists = np.sqrt(dist_temp)
    
    return dists


def Graph_Infor(points):
    """
    Create a graph and minimum spanning tree from a set of points.
    
    This function builds a complete graph where each point is connected to all others,
    with edge weights equal to the Euclidean distances between points. It then finds
    the minimum spanning tree of this graph.
    
    Parameters:
    -----------
    points : ndarray
        Array of point coordinates
        
    Returns:
    --------
    Graph : networkx.Graph
        Complete graph with all points connected
    Tree : networkx.Graph
        Minimum spanning tree of the graph
    """
    # Get number of points
    n_points = len(points)
    
    # Calculate all pairwise distances
    dist_matrix = Dists_Array(points, points)
    
    # Create a new graph
    Graph = nx.Graph()
    
    # Add all edges with distances as weights
    for i in range(n_points):
        for j in range(i + 1, n_points):
            Graph.add_edge(i, j, weight=dist_matrix[i, j])
    
    # Find the minimum spanning tree
    Tree = nx.minimum_spanning_tree(Graph)
    
    return Graph, Tree


def Graph_Infor_Connected(points):
    """
    Create a graph and minimum spanning tree connecting only nearby points.
    
    Unlike Graph_Infor, this function only connects points that are between 0.5 and 2
    distance units apart, creating a graph of local connections.
    
    Parameters:
    -----------
    points : ndarray
        Array of point coordinates
        
    Returns:
    --------
    Graph : networkx.Graph
        Graph with only nearby points connected
    Tree : networkx.Graph
        Minimum spanning tree of the graph
    """
    # Calculate all pairwise distances
    dist_matrix = Dists_Array(points, points)
    
    # Find pairs of points with distances between 0.5 and 2 units
    mask_coords_in_dm = np.where(np.logical_and(dist_matrix > 0.5, dist_matrix < 2))
    
    # Create a new graph
    Graph = nx.Graph()
    
    # Add edges only between nearby points with uniform weight
    for i, j in zip(mask_coords_in_dm[0], mask_coords_in_dm[1]):
        weight_ij = 1  # uniform weight instead of using dist_matrix[i,j]
        Graph.add_edge(i, j, weight=weight_ij)
    
    # Find the minimum spanning tree
    Tree = nx.minimum_spanning_tree(Graph)
    
    return Graph, Tree


def Graph_Infor_SubStructure(origin_data, filament_mask_2D, filament_centers_LBV, filament_clumps_id, connected_ids_dict):
    """
    Create a weighted graph connecting substructures within a filament.
    
    This function builds a graph connecting clumps that are part of a filament,
    with edge weights based on spatial distance, velocity difference, and intensity.
    It only connects clumps that are physically connected according to the connected_ids_dict.
    
    Parameters:
    -----------
    origin_data : ndarray
        Original 3D data cube with intensity values
    filament_mask_2D : ndarray
        2D mask showing the filament's projection
    filament_centers_LBV : ndarray
        Coordinates of clump centers in (l,b,v) format
    filament_clumps_id : list
        IDs of clumps in the filament
    connected_ids_dict : dict
        Dictionary indicating which clumps are connected to each other
        
    Returns:
    --------
    Graph : networkx.Graph
        Graph connecting substructures within the filament
    Tree : networkx.Graph
        Minimum spanning tree of the graph
    """
    # Extract velocity and spatial coordinates
    points_V = filament_centers_LBV[:, 2]  # Velocity coordinates
    points_LB = np.c_[filament_centers_LBV[:, 0], filament_centers_LBV[:, 1]]  # Spatial coordinates
    
    n_points = len(filament_centers_LBV)
    
    # Calculate spatial distances between clump centers
    dist_matrix = Dists_Array(points_LB, points_LB)
    
    # Convert to integer coordinates for array indexing
    filament_centers_LBV = np.int64(np.around(filament_centers_LBV))
    
    # Create a new graph
    Graph = nx.Graph()
    
    # Connect only clumps that are neighbors according to connected_ids_dict
    for i in range(n_points):
        # Get IDs of clumps connected to the current clump
        neighboring_ids_T = connected_ids_dict[filament_clumps_id[i]]
        neighboring_ids = neighboring_ids_T.copy()
        
        # Create a unique set of neighboring IDs
        neighboring_ids = list(set(neighboring_ids))
        
        for j in range(i + 1, n_points):
            # Check if clump j is connected to clump i
            if filament_clumps_id[j] in neighboring_ids:
                # Get 2D coordinates for line between clumps
                line_coords = Get_Line_Coords_2D(points_LB[i], points_LB[j])
                # Check if line passes through the filament mask
                mask_2D_ids = filament_mask_2D[line_coords[:, 1], line_coords[:, 0]]
                
                # Get 3D coordinates for line between clumps
                line_coords = Get_Line_Coords_3D(filament_centers_LBV[i], filament_centers_LBV[j])
                # Calculate mean intensity along the line
                weight_ij = origin_data[line_coords[:, 2], line_coords[:, 1], line_coords[:, 0]].mean()
                
                # Only add edge if line is entirely within the filament
                if 0 not in mask_2D_ids:
                    # Weight is spatial distance * velocity difference / intensity
                    # (higher intensity means lower weight = preferred connection)
                    weight = dist_matrix[i, j] * np.abs(points_V[i] - points_V[j]) / (weight_ij)
                    Graph.add_edge(i, j, weight=weight)
    
    # Find the minimum spanning tree
    Tree = nx.minimum_spanning_tree(Graph)
    
    return Graph, Tree


def Get_Max_Path_Node(T):
    """
    Find the longest path in a tree based on number of nodes.
    
    This function identifies the path between two leaf nodes (degree 1) that
    contains the maximum number of nodes, representing the "backbone" of the tree.
    
    Parameters:
    -----------
    T : networkx.Graph
        A tree (acyclic connected graph)
        
    Returns:
    --------
    max_path : list
        List of node indices forming the longest path
    max_edges : list
        List of edge pairs (node1, node2) in the longest path
    """
    # Find all leaf nodes (nodes with only one connection)
    degree1_nodes = [node for node in T.nodes if T.degree(node) == 1]
    
    max_path = []
    max_num_nodes = -float('inf')
    
    # Check all pairs of leaf nodes to find the longest path
    for i in range(len(degree1_nodes) - 1):
        for j in range(i + 1, len(degree1_nodes)):
            if nx.has_path(T, degree1_nodes[i], degree1_nodes[j]):
                # Get shortest path between leaves (in a tree, there's only one path)
                path = nx.shortest_path(T, degree1_nodes[i], degree1_nodes[j])
                num_nodes = len(path)
                
                # Update if this path is longer than current maximum
                if num_nodes > max_num_nodes:
                    max_num_nodes = num_nodes
                    max_path = path
    
    # Convert path to list of edges
    max_edges = [(max_path[i], max_path[i + 1]) for i in range(len(max_path) - 1)]
    
    return max_path, max_edges


def Get_Max_Path_Weight(T):
    """
    Find the longest path in a tree based on accumulated edge weights.
    
    Similar to Get_Max_Path_Node, but uses edge weights instead of node count
    to determine the longest path.
    
    Parameters:
    -----------
    T : networkx.Graph
        A tree (acyclic connected graph) with edge weights
        
    Returns:
    --------
    max_path : list
        List of node indices forming the path with maximum total weight
    max_edges : list
        List of edge pairs (node1, node2) in the maximum weight path
    """
    # Find all leaf nodes (nodes with only one connection)
    degree1_nodes = [node for node in T.nodes if T.degree(node) == 1]
    
    paths_and_weights = []
    max_path = []
    max_edges = []
    
    # Check all pairs of leaf nodes to find the path with maximum total weight
    for i in range(len(degree1_nodes) - 1):
        for j in range(i + 1, len(degree1_nodes)):
            if nx.has_path(T, degree1_nodes[i], degree1_nodes[j]):
                # Get shortest path between leaves (in a tree, there's only one path)
                path = nx.shortest_path(T, degree1_nodes[i], degree1_nodes[j])
                
                # Calculate total weight of the path
                path_weight = 0
                for k in range(len(path) - 1):
                    path_weight += T[path[k]][path[k + 1]]['weight']
                
                paths_and_weights.append((path, path_weight))
    
    # Find the path with maximum weight
    if len(paths_and_weights) != 0:
        max_weight = max([weight for path, weight in paths_and_weights])
        max_path = [path for path, weight in paths_and_weights if weight == max_weight][0]
        max_edges = [(max_path[i], max_path[i + 1]) for i in range(len(max_path) - 1)]
    
    return max_path, max_edges


def Get_Max_Path_Weight_SubStructure(origin_data, filament_centers_LBV, T, Tree_0, sub_tree):
    """
    Find the maximum weighted path considering substructure characteristics.
    
    This function finds the path between leaf nodes with the highest total weight,
    where weight incorporates spatial distance, velocity difference, and intensity
    along the connecting lines.
    
    Parameters:
    -----------
    origin_data : ndarray
        Original 3D data cube with intensity values
    filament_centers_LBV : ndarray
        Coordinates of clump centers in (l,b,v) format
    T : networkx.Graph
        A tree (acyclic connected graph) with edge weights
        
    Returns:
    --------
    max_path : list
        List of node indices forming the path with maximum total weight
    max_edges : list
        List of edge pairs (node1, node2) in the maximum weight path
    """
    # Extract velocity and spatial coordinates
    points_V = filament_centers_LBV[:, 2]  # Velocity coordinates
    points_LB = np.c_[filament_centers_LBV[:, 0], filament_centers_LBV[:, 1]]  # Spatial coordinates
    
    # Calculate spatial distances between clump centers
    dist_matrix = Dists_Array(points_LB, points_LB)
    
    # Convert to integer coordinates for array indexing
    filament_centers_LBV = np.int64(np.around(filament_centers_LBV))

    # Find all leaf nodes (nodes with only one connection)
    degree1_nodes = [node for node in T.nodes if T.degree(node) == 1]

    common_nodes = set(T.nodes()) & set(Tree_0.nodes()) 
    common_degrees_in_Tree0 = {node: Tree_0.degree(node) for node in common_nodes}
    max_degree = max(common_degrees_in_Tree0.values()) 
    max_degree_nodes = [node for node, deg in common_degrees_in_Tree0.items() if deg == max_degree]

    if not sub_tree:
        max_degree_nodes = degree1_nodes
    
    paths_and_weights = []
    max_path = []
    max_edges = []
    
    # Check all pairs of leaf nodes to find the path with maximum total weight
    for i in range(len(degree1_nodes)):
        for j in range(len(max_degree_nodes)):
            if nx.has_path(T, degree1_nodes[i], max_degree_nodes[j]):
                # Get shortest path between leave and leave with max degree (in a tree, there's only one path)
                path = nx.shortest_path(T, degree1_nodes[i], max_degree_nodes[j]) 
                
                # Calculate total weight of the path incorporating intensity
                path_weight = 0
                for k in range(len(path) - 1):
                    # Get 3D coordinates for line between consecutive nodes
                    line_coords = Get_Line_Coords_3D(
                        filament_centers_LBV[path[k]], filament_centers_LBV[path[k + 1]])
                    
                    # Calculate mean intensity along the line
                    weight_ij = origin_data[line_coords[:, 2], line_coords[:, 1], line_coords[:, 0]].mean()
                    
                    # Weight is spatial distance * velocity difference * intensity
                    path_weight += dist_matrix[path[k], path[k + 1]] * np.abs(
                        points_V[path[k]] - points_V[path[k + 1]]) * weight_ij
                
                paths_and_weights.append((path, path_weight))
    
    # Find the path with maximum weight
    if len(paths_and_weights) != 0:
        max_weight = max([weight for path, weight in paths_and_weights])
        max_path = [path for path, weight in paths_and_weights if weight == max_weight][0]
        max_edges = [(max_path[i], max_path[i + 1]) for i in range(len(max_path) - 1)]
    
    return max_path, max_edges


def Get_Max_Path_Recursion(origin_data, filament_centers_LBV, max_path_record, max_edges_record, G, T , Tree_0 ,sub_tree=False):
    """
    Recursively find all significant paths in a tree representing a filament.
    
    This function identifies the main path (backbone) of a filament, removes it,
    and then recursively finds additional significant paths in the remaining parts.
    This helps map complex filamentary structures with branches.
    
    Parameters:
    -----------
    origin_data : ndarray
        Original 3D data cube with intensity values
    filament_centers_LBV : ndarray
        Coordinates of clump centers in (l,b,v) format
    max_path_record : list
        List to accumulate all identified significant paths
    max_edges_record : list
        List to accumulate edges corresponding to the paths
    G : networkx.Graph
        The original complete graph
    T : networkx.Graph
        The current tree being analyzed
        
    Returns:
    --------
    max_path_record : list
        Updated list containing all identified significant paths
    max_edges_record : list
        Updated list containing all edges in the paths
    """
    # Find the maximum weight path in the current tree
    max_path, max_edges = Get_Max_Path_Weight_SubStructure(origin_data, filament_centers_LBV, T , Tree_0 ,sub_tree)
    
    # Record this path and its edges
    max_path_record.append(max_path)
    max_edges_record.append(max_edges)
    
    # Create a copy of the tree and remove the identified path edges
    new_T = T.copy()
    for i in range(len(max_path) - 1):
        new_T.remove_edge(max_path[i], max_path[i + 1])
    # for i in np.arange(0,len(max_path)-1,2):
    #     Tree_0.remove_edge(max_path[i], max_path[i+1])

    # Get degrees of remaining nodes
    sub_degrees = new_T.degree
    nodes_T = new_T.nodes
    
    # Create another copy to remove isolated nodes
    new_T_2 = new_T.copy()
    nodes = nodes_T
    for node in nodes:
        if sub_degrees[node] == 0:
            new_T_2.remove_node(node)

    # Check if there are still connected components in the remaining graph
    if new_T_2.nodes != 0:
        # Find all connected components (potential sub-filaments)
        subgraphs = list(nx.connected_components(new_T_2))
        
        # Process each subgraph recursively
        for subgraph in subgraphs:
            # Create subgraph from the original graph
            new_T_3 = G.subgraph(subgraph)
            T = new_T_3.copy()
            
            # Remove edges that aren't in the current tree
            for sub_edge in T.edges:
                if sub_edge not in new_T_2.edges:
                    T.remove_edge(sub_edge[0], sub_edge[1])
            
            # Recursively find paths in this subgraph
            max_path_record, max_edges_record = \
                Get_Max_Path_Recursion(origin_data, filament_centers_LBV, max_path_record, max_edges_record, G, T, Tree_0, sub_tree=True)
    
    return max_path_record, max_edges_record


def Extend_Skeleton_Coords(skeleton_coords, filament_mask_2D):
    """
    Extend the skeleton coordinates at endpoints based on local direction.
    
    This function identifies the endpoints of a skeleton and extends them in the 
    direction of the skeleton until reaching the edge of the filament mask.
    This helps ensure the skeleton spans the entire filament.
    
    Parameters:
    -----------
    skeleton_coords : ndarray
        Coordinates of the current skeleton pixels
    filament_mask_2D : ndarray
        2D mask showing the extent of the filament
        
    Returns:
    --------
    skeleton_coords : ndarray
        Updated skeleton coordinates with extended endpoints
    """
    add_coords = []
    
    # Create a graph from the skeleton coordinates to identify endpoints
    G_longest_skeleton, T_longest_skeleton = Graph_Infor(skeleton_coords)
    
    # Find nodes with degree 1 (endpoints)
    end_points = np.array(T_longest_skeleton.degree)[:, 0][
        np.where(np.array(T_longest_skeleton.degree)[:, 1] == 1)[0]]
    
    # Process each endpoint
    for end_point_id in range(len(end_points)):
        # Get current endpoint and its neighborhood
        current_node_0 = end_points[end_point_id]
        
        # Get the next node along the skeleton
        current_node_1 = list(T_longest_skeleton.neighbors(current_node_0))
        
        # Get the next-next node (2 steps from endpoint)
        current_node_2 = list(T_longest_skeleton.neighbors(current_node_1[0]))
        current_node_2.remove(current_node_0)
        
        # Try to get a third node if possible (3 steps from endpoint)
        current_node_3 = list(T_longest_skeleton.neighbors(current_node_2[0]))
        current_node_3.remove(current_node_1[0])
        
        # Determine how many points to use for direction calculation
        if len(current_node_3) == 1:
            used_points = [current_node_0, current_node_1[0], current_node_2[0], current_node_3[0]]
        else:
            used_points = [current_node_0, current_node_1[0], current_node_2[0]]
        
        used_points_len = len(used_points)
        
        # Calculate the average direction at the endpoint
        direction = np.array([0, 0])
        for i, j in zip(range(used_points_len - 1), range(1, used_points_len)):
            direction += skeleton_coords[used_points[i]] - skeleton_coords[used_points[j]]
        direction = direction / (used_points_len - 1)
        
        # Start from the endpoint and extend in the calculated direction
        add_coord = skeleton_coords[end_points[end_point_id]].astype('float')
        while True:
            add_coord += direction
            
            # Stop if we go outside the image bounds
            if not (0 <= round(add_coord[0]) < filament_mask_2D.shape[0] and \
                    0 <= round(add_coord[1]) < filament_mask_2D.shape[1]):
                break
                
            # Stop if we reach the edge of the filament mask
            if filament_mask_2D[round(add_coord[0]), round(add_coord[1])] == 0:
                break
                
            # Add this point to our extended coordinates
            add_coords.append(list(np.around(add_coord, 0).astype('int')))
    
    # Combine original and additional coordinates
    if len(add_coords) != 0:
        skeleton_coords = np.r_[skeleton_coords, np.array(add_coords)]
        # Remove any duplicate coordinates
        skeleton_coords = np.array(list(set(list(map(tuple, skeleton_coords)))))
    
    return skeleton_coords


def Get_Longest_Skeleton_Coords(filament_mask_2D):
    """
    Extract coordinates of the skeleton (centerline) of a filament mask.
    
    This function applies morphological operations to extract the skeleton/centerline
    of a 2D filament mask, which represents the spine of the filament.
    
    Parameters:
    -----------
    filament_mask_2D : ndarray
        2D binary mask of the filament
        
    Returns:
    --------
    all_skeleton_coords : ndarray
        Coordinates of all pixels in the skeleton
    """
    # Apply morphological closing to fill small gaps
    closing_mask = morphology.closing(filament_mask_2D, morphology.square(3))
    
    # Extract the skeleton (centerline) of the mask
    skeleton = morphology.skeletonize(closing_mask)
    
    # Find all connected components in the skeleton
    props = measure.regionprops(measure.label(skeleton))
    
    # Get coordinates of the first component
    all_skeleton_coords = props[0].coords
    
    # Add coordinates from any additional components
    for skeleton_index in range(1, len(props)):
        all_skeleton_coords = np.r_[all_skeleton_coords, props[skeleton_index].coords]
    
    return all_skeleton_coords


def Get_Single_Filament_Skeleton(filament_mask_2D):
    """
    Extract and process the main skeleton of a filament.
    
    This function finds the skeleton of a filament, extends its endpoints,
    and extracts the longest continuous path through the skeleton.
    
    Parameters:
    -----------
    filament_mask_2D : ndarray
        2D binary mask of the filament
        
    Returns:
    --------
    skeleton_coords_2D : ndarray
        Coordinates of the main skeletal path
    filament_skeleton : ndarray
        Binary image of the skeleton
    all_skeleton_coords : ndarray
        Coordinates of all skeleton pixels including branches
    """
    # Get the initial skeleton coordinates
    all_skeleton_coords = Get_Longest_Skeleton_Coords(filament_mask_2D)
    
    # Extend skeleton if it has enough points
    if len(all_skeleton_coords) > 4:
        all_skeleton_coords = Extend_Skeleton_Coords(all_skeleton_coords, filament_mask_2D)
    
    # Create a graph from the skeleton coordinates
    G_longest_skeleton, T_longest_skeleton = Graph_Infor_Connected(all_skeleton_coords)
    
    # Find the longest path through the skeleton
    max_path, max_edges = Get_Max_Path_Weight(T_longest_skeleton)
    
    # Extract coordinates of the longest path
    single_longest_skeleton_coords = all_skeleton_coords[max_path]
    skeleton_coords_2D = single_longest_skeleton_coords
    
    # Create a binary image of the skeleton
    filament_skeleton = np.zeros_like(filament_mask_2D)
    filament_skeleton[skeleton_coords_2D[:, 0], skeleton_coords_2D[:, 1]] = 1
    
    # Trim and refine the skeleton coordinates
    skeleton_coords_2D, small_sc = Trim_Skeleton_Coords_2D(skeleton_coords_2D)
    
    return skeleton_coords_2D, filament_skeleton, all_skeleton_coords


def Cal_B_Spline(SampInt, skeleton_coords_2D, fil_mask):
    """
    Calculate a B-spline representation of the skeleton.
    
    This function fits a smooth B-spline to the skeleton coordinates and
    samples points along it at regular intervals, ensuring they stay within
    the filament mask.
    
    Parameters:
    -----------
    SampInt : int
        Sampling interval along the spline
    skeleton_coords_2D : ndarray
        Coordinates of the skeleton pixels
    fil_mask : ndarray
        Binary mask of the filament
        
    Returns:
    --------
    points : ndarray
        Sampled points along the B-spline
    fprime : ndarray
        Derivatives (tangent vectors) at the sampled points
    """
    # Add tiny random offsets to avoid duplicate coordinates
    random_coords = np.random.random(skeleton_coords_2D.shape) / 1000000 - np.random.random(
        skeleton_coords_2D.shape) / 1000000
    skeleton_coords_2D = skeleton_coords_2D + random_coords
    
    # Extract x, y coordinates
    x, y = skeleton_coords_2D[:, 0], skeleton_coords_2D[:, 1]
    
    # Fit a B-spline to the coordinates
    tckp, up = splprep([x, y], nest=-1)
    
    # Evaluate the spline and its derivative at the original parametrization
    xspline, yspline = splev(up, tckp)
    xprime, yprime = splev(up, tckp, der=1)
    
    # Filter points to ensure they're within image bounds
    logic_used_spline = (xspline > 0) * (xspline < fil_mask.shape[0] - 0.5) * (yspline > 0) * (
                yspline < fil_mask.shape[1] - 0.5)
    
    xspline = xspline[logic_used_spline]
    yspline = yspline[logic_used_spline]
    xprime = xprime[logic_used_spline]
    yprime = yprime[logic_used_spline]
    
    # Sample points at the specified interval and filter to keep only those in the mask
    pts_mask = (fil_mask[np.round(xspline[0:-1:SampInt]).astype(int), np.round(yspline[0:-1:SampInt]).astype(int)])
    
    xspline = xspline[0:-1:SampInt][pts_mask]
    yspline = yspline[0:-1:SampInt][pts_mask]
    xprime = xprime[0:-1:SampInt][pts_mask]
    yprime = yprime[0:-1:SampInt][pts_mask]
    
    # Return points and their derivatives
    points = np.c_[yspline, xspline]
    fprime = np.c_[yprime, xprime]
    
    return points, fprime


def Profile_Builder(image, mask, point, derivative, shift=True, fold=False):
    '''
    Build a profile perpendicular to the filament at a given point.
    
    This function is adapted from RadFil. It measures the intensity profile
    perpendicular to the filament's spine at a specified point.
    
    Parameters:
    -----------
    image : ndarray
        2D image containing the filament
    mask : ndarray
        Binary mask of the filament
    point : array-like
        Coordinates of the point on the spine to measure the profile
    derivative : array-like
        Direction vector of the spine at the point
    shift : bool
        If True, center the profile at the peak intensity rather than the spine point
    fold : bool
        If True, fold the profile to average both sides
        
    Returns:
    --------
    final_dist : ndarray
        Distances from the center (peak or spine point) along the profile
    image_line_T : ndarray
        Intensity values along the profile
    peak : ndarray
        Coordinates of the intensity peak
    (start, end) : tuple
        Coordinates of the start and end points of the profile
    final_dist_com : ndarray
        Distances from the center of mass along the profile
    com : ndarray
        Coordinates of the center of mass
    '''
    # Read the point and double check whether it's inside the mask.
    x0, y0 = point
    if (not mask[int(round(y0)), int(round(x0))]):
        raise ValueError("The point is not in the mask.")

    # Create the grid to calculate where the profile cut crosses edges of the
    # pixels.
    shapex, shapey = image.shape[1], image.shape[0]
    edgex, edgey = np.arange(.5, shapex - .5, 1.), np.arange(.5, shapey - .5, 1.)

    # Extreme cases when the derivative is (1, 0) or (0, 1)
    if (derivative[0] == 0) or (derivative[1] == 0):
        if (derivative[0] == 0) and (derivative[1] == 0):
            raise ValueError("Both components of the derivative are zero; unable to derive a tangent.")
        elif (derivative[0] == 0):
            y_edgex = []
            edgex = []
            x_edgey = np.ones(len(edgey)) * x0
        elif (derivative[1] == 0):
            y_edgex = np.ones(len(edgex)) * y0
            x_edgey = []
            edgey = []

    ## The regular cases go here: calculate the crossing points of the cut and the grid.
    else:
        # Calculate perpendicular slope: if spine direction is (dx,dy), perpendicular is (-dy,dx)
        # Here we use -1/(dy/dx) which gives the same result
        slope = -1. / (derivative[1] / derivative[0])
        
        # Calculate where the perpendicular line crosses the pixel grid
        y_edgex = slope * (edgex - x0) + y0
        x_edgey = (edgey - y0) / slope + x0

        ### Mask out points outside the image.
        pts_maskx = ((np.round(x_edgey) >= 0.) & (np.round(x_edgey) < shapex))
        pts_masky = ((np.round(y_edgex) >= 0.) & (np.round(y_edgex) < shapey))

        edgex, edgey = edgex[pts_masky], edgey[pts_maskx]
        y_edgex, x_edgey = y_edgex[pts_masky], x_edgey[pts_maskx]

    # Sort the points to find the center of each segment inside a single pixel.
    ## This also deals with when the cut crosses at the 4-corner point(s).
    ## The sorting is done by sorting the x coordinates
    stack = sorted(list(set(zip(np.concatenate([edgex, x_edgey]), \
                                np.concatenate([y_edgex, edgey])))))
    coords_total = stack[:-1] + .5 * np.diff(stack, axis=0)
    
    ## Extract the values from the image and the original mask
    # Setup interpolation
    xgrid = np.arange(0.5, image.shape[1] + 0.5, 1.0)
    ygrid = np.arange(0.5, image.shape[0] + 0.5, 1.0)
    interpolator = RegularGridInterpolator((xgrid, ygrid), image.T, bounds_error=False, fill_value=None)

    # Get intensity values along the profile using interpolation
    image_line = interpolator(coords_total)
    
    # Get mask values along the profile
    mask_line = mask[np.round(coords_total[:, 1]).astype(int), np.round(coords_total[:, 0]).astype(int)]
    
    # Select the part of the mask that includes the original point
    mask_p0 = (np.round(coords_total[:, 0]).astype(int) == int(round(x0))) & \
              (np.round(coords_total[:, 1]).astype(int) == int(round(y0)))
    
    # Get connected region containing the spine point
    mask_line = (morphology.label(mask_line) == morphology.label(mask_line)[mask_p0])

    # Extract the profile from the image.
    ## For the points within the original mask; to find the peak
    if derivative[1] < 0.:
        image_line0 = image_line[mask_line][::-1]
    else:
        image_line0 = image_line[mask_line]
    
    ## For the entire map
    # This is different from RadFil. It will improve the accuracy of the profiles, especially for curving skeleton.
    image_line_T = np.zeros_like(image_line)
    image_line_T[mask_line] = image_line[mask_line]
    
    if derivative[1] < 0.:
        coords_total = coords_total[::-1]
        mask_line = mask_line[::-1]
        mask_p0 = mask_p0[::-1]
        image_line_T = image_line_T[::-1]
    else:
        image_line_T = image_line_T

    # Plot.
    peak_finder = coords_total[mask_line]
    ## Find the end points of the cuts (within the original mask)
    start, end = peak_finder[0], peak_finder[-1]

    ## Find the peak here
    xpeak, ypeak = peak_finder[image_line0 == np.nanmax(image_line0)][0]
    ## The peak mask is used to determine where to unfold when shift = True
    mask_peak = (np.round(coords_total[:, 0]).astype(int) == int(round(xpeak))) & \
                (np.round(coords_total[:, 1]).astype(int) == int(round(ypeak)))

    # Calculate center of mass
    mass_array = np.c_[image_line0, image_line0]
    xcom, ycom = np.around((np.c_[mass_array] * peak_finder).sum(0) \
                           / image_line0.sum(), 3).tolist()
    
    mask_com = (np.round(coords_total[:, 0]).astype(int) == int(round(xcom))) & \
               (np.round(coords_total[:, 1]).astype(int) == int(round(ycom)))
    
    # Shift profile to be centered on the peak
    if shift:
        final_dist = np.hypot(coords_total[:, 0] - xpeak, coords_total[:, 1] - ypeak)
        # Unfold (make distances negative on one side of peak)
        pos0 = np.where(mask_peak)[0][0]
        final_dist[:pos0] = -final_dist[:pos0]
    else:
        final_dist = np.hypot(coords_total[:, 0] - x0, coords_total[:, 1] - y0)
        # Unfold (make distances negative on one side of original point)
        pos0 = np.where(mask_p0)[0][0]
        final_dist[:pos0] = -final_dist[:pos0]

    # Fold (make all distances positive for symmetric profile)
    if fold:
        final_dist = abs(final_dist)

    # Calculate distances from center of mass
    if shift:
        final_dist_com = np.hypot(coords_total[:, 0] - xcom, coords_total[:, 1] - ycom)
        # Unfold
        if len(np.where(mask_com)[0]) == 0:
            pos0 = np.where(mask_p0)[0][0]
        else:
            pos0 = np.where(mask_com)[0][0]
        final_dist_com[:pos0] = -final_dist_com[:pos0]
    else:
        final_dist_com = np.hypot(coords_total[:, 0] - x0, coords_total[:, 1] - y0)
        # Unfold
        if len(np.where(mask_com)[0]) == 0:
            pos0 = np.where(mask_p0)[0][0]
        else:
            pos0 = np.where(mask_com)[0][0]
        final_dist_com[:pos0] = -final_dist_com[:pos0]

    # Fold
    if fold:
        final_dist = abs(final_dist)

    peak = np.around([xpeak, ypeak]).astype(int)
    com = np.array([xcom, ycom])
    
    return final_dist, image_line_T, peak, (start, end), final_dist_com, com


def Get_Sub_Mask(point, regions_data, related_ids_T, connected_ids_dict, clump_coords_dict, start_coords):
    """
    Extract a sub-mask containing clumps connected to a specific point.
    
    This function identifies which clump a point belongs to and creates a
    mask of that clump and all connected clumps that are part of the filament.
    
    Parameters:
    -----------
    point : array-like
        Coordinates of the point to check
    regions_data : ndarray
        3D or 2D array with region IDs
    related_ids_T : list
        List of clump IDs that are part of the filament
    connected_ids_dict : dict
        Dictionary indicating which clumps are connected to each other
    clump_coords_dict : dict
        Dictionary mapping clump IDs to their constituent pixel coordinates
    start_coords : ndarray
        Offset coordinates for the local coordinate system
        
    Returns:
    --------
    fil_mask_sub_profiles : ndarray
        Binary mask of the connected clumps at the point
    """
    # For 2D regions data
    if len(regions_data.shape) == 2:
        # Get the clump ID at the given point (-1 to adjust for 1-based indexing)
        clump_id = regions_data[np.int64(np.around(point))[1], np.int64(np.around(point))[0]] - 1
        
        # Get all connected clump IDs
        connected_ids_i = [clump_id] + connected_ids_dict[clump_id]
        
        # Create an empty mask
        fil_mask_sub_profiles = np.zeros_like(regions_data)
        
        # Fill the mask with connected clumps that are part of the filament
        for connected_id in connected_ids_i:
            if connected_id in related_ids_T:
                coords = clump_coords_dict[connected_id]
                fil_mask_sub_profiles[(coords[:, 0] - start_coords[0], \
                                       coords[:, 1] - start_coords[1])] = 1
    
    # For 3D regions data
    elif len(regions_data.shape) == 3:
        # Get all clump IDs at the given point (may be multiple in 3D)
        clump_ids = regions_data[:, np.int64(np.around(point))[1], np.int64(np.around(point))[0]] - 1
        clump_ids = list(set(clump_ids))
        
        # Remove invalid ID (-1)
        if -1 in clump_ids:
            clump_ids.remove(-1)
        
        # Get all connected clump IDs
        connected_ids_i = []
        for clump_id in clump_ids:
            connected_ids_i += [clump_id] + connected_ids_dict[clump_id]
        
        # Create an empty mask (2D projection)
        fil_mask_sub_profiles = np.zeros_like(regions_data.sum(0))
        
        # Fill the mask with connected clumps that are part of the filament
        for connected_id in connected_ids_i:
            if connected_id in related_ids_T:
                if connected_id in related_ids_T:
                    coords = clump_coords_dict[connected_id]
                    fil_mask_sub_profiles[(coords[:, 1] - start_coords[1], coords[:, 2] - start_coords[2])] = 1
    
    return fil_mask_sub_profiles


def Cal_Dictionary_Cuts(SampInt, CalSub, regions_data, related_ids_T, connected_ids_dict, clump_coords_dict, \
                        skeleton_coords_2D, fil_image, fil_mask, dictionary_cuts, start_coords=None):
    """
    Calculate intensity profiles perpendicular to the filament spine.
    
    This function samples points along the filament spine and calculates
    intensity profiles perpendicular to the spine at each point.
    
    Parameters:
    -----------
    SampInt : int
        Sampling interval along the spine
    CalSub : bool
        Flag to calculate sub-structure masks
    regions_data : ndarray
        3D or 2D array with region IDs
    related_ids_T : list
        List of clump IDs that are part of the filament
    connected_ids_dict : dict
        Dictionary indicating which clumps are connected to each other
    clump_coords_dict : dict
        Dictionary mapping clump IDs to their constituent pixel coordinates
    skeleton_coords_2D : ndarray
        Coordinates of the filament spine
    fil_image : ndarray
        2D intensity image of the filament
    fil_mask : ndarray
        Binary mask of the filament
    dictionary_cuts : dict
        Dictionary to store profile information
    start_coords : ndarray, optional
        Offset coordinates for the local coordinate system
        
    Returns:
    --------
    dictionary_cuts : dict
        Updated dictionary with profile information
    """
    # Only proceed if there are enough skeleton points
    if len(skeleton_coords_2D) > 3:
        # Calculate B-spline representation of the skeleton
        points, fprime = Cal_B_Spline(SampInt, skeleton_coords_2D, fil_mask)
        
        # Make copies of the points and derivatives for updating
        points_updated = points.copy().tolist()
        fprime_updated = fprime.copy().tolist()
        
        # Process each point along the spline
        for point_id in range(len(points)):
            fil_image_shape = fil_image.shape
            
            # Check if point is within bounds with margin
            if np.round(points[point_id][0]) > 0 and np.round(points[point_id][1]) > 0 \
                    and np.round(points[point_id][0]) < fil_image_shape[1] - 1 and \
                    np.round(points[point_id][1]) < fil_image_shape[0] - 1:
                
                # Get sub-mask if calculating sub-structure
                if CalSub:
                    fil_mask = Get_Sub_Mask(points[point_id], regions_data, related_ids_T, connected_ids_dict, \
                                            clump_coords_dict, start_coords)
                
                # Calculate profile perpendicular to the spine at this point
                profile = Profile_Builder(fil_image, fil_mask, points[point_id], fprime[point_id], shift=True, fold=False)
                
                # Store profile information in the dictionary
                dictionary_cuts['distance'].append(profile[0])
                dictionary_cuts['profile'].append(profile[1])
                dictionary_cuts['plot_peaks'].append(profile[2])
                dictionary_cuts['plot_cuts'].append(profile[3])
                
                # Calculate and store mask width
                mask_width = Dists_Array([profile[3][0]], [profile[3][1]])[0][0]
                dictionary_cuts['mask_width'].append(np.around(mask_width, 3))
                
                # Store center of mass information
                dictionary_cuts['distance_com'].append(profile[4])
                dictionary_cuts['plot_coms'].append(profile[5])
            else:
                # Remove points that are out of bounds
                points_updated.remove(points[point_id].tolist())
                fprime_updated.remove(fprime[point_id].tolist())
        
        # Store the valid points and derivatives
        dictionary_cuts['points'].append(np.array(points_updated))
        dictionary_cuts['fprime'].append(np.array(fprime_updated))
    
    return dictionary_cuts


def Update_Dictionary_Cuts(dictionary_cuts, start_coords):
    """
    Update the coordinates in dictionary_cuts to the original coordinate system.
    
    This function transforms coordinates stored in dictionary_cuts from the
    local coordinate system back to the original global coordinate system.
    
    Parameters:
    -----------
    dictionary_cuts : dict
        Dictionary containing profile information in local coordinates
    start_coords : ndarray
        Offset coordinates to convert to global system
        
    Returns:
    --------
    dictionary_cuts : dict
        Updated dictionary with coordinates in global system
    """
    # Only update if there are points to update
    if len(dictionary_cuts['points']) > 0:
        # Update the spline points
        dictionary_cuts['points'][-1] = dictionary_cuts['points'][-1] + start_coords[1:][::-1]
        
        # Get number of points to update
        len_new = len(dictionary_cuts['points'][-1])
        
        # Update other coordinate fields
        for key in ['plot_peaks', 'plot_cuts', 'plot_coms']:
            dictionary_cuts[key][-len_new:] = \
                list(np.array(dictionary_cuts[key])[-len_new:] + start_coords[1:][::-1])
    
    return dictionary_cuts


def Get_Common_Skeleton(filament_clumps_id, related_ids_T, max_path_i, max_path_used, skeleton_coords_record, \
                        start_coords, clump_coords_dict):
    """
    Find a common skeleton segment between different branches of a filament.
    
    This function identifies a clump that is common to different filament paths
    and extracts a representative skeleton segment through that clump.
    
    Parameters:
    -----------
    filament_clumps_id : list
        IDs of clumps in the filament
    related_ids_T : list
        List of clump IDs that are part of the filament
    max_path_i : list
        Current path being analyzed
    max_path_used : list
        List of paths that have already been processed
    skeleton_coords_record : list
        Record of skeleton coordinates for each path
    start_coords : ndarray
        Offset coordinates for the local coordinate system
    clump_coords_dict : dict
        Dictionary mapping clump IDs to their constituent pixel coordinates
        
    Returns:
    --------
    common_clump_id : int or None
        ID of the clump that is common to different paths
    common_sc_item : ndarray or None
        Coordinates of the skeleton segment through the common clump
    """
    common_clump_id = None
    common_sc_item = None
    subpart_id_used = None
    
    # Only look for common parts if there's more than one path
    if len(max_path_used) != 1:
        common_path_id = -1
        break_logic = False
        
        # Try different subparts, starting from the most recent
        for subpart_id_used in np.int32(np.linspace(len(max_path_used) - 2, 0, len(max_path_used) - 1)):
            # Look for a node that appears in both the current path and a previous path
            for max_path_ii in max_path_i:
                if max_path_ii in max_path_used[subpart_id_used]:
                    common_path_id = max_path_ii
                    break_logic = True
                    break
            if break_logic:
                break
        
        # Get the clump ID for the common node
        common_clump_id = filament_clumps_id[common_path_id]
        
        # Get the skeleton coordinates for the subpart containing the common node
        skeleton_coords_i = skeleton_coords_record[subpart_id_used]
        
        # Find the intersection of skeleton coordinates and clump coordinates
        array1_tuples = set(map(tuple, skeleton_coords_i))
        array2_tuples = set(map(tuple, clump_coords_dict[common_clump_id][:, 1:]))
        common_elements_tuples = array1_tuples & array2_tuples
        common_skeleton_coords = np.array(list(common_elements_tuples))
        
        # Sort the common coordinates by their original order in the skeleton
        common_skeleton_coord_ids = []
        for common_skeleton_coord in common_skeleton_coords:
            index = np.where(np.all(skeleton_coords_i == common_skeleton_coord, axis=1))[0]
            common_skeleton_coord_ids.append(index[0])
        common_skeleton_coords = common_skeleton_coords[np.argsort(common_skeleton_coord_ids)]
        
        # Use the middle third of the common segment if it's long enough
        if len(common_skeleton_coords) > 3:
            third_len = len(common_skeleton_coords) // 3 + 1
            common_skeleton_coords = common_skeleton_coords[third_len:2 * third_len]
            common_sc_item = common_skeleton_coords - start_coords[1:]
    
    # Only use common parts if they connect to the ends of the filament
    if common_clump_id not in related_ids_T[:1] and common_clump_id not in related_ids_T[-1:]:
        common_sc_item = None
    
    return common_clump_id, common_sc_item


def Update_Max_Path_Record(max_path_record):
    """
    Sort and organize paths by length and connectivity.
    
    This function orders paths by length (longest first) and then ensures that
    paths are ordered by connectivity, with paths that share nodes appearing
    consecutively when possible.
    
    Parameters:
    -----------
    max_path_record : list
        List of paths (each path is a list of node indices)
        
    Returns:
    --------
    max_path_record_T : list
        Reordered list of paths
    """
    # Get the length of each path
    max_path_lens = []
    for max_path_i in max_path_record:
        max_path_lens.append(len(max_path_i))
    
    # Convert to numpy array for sorting
    max_path_record_array = np.array(max_path_record, dtype=object)
    
    # Sort paths by length (longest first)
    max_path_record = max_path_record_array[np.argsort(max_path_lens)[::-1]].tolist()
    
    # Make a copy for reordering
    max_path_record_copy = max_path_record.copy()
    
    # Start with the longest path
    max_path_T = max_path_record_copy[0].copy()
    max_path_record_T = [max_path_record_copy[0].copy()]
    max_path_record_copy.remove(max_path_record_copy[0])
    
    # Process remaining paths
    while len(max_path_record_copy) > 0:
        len_1 = len(max_path_record_copy)
        
        # Try to find a path that shares a node with already processed paths
        for max_path_i in max_path_record_copy:
            for max_path_ii in max_path_i:
                if max_path_ii in max_path_T:
                    # Add this path and its nodes to the processed list
                    max_path_record_T.append(max_path_i)
                    max_path_record_copy.remove(max_path_i)
                    max_path_T += max_path_i
                    break
        
        len_2 = len(max_path_record_copy)
        
        # If no paths could be connected, add all remaining paths and exit
        if len_1 == len_2:
            max_path_record_T += max_path_record_copy
            break
    
    return max_path_record_T


def Trim_Skeleton_Coords_2D(skeleton_coords_2D, SmallSkeleton=6):
    """
    Trim and clean up skeleton coordinates by removing redundancies and loops.
    
    This function removes overlapping or redundant points in the skeleton and
    ensures it forms a clean, non-branching path through the filament.
    
    Parameters:
    -----------
    skeleton_coords_2D : ndarray
        Input skeleton coordinates
    SmallSkeleton : int
        Small skeleton threshold
        
    Returns:
    --------
    coords : ndarray
        Trimmed skeleton coordinates
    small_sc : bool
        Whether skeleton is small
    """
    
    def is_connected(coords):
        """Robust connectivity check"""
        if len(coords) < 2:
            return True
        distances = cdist(coords, coords)
        adj = distances <= np.sqrt(2) + 1e-6
        np.fill_diagonal(adj, False)
        
        visited = set([0])
        queue = deque([0])
        while queue:
            current = queue.popleft()
            for i in np.where(adj[current])[0]:
                if i not in visited:
                    visited.add(i)
                    queue.append(i)
        return len(visited) == len(coords)
    
    def find_main_path_with_priorities(coords):
        """Find main skeleton path and assign priorities"""
        if len(coords) < 3:
            return list(range(len(coords))), np.ones(len(coords))
        
        distances = cdist(coords, coords)
        adj_matrix = distances <= 2 #np.sqrt(2) + 1e-6
        np.fill_diagonal(adj_matrix, False)
        
        # Find endpoints (points with few neighbors)
        neighbor_counts = np.sum(adj_matrix, axis=1)
        endpoints = np.where(neighbor_counts <= 2)[0]
        
        if len(endpoints) < 2:
            # Use furthest apart points as endpoints
            max_dist = 0
            start_idx, end_idx = 0, len(coords)-1
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    if distances[i, j] > max_dist:
                        max_dist = distances[i, j]
                        start_idx, end_idx = i, j
        else:
            start_idx, end_idx = endpoints[0], endpoints[-1]
        
        # Build path from start to end
        path = [start_idx]
        current = start_idx
        visited = {start_idx}
        
        while current != end_idx and len(path) < len(coords):
            neighbors = np.where(adj_matrix[current])[0]
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if not unvisited_neighbors:
                break
                
            # Choose neighbor closest to end point
            best_neighbor = min(unvisited_neighbors, 
                              key=lambda n: distances[n, end_idx])
            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor
        
        # Assign priority scores (main path gets higher scores)
        priorities = np.zeros(len(coords))
        for i, idx in enumerate(path):
            priorities[idx] = len(path) - i + 10  # Boost main path points
        
        # Give remaining points lower priority based on connectivity
        for i in range(len(coords)):
            if priorities[i] == 0:
                priorities[i] = neighbor_counts[i]
        
        return path, priorities
    
    def get_box_neighbors(center_coord, coords, box_size):
        """Get neighbors within box"""
        half = box_size // 2
        neighbors = []
        for i, coord in enumerate(coords):
            if (abs(coord[0] - center_coord[0]) <= half and 
                abs(coord[1] - center_coord[1]) <= half):
                neighbors.append(i)
        return neighbors
    
    def find_best_bridge_points(current_coords, original_coords):
        """Find optimal bridge points for reconnection"""
        if is_connected(current_coords):
            return []
        
        # Find disconnected components
        distances = cdist(current_coords, current_coords)
        adj = distances <= np.sqrt(2) + 1e-6
        np.fill_diagonal(adj, False)
        
        visited = set()
        components = []
        for i in range(len(current_coords)):
            if i not in visited:
                component = []
                stack = [i]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        neighbors = np.where(adj[current])[0]
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                stack.append(neighbor)
                components.append(component)
        
        if len(components) <= 1:
            return []
        
        # Find missing points that can bridge components
        current_set = set(map(tuple, current_coords))
        original_set = set(map(tuple, original_coords))
        missing_points = [np.array(p) for p in original_set - current_set]
        
        bridge_candidates = []
        for missing_point in missing_points:
            connected_components = set()
            
            # Check which components this point connects
            for comp_idx, component in enumerate(components):
                for point_idx in component:
                    dist = np.linalg.norm(current_coords[point_idx] - missing_point)
                    if dist <= np.sqrt(2) + 1e-6:
                        connected_components.add(comp_idx)
                        break
            
            if len(connected_components) >= 2:
                bridge_candidates.append((len(connected_components), missing_point))
        
        # Sort by number of components connected (prefer points that connect more)
        bridge_candidates.sort(reverse=True, key=lambda x: x[0])
        return [point for _, point in bridge_candidates]
    
    if len(skeleton_coords_2D) <= SmallSkeleton:
        return skeleton_coords_2D, True
    
    original_coords = skeleton_coords_2D.copy()
    coords = skeleton_coords_2D.copy()
    
    # Find main path and priorities
    main_path, priorities = find_main_path_with_priorities(coords)
    
    # Iterative trimming with priority-based removal
    max_iterations = 20
    max_3x3 = 3
    for iteration in range(max_iterations):
        coords_changed = False
        points_to_remove = set()
        
        # Update priorities after each iteration
        if iteration > 0:
            _, priorities = find_main_path_with_priorities(coords)
        
        # Check density constraints
        for i in range(len(coords)):
            # Check 3x3 neighborhood
            neighbors_3x3 = get_box_neighbors(coords[i], coords, 3)
            if len(neighbors_3x3) > max_3x3:
                excess = len(neighbors_3x3) - max_3x3
                # Sort by priority (remove lowest priority first)
                neighbor_priorities = [(priorities[j] if j < len(priorities) else 0, j) 
                                     for j in neighbors_3x3 if j != i]
                neighbor_priorities.sort()
                
                for _, remove_idx in neighbor_priorities[:excess]:
                    points_to_remove.add(remove_idx)
        
        # Apply removals while checking connectivity
        final_removals = []
        for remove_idx in sorted(points_to_remove, reverse=True):
            temp_coords = np.delete(coords, remove_idx, axis=0)
            if len(temp_coords) > SmallSkeleton and is_connected(temp_coords):
                final_removals.append(remove_idx)
        
        if final_removals:
            # Remove points and update priorities
            for remove_idx in final_removals:
                coords = np.delete(coords, remove_idx, axis=0)
                if remove_idx < len(priorities):
                    priorities = np.delete(priorities, remove_idx)
            coords_changed = True
        
        if not coords_changed:
            break
    
    # Repair connectivity if broken
    max_repair_attempts = len(coords)
    for repair_attempt in range(max_repair_attempts):
        if is_connected(coords):
            break
            
        bridge_points = find_best_bridge_points(coords, original_coords)
        if not bridge_points:
            # print("No bridge points found")
            break
        
        # Add bridge points until connected
        for bridge_point in bridge_points:  # Try up to 5 bridge points
            test_coords = np.vstack([coords, bridge_point.reshape(1, -1)])
            # if is_connected(test_coords):
            coords = test_coords
            break
        else:
            # If no single bridge works, try combinations
            if len(bridge_points) >= 1:
                test_coords = np.vstack([coords] + [bp.reshape(1, -1) for bp in bridge_points[:2]])
                # if is_connected(test_coords):
                coords = test_coords
                break
    
    # Final fallback
    # if not is_connected(coords):
    #     print("Warning: Could not restore full connectivity, using original skeleton")
    #     coords = original_coords

    # Create a graph from the skeleton coordinates
    G_sorted_skeleton, T_sorted_skeleton = Graph_Infor_Connected(coords)
    
    # Find the longest path through the skeleton
    max_path, max_edges = Get_Max_Path_Weight(T_sorted_skeleton)
    
    sorted_skeleton_coords = coords[max_path]
    small_sc = len(sorted_skeleton_coords) <= SmallSkeleton
    return sorted_skeleton_coords, small_sc


def Search_Max_Path_And_Edges(paths_and_weights):
    """
    Find the path with maximum weight from a list of paths and weights.
    
    Parameters:
    -----------
    paths_and_weights : list
        List of tuples (path, weight) where path is a list of node indices
        
    Returns:
    --------
    max_weight : float
        Weight of the maximum weight path
    max_path : list
        List of node indices forming the path with maximum weight
    max_edges : list
        List of edge pairs (node1, node2) in the maximum weight path
    """
    max_path = []
    max_edges = []
    max_weight = 0

    # Find the path with maximum weight
    if len(paths_and_weights) != 0:
        max_weight = max([weight for path, weight in paths_and_weights])
        max_path = [path for path, weight in paths_and_weights if weight == max_weight][0]
        max_edges = [(max_path[i], max_path[i + 1]) for i in range(len(max_path) - 1)]
    
    return max_weight, max_path, max_edges


def Fill_Mask_Holes(fil_mask, max_hole_size=4):
    """
    Fill small holes in a binary mask.
    
    This function identifies and fills small holes (background regions surrounded
    by foreground) in a binary mask, and also creates a contour mask.
    
    Parameters:
    -----------
    fil_mask : ndarray
        Binary mask with possible holes
    max_hole_size : int, optional
        Maximum size of holes to fill
        
    Returns:
    --------
    filtered_mask : ndarray
        Binary mask with small holes filled
    contour_data : ndarray
        Binary mask showing the contour (edge) of the mask
    """
    # Label all background regions
    labeled_image, num_features = ndimage.label(fil_mask == 0)
    
    # Count the size of each region
    sizes = np.bincount(labeled_image.ravel())
    
    # Create a copy of the mask for filling
    filtered_mask = fil_mask.copy()
    
    # Fill small holes
    for region_id, size in enumerate(sizes):
        if size > 0 and size <= max_hole_size:
            filtered_mask[labeled_image == region_id] = 1
    
    # Create erosion and dilation of the mask for contour detection
    fil_mask_erosion = morphology.binary_erosion(filtered_mask, morphology.disk(1))
    fil_mask_dilation = morphology.binary_dilation(filtered_mask, morphology.disk(1))
    
    # Contour is the difference between dilation and erosion
    contour_data = fil_mask_dilation * ~fil_mask_erosion
    
    return filtered_mask, contour_data


def Get_Max_Path_Intensity_Weighted(fil_mask, mask_coords, Tree, common_mask_coords_id=None):
    """
    Find the maximum intensity-weighted path through a mask.
    
    This function identifies the longest path through a mask, with preference for
    high-intensity regions. It can either find a path between endpoints or through
    specified common coordinates.
    
    Parameters:
    -----------
    fil_mask : ndarray
        Binary mask of the filament
    mask_coords : ndarray
        Coordinates of all points in the mask
    Tree : networkx.Graph
        Minimum spanning tree connecting mask points
    common_mask_coords_id : list, optional
        List of node IDs that should be included in the path
        
    Returns:
    --------
    max_path : list
        List of node indices forming the maximum intensity-weighted path
    max_edges : list
        List of edge pairs (node1, node2) in the path
    """
    # Fill holes and get contour
    fil_mask, contour_data = Fill_Mask_Holes(fil_mask)
    
    # Find nodes that are on the contour and have degree 1 (endpoints)
    degree1_nodes = [node for node in Tree.nodes if Tree.degree(node) == 1 and \
                     contour_data[mask_coords[node][0], mask_coords[node][1]]]
    
    paths_and_weights = []
    
    # If common coordinates are specified, find paths through them
    if type(common_mask_coords_id) != type(None) and len(common_mask_coords_id) > 0:
        for i in common_mask_coords_id:
            for j in range(len(degree1_nodes)):
                # Find path from common coordinate to endpoint
                path = nx.shortest_path(Tree, i, degree1_nodes[j])
                path_weight = 0
                
                # Calculate path weight (sum of inverse edge weights)
                for k in range(len(path) - 1):
                    if Tree[path[k]][path[k + 1]]['weight'] != 0:
                        path_weight += 1 / Tree[path[k]][path[k + 1]]['weight']
                
                paths_and_weights.append((path, path_weight))
    else:
        # Otherwise, find paths between all pairs of endpoints
        for i in range(len(degree1_nodes) - 1):
            for j in range(i + 1, len(degree1_nodes)):
                # Find path between endpoints
                path = nx.shortest_path(Tree, degree1_nodes[i], degree1_nodes[j])
                path_weight = 0
                
                # Calculate path weight (sum of inverse edge weights)
                for k in range(len(path) - 1):
                    if Tree[path[k]][path[k + 1]]['weight'] != 0:
                        path_weight += 1 / Tree[path[k]][path[k + 1]]['weight']
                
                paths_and_weights.append((path, path_weight))
    
    # Find the path with maximum weight
    max_weight, max_path, max_edges = Search_Max_Path_And_Edges(paths_and_weights)
    
    return max_path, max_edges


def Get_Max_Path_Intensity_Weighted_Fast(fil_mask, mask_coords, Tree, clump_numbers):
    """
    A faster version of Get_Max_Path_Intensity_Weighted for large filaments.
    
    This function uses heuristics to reduce the number of paths to check when
    finding the maximum intensity-weighted path through a large mask.
    
    Parameters:
    -----------
    fil_mask : ndarray
        Binary mask of the filament
    mask_coords : ndarray
        Coordinates of all points in the mask
    Tree : networkx.Graph
        Minimum spanning tree connecting mask points
    clump_numbers : int
        Number of clumps in the filament (used to determine approach)
        
    Returns:
    --------
    max_path_2 : list
        List of node indices forming the maximum intensity-weighted path
    max_edges_2 : list
        List of edge pairs (node1, node2) in the path
    """
    # Fill holes and get contour
    fil_mask, contour_data = Fill_Mask_Holes(fil_mask)
    
    # Find minimum weight edge as a starting point
    min_weight = float('inf')
    
    # Find points near the edges of the mask
    edge_coords_1 = np.where(mask_coords[:, 0] == mask_coords[:, 0].min())[0].tolist()
    edge_coords_2 = np.where(mask_coords[:, 0] == mask_coords[:, 0].max())[0].tolist()
    edge_coords_3 = np.where(mask_coords[:, 1] == mask_coords[:, 1].min())[0].tolist()
    edge_coords_4 = np.where(mask_coords[:, 1] == mask_coords[:, 1].max())[0].tolist()
    edge_coords = list(set(edge_coords_1 + edge_coords_2 + edge_coords_3 + edge_coords_4))
    
    # For smaller filaments, use a more thorough approach
    if clump_numbers < 1000:
        # Find contour points with degree 1 (endpoints)
        degree1_nodes = [node for node in Tree.nodes if Tree.degree(node) == 1 and \
                         contour_data[mask_coords[node][0], mask_coords[node][1]]]
        
        # Find minimum weight edge
        for (u, v, data) in Tree.edges(data=True):
            if data['weight'] < min_weight:
                min_weight = data['weight']
                min_weight_node = [u, v]
        
        # First phase: find paths from minimum weight edge to endpoints
        paths_and_weights = []
        for source_node_id in min_weight_node:
            for target_node_id in degree1_nodes:
                path = nx.shortest_path(Tree, source_node_id, target_node_id)
                path_weight = 0
                for i in range(len(path) - 1):
                    path_weight += 1 / Tree[path[i]][path[i + 1]]['weight']
                paths_and_weights.append((path, path_weight))
        
        # Find maximum weight path from first phase
        max_weight_1, max_path_1, max_edges_1 = Search_Max_Path_And_Edges(paths_and_weights)
        
        # Second phase: find paths from endpoints to the end of the first path
        paths_and_weights = []
        for target_node_id in degree1_nodes:
            # Find path from end of first path to endpoint
            path = nx.shortest_path(Tree, max_path_1[-1], target_node_id)
            
            # Check if path goes through minimum weight nodes
            in_path_logic = min_weight_node[0] in path
            for min_weight_node_id in range(1, len(min_weight_node)):
                in_path_logic = in_path_logic and (min_weight_node[min_weight_node_id] in path)
            
            # Add path if it goes through minimum weight nodes
            if in_path_logic:
                path_weight = 0
                for i in range(len(path) - 1):
                    path_weight += 1 / Tree[path[i]][path[i + 1]]['weight']
                paths_and_weights.append((path, path_weight))
            
            # Also check paths from endpoint to edge points
            for edge_coord in edge_coords:
                path = nx.shortest_path(Tree, target_node_id, edge_coord)
                path_weight = 0
                for i in range(len(path) - 1):
                    path_weight += 1 / Tree[path[i]][path[i + 1]]['weight']
                paths_and_weights.append((path, path_weight))
    else:
        # For larger filaments, only check paths between edge points
        paths_and_weights = []
        for i in range(len(edge_coords) - 1):
            for j in range(i + 1, len(edge_coords)):
                path = nx.shortest_path(Tree, edge_coords[i], edge_coords[j])
                path_weight = 0
                for k in range(len(path) - 1):
                    path_weight += 1 / Tree[path[k]][path[k + 1]]['weight']
                    paths_and_weights.append((path, path_weight))
    
    # Find the path with maximum weight
    max_weight_2, max_path_2, max_edges_2 = Search_Max_Path_And_Edges(paths_and_weights)
    
    return max_path_2, max_edges_2


def Get_Single_Filament_Skeleton_Weighted(fil_image, fil_mask, clump_numbers, common_sc_item=None, SmallSkeleton=6):
    """
    Extract an intensity-weighted skeleton from a filament mask.
    
    This function creates a graph from mask coordinates with weights based on
    intensity, then finds the longest path through this graph to represent
    the filament's spine.
    
    Parameters:
    -----------
    fil_image : ndarray
        2D intensity image of the filament
    fil_mask : ndarray
        Binary mask of the filament
    clump_numbers : int
        Number of clumps in the filament
    common_sc_item : ndarray, optional
        Common skeleton coordinates to include in the path
    SmallSkeleton : int, optional
        Threshold to define a "small" skeleton
        
    Returns:
    --------
    skeleton_coords_2D : ndarray
        Coordinates of the intensity-weighted skeleton
    small_sc : bool
        Flag indicating if the skeleton is considered "small"
    """
    # Flag to indicate if we're using common skeleton coordinates
    CalSubSK = type(common_sc_item) != type(None)
    
    # Apply a smoothing filter to reduce noise
    fil_image_filtered = ndimage.uniform_filter(fil_image, size=3)
    
    # Get coordinates of all pixels in the mask
    regions_list = measure.regionprops(np.array(fil_mask, dtype='int'))
    mask_coords = regions_list[0].coords
    
    # Calculate distances between neighboring coordinates
    dist_matrix = Dists_Array(mask_coords, mask_coords)
    mask_coords_in_dm = np.where(np.logical_and(dist_matrix > 0.5, dist_matrix < 2))
    
    # Create a graph with edges between neighboring pixels
    Graph_find_skeleton = nx.Graph()
    common_mask_coords_id = []
    
    # Add edges with weights inversely proportional to intensity
    for i, j in zip(mask_coords_in_dm[0], mask_coords_in_dm[1]):
        weight_ij = fil_image_filtered[mask_coords[i][0], mask_coords[i][1]] + \
                    fil_image_filtered[mask_coords[j][0], mask_coords[j][1]]
        
        # Set weight to 0 if intensity is 0, otherwise make it inversely proportional
        if weight_ij != 0:
            weight_ij = dist_matrix[i, j] / weight_ij
        
        # Add edge to graph
        Graph_find_skeleton.add_edge(i, j, weight=weight_ij)
        
        # If using common skeleton coordinates, record their IDs
        if CalSubSK:
            if tuple(mask_coords[i]) in map(tuple, common_sc_item):
                common_mask_coords_id.append(i)
    
    # Create unique list of common coordinate IDs
    if CalSubSK:
        common_mask_coords_id = list(set(common_mask_coords_id))
    else:
        common_mask_coords_id = None
    
    # Find minimum spanning tree
    Tree = nx.minimum_spanning_tree(Graph_find_skeleton)
    
    # Find the longest path through the tree
    if clump_numbers < 100 or CalSubSK:
        max_path, max_edges = Get_Max_Path_Intensity_Weighted(fil_mask, mask_coords, Tree, common_mask_coords_id)
    else:
        max_path, max_edges = Get_Max_Path_Intensity_Weighted_Fast(fil_mask, mask_coords, Tree, clump_numbers)
    
    # Extract coordinates for the maximum path
    skeleton_coords_2D = mask_coords[max_path]
    
    # Trim and refine the skeleton
    skeleton_coords_2D, small_sc = Trim_Skeleton_Coords_2D(skeleton_coords_2D, SmallSkeleton)
    
    return skeleton_coords_2D, small_sc


def Cal_Lengh_Width_Ratio(CalSub, regions_data, related_ids_T, connected_ids_dict, clump_coords_dict, \
                          filament_mask_2D, filament_data, clumps_number, SkeletonType='Intensity'):
    """
    Calculate the length-to-width ratio of a filament.
    
    This function extracts the skeleton of a filament, measures profiles perpendicular
    to the skeleton, and calculates the ratio of the filament's length to its width.
    
    Parameters:
    -----------
    CalSub : bool
        Flag to calculate sub-structure information
    regions_data : ndarray
        3D or 2D array with region IDs
    related_ids_T : list
        List of clump IDs that are part of the filament
    connected_ids_dict : dict
        Dictionary indicating which clumps are connected to each other
    clump_coords_dict : dict
        Dictionary mapping clump IDs to their constituent pixel coordinates
    filament_mask_2D : ndarray
        2D binary mask of the filament
    filament_data : ndarray
        3D intensity data of the filament
    clumps_number : int
        Number of clumps in the filament
    SkeletonType : str, optional
        Method to use for skeletonization ('Intensity' or 'Morphology')
        
    Returns:
    --------
    dictionary_cuts : dict
        Dictionary with profile information
    lengh_dist : float
        Length of the filament
    lengh_width_ratio : float
        Ratio of filament length to width
    skeleton_coords_2D : ndarray
        Coordinates of the filament skeleton
    all_skeleton_coords : ndarray
        Coordinates of all skeleton pixels including branches
    """
    # Sampling interval for points along the skeleton
    samp_int = 1
    
    # Initialize dictionary to store profile information
    dictionary_cuts = defaultdict(list)
    
    # Get 2D projection of intensity data
    fil_image = filament_data.sum(0)
    
    # Convert mask to boolean
    fil_mask = filament_mask_2D.astype(bool)
    
    # Find all connected regions in the mask
    regions = measure.regionprops(measure.label(fil_mask, connectivity=2))
    
    # If there are multiple regions, keep only the largest one
    if len(regions) > 1:
        max_area = regions[0].area
        max_region = regions[0]
        
        # Clear all regions from the image and mask
        for region in regions:
            coords = region.coords
            fil_image[coords[:, 0], coords[:, 1]] = 0
            fil_mask[coords[:, 0], coords[:, 1]] = False
            
            # Keep track of the largest region
            if region.area > max_area:
                max_area = region.area
                max_region = region
        
        # Restore the largest region to the image and mask
        fil_image[max_region.coords[:, 0], max_region.coords[:, 1]] = \
            filament_data.sum(0)[max_region.coords[:, 0], max_region.coords[:, 1]]
        fil_mask[max_region.coords[:, 0], max_region.coords[:, 1]] = True
    
    # Extract the skeleton using the specified method
    if SkeletonType == 'Morphology':
        # Use morphological operations to find skeleton
        skeleton_coords_2D, filament_skeleton, all_skeleton_coords = Get_Single_Filament_Skeleton(fil_mask)
    elif SkeletonType == 'Intensity':
        # Use intensity-weighted method to find skeleton
        all_skeleton_coords = None
        skeleton_coords_2D, small_sc = Get_Single_Filament_Skeleton_Weighted(fil_image, fil_mask, clumps_number)
    else:
        print('Please choose the skeleton_type between Morphology and Intensity')

    # Only proceed with length/width calculations if skeleton is not too small
    if not small_sc:
        # Calculate intensity profiles perpendicular to the skeleton
        dictionary_cuts = Cal_Dictionary_Cuts(samp_int, CalSub, regions_data, related_ids_T, connected_ids_dict,
                                              clump_coords_dict, \
                                              skeleton_coords_2D, fil_image, fil_mask, dictionary_cuts)
        
        # Extract endpoints of profiles
        start_coords = np.array(dictionary_cuts['plot_cuts'])[:, 0]
        end_coords = np.array(dictionary_cuts['plot_cuts'])[:, 1]
        
        # Calculate widths at each point
        width_dists = np.diagonal(Dists_Array(start_coords, end_coords))
        
        # Use median width excluding endpoints (more robust than mean)
        width_dist_mean = np.median(width_dists[1:-1])
        
        # Length is number of sample points along the skeleton
        lengh_dist = len(dictionary_cuts['points'][0]) * samp_int
        
        # Calculate length-to-width ratio
        lengh_width_ratio = lengh_dist / width_dist_mean
    else:
        # For small skeletons, set default values
        lengh_dist = 1
        lengh_width_ratio = 1
    
    return dictionary_cuts, lengh_dist, lengh_width_ratio, skeleton_coords_2D, all_skeleton_coords


def Get_LBV_Table(coords):
    """
    Calculate spatial and velocity statistics for a set of coordinates.
    
    This function computes the bounding box, projected area, and velocity range
    of a set of (l,b,v) coordinates representing a filament or clump.
    
    Parameters:
    -----------
    coords : list or ndarray
        List of coordinates in the form [v, b, l] where v is velocity, 
        b is galactic latitude, and l is galactic longitude
        
    Returns:
    --------
    coords_range : list
        Minimum and maximum values in each dimension [v_min, v_max, b_min, b_max, l_min, l_max]
    lb_area : int
        Area of the structure in the longitude-latitude (spatial) plane
    v_delta : int
        Range of velocities covered by the structure
    box_data : ndarray
        2D binary mask of the structure in the l-b plane
    """
    # Find the range in each dimension
    x_min = np.array(coords[0]).min()
    x_max = np.array(coords[0]).max()
    y_min = np.array(coords[1]).min()
    y_max = np.array(coords[1]).max()
    z_min = np.array(coords[2]).min()
    z_max = np.array(coords[2]).max()
    
    # Calculate velocity range (along x/v dimension)
    v_delta = x_max - x_min + 1
    
    # Create a 2D binary mask of the structure's projection in the l-b plane
    # Add padding of 1 pixel on each side (hence the +3)
    box_data = np.zeros([y_max - y_min + 3, z_max - z_min + 3])
    box_data[coords[1] - y_min + 1, coords[2] - z_min + 1] = 1
    
    # Label connected regions
    box_label = measure.label(box_data)
    
    # Calculate properties of the regions
    box_region = measure.regionprops(box_label)
    
    # Get the area in the l-b plane
    lb_area = box_region[0].area
    
    # Compile coordinate ranges
    coords_range = [x_min, x_max, y_min, y_max, z_min, z_max]
    
    return coords_range, lb_area, v_delta, box_data


def Cal_Velocity_Map(filament_item, skeleton_coords_2D, data_wcs_item):
    """
    Calculate a velocity map for a filament and measure velocity variation along its spine.
    
    This function computes the intensity-weighted mean velocity at each position in the
    filament's projection, and measures how velocity varies along the filament's spine.
    
    Parameters:
    -----------
    filament_item : ndarray
        3D intensity data cube of the filament
    skeleton_coords_2D : ndarray
        Coordinates of the filament spine in 2D projection
    data_wcs_item : astropy.wcs.WCS
        World Coordinate System object for coordinate transformations
        
    Returns:
    --------
    lbv_item_start : list
        Starting coordinates in the world coordinate system [l, b, v]
    lbv_item_end : list
        Ending coordinates in the world coordinate system [l, b, v]
    velocity_map_item : ndarray
        2D map of intensity-weighted mean velocity at each position
    v_skeleton_com_delta : float
        Range of velocities along the filament spine
    """
    # Get the shape of the data cube
    filament_item_shape = filament_item.shape
    
    # Define the bounds in pixel coordinates
    l_min, l_max = 0, filament_item_shape[2] - 1  # Longitude range
    b_min, b_max = 0, filament_item_shape[1] - 1  # Latitude range
    v_min, v_max = 0, filament_item_shape[0] - 1  # Velocity range
    
    # Convert pixel coordinates to world coordinates
    if data_wcs_item.naxis == 4:
        # For 4D WCS (includes a time or stokes dimension)
        lbv_start = data_wcs_item.all_pix2world(l_min, b_min, v_min, 0, 0)
        lbv_end = data_wcs_item.all_pix2world(l_max, b_max, v_max, 0, 0)
    elif data_wcs_item.naxis == 3:
        # For 3D WCS (just l, b, v)
        lbv_start = data_wcs_item.all_pix2world(l_min, b_min, v_min, 0)
        lbv_end = data_wcs_item.all_pix2world(l_max, b_max, v_max, 0)
    
    # Format coordinates for readability (convert v from m/s to km/s)
    lbv_item_start = [np.around(lbv_start[0], 2), np.around(lbv_start[1], 2), np.around(lbv_start[2] / 1000, 2)]
    lbv_item_end = [np.around(lbv_end[0], 2), np.around(lbv_end[1], 2), np.around(lbv_end[2] / 1000, 2)]
    
    # Create a linear scale for the velocity dimension
    velocity_range = np.linspace(lbv_item_start[2], lbv_item_end[2], filament_item_shape[0])
    
    # Calculate intensity-weighted mean velocity at each l,b position
    # This is equivalent to the first moment of the velocity dimension
    velocity_map_item = np.tensordot(filament_item, velocity_range, axes=((0,), (0,))) / np.sum(filament_item, axis=0)
    
    # Replace NaN values (from division by zero) with zeros
    velocity_map_item = np.nan_to_num(velocity_map_item)
    
    # Extract velocities along the filament spine
    v_skeleton_com_i = velocity_map_item[(skeleton_coords_2D[:, 0], skeleton_coords_2D[:, 1])]
    
    # Calculate the velocity range along the spine
    v_skeleton_com_delta = np.around(v_skeleton_com_i.max() - v_skeleton_com_i.min(), 3)
    
    return lbv_item_start, lbv_item_end, velocity_map_item, v_skeleton_com_delta



