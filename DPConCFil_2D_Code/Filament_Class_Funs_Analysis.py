import time
import numpy as np
from skimage import measure,morphology
import scipy.ndimage as ndimage
from collections import defaultdict,deque
from scipy.interpolate import splprep, splev,RegularGridInterpolator
import networkx as nx
from scipy.spatial.distance import cdist

def Filament_Coords(origin_data, regions_data, data_wcs, clump_coords_dict, related_ids_T, CalSub=False):
    filament_coords = clump_coords_dict[related_ids_T[0]]
    for clump_id in related_ids_T[1:]:
        filament_coords = np.r_[filament_coords, clump_coords_dict[clump_id]]
    x_min = np.array(filament_coords[:, 0]).min()
    x_max = np.array(filament_coords[:, 0]).max()
    y_min = np.array(filament_coords[:, 1]).min()
    y_max = np.array(filament_coords[:, 1]).max()
    length = np.max([x_max - x_min, y_max - y_min])
    if np.int32(length*0.1) % 2 == 0:
        length += np.int32(length*0.1) + 5
    else:
        length += np.int32(length*0.1) + 6
    filament_item = np.zeros([length, length])
    regions_data_T = np.zeros([length, length], dtype=np.int32)
    start_x = np.int64((length - (x_max - x_min)) / 2)
    start_y = np.int64((length - (y_max - y_min)) / 2)
    filament_item[filament_coords[:, 0] - x_min + start_x, filament_coords[:, 1] - y_min + start_y] = \
                    origin_data[filament_coords[:, 0], filament_coords[:, 1]]
    regions_data_T[filament_coords[:, 0] - x_min + start_x, filament_coords[:, 1] - y_min + start_y] = \
                    regions_data[filament_coords[:, 0], filament_coords[:, 1]]
    start_coords = np.array([x_min - start_x, y_min - start_y])

    filament_item_mask_2D = np.zeros_like(filament_item, dtype=np.int32)
    filament_item_mask_2D[filament_coords[:, 0] - x_min + start_x,filament_coords[:, 1] - y_min + start_y] = 1
    if CalSub == False:
        box_region = measure.regionprops(filament_item_mask_2D)
        lb_area = box_region[0].area
        data_wcs_item = data_wcs.deepcopy()
        data_wcs_item.wcs.crpix[0] -= start_coords[0]
        data_wcs_item.wcs.crpix[1] -= start_coords[1]
    else:
        lb_area = None
        data_wcs_item = None
    
    return filament_coords, filament_item, data_wcs_item, regions_data_T, start_coords, filament_item_mask_2D, lb_area

def Get_DV(box_data,box_center):
    #2D
    box_region = np.where(box_data!= 0)
    A11 = np.sum((box_region[0]-box_center[0])**2*\
        box_data[box_region])
    A12 = -np.sum((box_region[0]-box_center[0])*\
        (box_region[1]-box_center[1])*\
        box_data[box_region])
    A21 = A12
    A22 = np.sum((box_region[1]-box_center[1])**2*\
        box_data[box_region])
    A = np.array([[A11,A12],[A21,A22]])/len(box_region[0])
    D, V = np.linalg.eig(A)
    if D[0] < D[1]:
        D = D[[1,0]]
        V = V[[1,0]]

    if V[1][0]<0 and V[0][0]>0 and V[1][1]>0:
        V = -V
    size_ratio = np.sqrt(D[0]/D[1])
    angle = np.around(np.arccos(V[0][0])*180/np.pi-90,2)
    return D,V,size_ratio,angle


def Graph_Infor_SubStructure(origin_data,filament_mask_2D,filament_centers_LBV,filament_clumps_id,connected_ids_dict):
    points_LB = np.c_[filament_centers_LBV[:,0],filament_centers_LBV[:,1]]
    n_points = len(filament_centers_LBV)
    dist_matrix = Dists_Array(points_LB, points_LB)
    filament_centers_LBV = np.int64(np.around(filament_centers_LBV))
    
    Graph = nx.Graph()
    for i in range(n_points):
        neighboring_ids_T = connected_ids_dict[filament_clumps_id[i]]
        neighboring_ids = neighboring_ids_T.copy()
#         for neighboring_id in neighboring_ids_T:
#             neighboring_ids += list(connected_ids_dict[neighboring_id])
        neighboring_ids = list(set(neighboring_ids))
        for j in range(i + 1, n_points):
            if filament_clumps_id[j] in neighboring_ids:
                line_coords = Get_Line_Coords_2D(points_LB[i],points_LB[j])
                mask_2D_ids = filament_mask_2D[line_coords[:,1],line_coords[:,0]] # mask_2D_ids = [0,1]
                weight_ij = origin_data[line_coords[:,1],line_coords[:,0]].mean()
#                 if 0 not in mask_2D_ids:  #This is just for 2D.
                weight = dist_matrix[i, j]/(weight_ij)
                Graph.add_edge(i, j, weight=weight)
    Tree = nx.minimum_spanning_tree(Graph)
    return Graph,Tree


def Get_Max_Path_Weight_SubStructure(origin_data,filament_centers_LBV,T,Tree_0,sub_tree):
    #Weight
    points_LB = np.c_[filament_centers_LBV[:,0],filament_centers_LBV[:,1]]
    dist_matrix = Dists_Array(points_LB, points_LB)
    filament_centers_LBV = np.int64(np.around(filament_centers_LBV))

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
    for i in range(len(degree1_nodes)):
        for j in range(len(max_degree_nodes)):
            if nx.has_path(T, degree1_nodes[i], max_degree_nodes[j]):
                path = nx.shortest_path(T, degree1_nodes[i], max_degree_nodes[j]) 
                path_weight = 0
                for k in range(len(path)-1):
                    line_coords = Get_Line_Coords_2D(filament_centers_LBV[path[k]],filament_centers_LBV[path[k+1]])
                    weight_ij = origin_data[line_coords[:,1],line_coords[:,0]].mean()
                    path_weight += dist_matrix[path[k],path[k+1]]*weight_ij
                paths_and_weights.append((path, path_weight))
    if len(paths_and_weights) != 0:
        max_weight = max([weight for path, weight in paths_and_weights])
        max_path = [path for path, weight in paths_and_weights if weight == max_weight][0]
        max_edges = [(max_path[i], max_path[i+1]) for i in range(len(max_path)-1)]
    return max_path,max_edges



def Get_Common_Skeleton(filament_clumps_id,related_ids_T,max_path_i,max_path_used,skeleton_coords_record,\
                        start_coords,clump_coords_dict):
    common_clump_id = None
    common_sc_item = None
    subpart_id_used = None
    if len(max_path_used) != 1:
        common_path_id = -1
        break_logic = False
        for subpart_id_used in np.int32(np.linspace(len(max_path_used)-2,0,len(max_path_used)-1)):
            for max_path_ii in max_path_i:
                if max_path_ii in max_path_used[subpart_id_used]:
                    common_path_id = max_path_ii
                    break_logic = True
                    break
            if break_logic:
                break
        common_clump_id = filament_clumps_id[common_path_id]
        skeleton_coords_i = skeleton_coords_record[subpart_id_used]
        array1_tuples = set(map(tuple, skeleton_coords_i))
        array2_tuples = set(map(tuple, clump_coords_dict[common_clump_id]))
        common_elements_tuples = array1_tuples & array2_tuples
        common_skeleton_coords = np.array(list(common_elements_tuples))
        common_skeleton_coord_ids = []
        for common_skeleton_coord in common_skeleton_coords:
            index = np.where(np.all(skeleton_coords_i == common_skeleton_coord, axis=1))[0]
            common_skeleton_coord_ids.append(index[0])
        common_skeleton_coords = common_skeleton_coords[np.argsort(common_skeleton_coord_ids)]
        if len(common_skeleton_coords)>3:
            third_len = len(common_skeleton_coords) // 3 + 1
            common_skeleton_coords = common_skeleton_coords[third_len:2*third_len]
            common_sc_item = common_skeleton_coords - start_coords
    if common_clump_id not in related_ids_T[:1] and common_clump_id not in related_ids_T[-1:]:
        common_sc_item = None
    return common_clump_id,common_sc_item
    

def Get_Common_Skeleton(filament_clumps_id, related_ids_T, max_path_i, max_path_used, skeleton_coords_record, \
                        start_coords, clump_coords_dict, centers):
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
    common_path_id = -1
    
    # Only look for common parts if there's more than one path
    if len(max_path_used) != 1:
        common_path_id = -1
        break_logic = False
        # Try different subparts, starting from the most recent
        # for subpart_id_used in np.int32(np.linspace(len(max_path_used) - 2, 0, len(max_path_used) - 1)):
        for subpart_id_used in np.int32(np.linspace(0,len(max_path_used) - 2, len(max_path_used) - 1)):
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
        array2_tuples = set(map(tuple, clump_coords_dict[common_clump_id]))
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
            common_sc_item = common_skeleton_coords - start_coords
        if len(common_skeleton_coords) == 0:
            common_sc_item = np.int32(np.around([centers[common_clump_id] - start_coords]))
    
    # Only use common parts if they connect to the ends of the filament
    # if common_clump_id not in related_ids_T[:1] and common_clump_id not in related_ids_T[-1:]:
    #     common_sc_item = None

    sub_centers_item = []
    for max_path_ii in max_path_i:
        if max_path_ii != common_path_id:
            sub_centers_item.append(centers[filament_clumps_id[max_path_ii]] - start_coords)
    sub_centers_item = np.int32(np.around(sub_centers_item))
    
    return common_clump_id, common_sc_item, sub_centers_item




def Cal_Lengh_Width_Ratio(CalSub,regions_data,related_ids_T,connected_ids_dict,clump_coords_dict,\
                          filament_mask_2D,filament_data,clumps_number,SkeletonType='Intensity'):
    samp_int = 1
    dictionary_cuts = defaultdict(list)
    fil_image = filament_data
    fil_mask = filament_mask_2D.astype(bool)
    regions = measure.regionprops(measure.label(fil_mask,connectivity=2))
    if len(regions)>1:
        max_area = regions[0].area
        max_region = regions[0]
        for region in regions:
            coords = region.coords
            fil_image[coords[:,0],coords[:,1]] = 0
            fil_mask[coords[:,0],coords[:,1]] = False
            if region.area > max_area:
                max_area = region.area
                max_region = region
        fil_image[max_region.coords[:,0],max_region.coords[:,1]] = \
                            filament_data.sum(0)[max_region.coords[:,0],max_region.coords[:,1]]
        fil_mask[max_region.coords[:,0],max_region.coords[:,1]] = True
    if SkeletonType == 'Morphology':
        skeleton_coords_2D,filament_skeleton,all_skeleton_coords = Get_Single_Filament_Skeleton(fil_mask)
    elif SkeletonType == 'Intensity':
        all_skeleton_coords = None
        skeleton_coords_2D,small_sc = Get_Single_Filament_Skeleton_Weighted(fil_image,fil_mask,clumps_number)
    else:
        print('Please choose the skeleton_type between Morphology and Intensity')
        
    if not small_sc:
        dictionary_cuts = Cal_Dictionary_Cuts(samp_int,CalSub,regions_data,related_ids_T,connected_ids_dict,clump_coords_dict,\
                                        skeleton_coords_2D,fil_image,fil_mask,dictionary_cuts)
        start_coords = np.array(dictionary_cuts['plot_cuts'])[:,0]
        end_coords = np.array(dictionary_cuts['plot_cuts'])[:,1]
        width_dists = np.diagonal(Dists_Array(start_coords,end_coords))
        width_dist_mean = np.median(width_dists[1:-1])
        lengh_dist = len(dictionary_cuts['points'][0])*samp_int
        lengh_width_ratio = lengh_dist/width_dist_mean
    else:
        lengh_dist = 1
        lengh_width_ratio = 1
    return dictionary_cuts,lengh_dist,lengh_width_ratio,skeleton_coords_2D,all_skeleton_coords


def Update_Dictionary_Cuts(dictionary_cuts,start_coords):
    if len(dictionary_cuts['points']) > 0:
        dictionary_cuts['points'][-1] = dictionary_cuts['points'][-1]+start_coords[::-1]
        len_new = len(dictionary_cuts['points'][-1])
        for key in ['plot_peaks', 'plot_cuts', 'plot_coms']:
            dictionary_cuts[key][-len_new:] = \
                list(np.array(dictionary_cuts[key])[-len_new:] + start_coords[::-1])
    return dictionary_cuts


def Get_LBV_Table(coords):
    x_min = np.array(coords[0]).min()
    x_max = np.array(coords[0]).max()
    y_min = np.array(coords[1]).min()
    y_max = np.array(coords[1]).max()
    v_delta = 0
    box_data = np.zeros([y_max-y_min+3])
    box_data[coords[0]-x_min+1,coords[1]-y_min+1] = 1
    box_label = measure.label(box_data)
    box_region = measure.regionprops(box_label)
    lb_area = box_region[0].area
    coords_range = [x_min,x_max,y_min,y_max]
    return coords_range,lb_area,v_delta,box_data


def Cal_Tree_By_Clump_IDs(filamentObj, clumps_id):
    centers_LB = []
    max_path_record = []
    max_edges_record = []
    
    data_wcs = filamentObj.clumpsObj.data_wcs
    origin_data = filamentObj.clumpsObj.origin_data
    regions_data = filamentObj.clumpsObj.regions_data
    centers = filamentObj.clumpsObj.centers
    connected_ids_dict = filamentObj.clumpsObj.connected_ids_dict
    clump_coords_dict = filamentObj.clumpsObj.clump_coords_dict
    
    filament_coords, filament_item, data_wcs_item, regions_data_T, start_coords, filament_item_mask_2D, lb_area = \
        Filament_Coords(origin_data, regions_data, data_wcs, clump_coords_dict, clumps_id)
    
    for index in clumps_id:
        centers_LB.append((centers[index]-start_coords)[::-1])
    centers_item_LB = np.array(centers_LB)
    
    Graph,Tree = Graph_Infor_SubStructure(filament_item,filament_item_mask_2D,centers_item_LB,\
                                               clumps_id,connected_ids_dict)
    max_path_record, max_edges_record = Get_Max_Path_Recursion(filament_item, centers_item_LB, \
                                                                    max_path_record, max_edges_record, Graph, Tree, Tree)
    max_path_record = Update_Max_Path_Record(max_path_record)
    return filament_item, Tree, centers_item_LB, max_path_record
    




# Bellow funs are same as 3D.


def Get_Line_Coords_2D(point_a,point_b):
    coords = []
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    sort_index = np.argsort(np.abs(point_b - point_a))
    point_a_temp = point_a[sort_index]
    point_b_temp = point_b[sort_index]

    x1, y1 = point_a_temp[0], point_a_temp[1]
    x2, y2 = point_b_temp[0], point_b_temp[1]
    kx, ky = (x2 - x1), (y2 - y1)
    k_norm = np.sqrt(kx**2 + ky**2)   
    for y in range(min(int(round(y1)),int(round(y2))), max(int(round(y1)),int(round(y2)))+1):
        x = x1 + kx * (y - y1) / ky
        coords.append((int(round(x)), int(round(y))))
    index_0 = np.where(sort_index==0)[0]
    index_1 = np.where(sort_index==1)[0]
    line_coords = np.c_[np.array(coords)[:,index_0],np.array(coords)[:,index_1]].astype('int')  
    return line_coords

def Get_Line_Coords_3D(point_a,point_b):
    coords = []
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    sort_index = np.argsort(np.abs(point_b - point_a))
    point_a_temp = point_a[sort_index]
    point_b_temp = point_b[sort_index]

    x1, y1, z1 = point_a_temp[0], point_a_temp[1], point_a_temp[2]
    x2, y2, z2 = point_b_temp[0], point_b_temp[1], point_b_temp[2]
    kx, ky, kz = (x2 - x1), (y2 - y1), (z2 - z1)
    k_norm = np.sqrt(kx**2 + ky**2 + kz**2)   
    for z in range(min(int(round(z1)),int(round(z2))), max(int(round(z1)),int(round(z2)))+1):
        x = x1 + kx * (z - z1) / kz
        y = y1 + ky * (z - z1) / kz
        coords.append((int(round(x)), int(round(y)), int(round(z))))
    index_0 = np.where(sort_index==0)[0]
    index_1 = np.where(sort_index==1)[0]
    index_2 = np.where(sort_index==2)[0]
    coords_xy = np.c_[np.array(coords)[:,index_0],np.array(coords)[:,index_1]]        
    line_coords = np.c_[coords_xy,np.array(coords)[:,index_2]].astype('int')
    return line_coords


def Dists_Array(matrix_1, matrix_2):
    matrix_1 = np.array(matrix_1)
    matrix_2 = np.array(matrix_2)
    num_1 = matrix_1.shape[0]
    num_2 = matrix_2.shape[0]
    dist_1 = -2 * np.dot(matrix_1, matrix_2.T)
    dist_2 = np.sum(np.square(matrix_1), axis=1, keepdims=True) 
    dist_3 = np.sum(np.square(matrix_2), axis=1)   
    random_dist = np.random.random(dist_1.shape)/1000000
    dist_temp = dist_1 + dist_2 + dist_3 + random_dist
    dists = np.sqrt(dist_temp) 
    return dists

def Graph_Infor(points):
#     points = np.array(sorted(points, key=lambda x: x[0], reverse=False))
    n_points = len(points)
    dist_matrix = Dists_Array(points, points)
    Graph = nx.Graph()
    for i in range(n_points):
        for j in range(i + 1, n_points):
            Graph.add_edge(i, j, weight=dist_matrix[i, j])
    Tree = nx.minimum_spanning_tree(Graph)
    return Graph,Tree

def Graph_Infor_Connected(points):
#     points = np.array(sorted(points, key=lambda x: x[0], reverse=False))
    dist_matrix = Dists_Array(points, points)
    mask_coords_in_dm = np.where(np.logical_and(dist_matrix>0.5,dist_matrix<2))
    Graph = nx.Graph()
    for i,j in zip(mask_coords_in_dm[0],mask_coords_in_dm[1]):
        weight_ij = 1#dist_matrix[i,j]
        Graph.add_edge(i, j, weight=weight_ij)
    Tree = nx.minimum_spanning_tree(Graph)
    return Graph,Tree

def Get_Max_Path_Node(T):
    #Node
    degree1_nodes = [node for node in T.nodes if T.degree(node) == 1] 
    max_path = []
    max_num_nodes = -float('inf')
    for i in range(len(degree1_nodes)-1):
        for j in range(i+1, len(degree1_nodes)):
            if nx.has_path(T, degree1_nodes[i], degree1_nodes[j]):
                path = nx.shortest_path(T, degree1_nodes[i], degree1_nodes[j])
                num_nodes = len(path)
                if num_nodes > max_num_nodes:
                    max_num_nodes = num_nodes
                    max_path = path
    max_edges = [(max_path[i], max_path[i+1]) for i in range(len(max_path)-1)]
    return max_path,max_edges

def Get_Max_Path_Weight(T):
    #Weight
    degree1_nodes = [node for node in T.nodes if T.degree(node) == 1] 
    paths_and_weights = []
    max_path = []
    max_edges = []
    for i in range(len(degree1_nodes)-1):
        for j in range(i+1, len(degree1_nodes)):
            if nx.has_path(T, degree1_nodes[i], degree1_nodes[j]):
                path = nx.shortest_path(T, degree1_nodes[i], degree1_nodes[j]) 
                path_weight = 0
                for k in range(len(path)-1):
                    path_weight += T[path[k]][path[k+1]]['weight']
                paths_and_weights.append((path, path_weight))
    if len(paths_and_weights) != 0:
        max_weight = max([weight for path, weight in paths_and_weights])
        max_path = [path for path, weight in paths_and_weights if weight == max_weight][0]
        max_edges = [(max_path[i], max_path[i+1]) for i in range(len(max_path)-1)]
    return max_path,max_edges


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


def Extend_Skeleton_Coords(skeleton_coords,filament_mask_2D):
    add_coords = []
#     G_longest_skeleton,T_longest_skeleton = Graph_Infor_Connected(skeleton_coords)
    G_longest_skeleton,T_longest_skeleton = Graph_Infor(skeleton_coords)
    end_points = np.array(T_longest_skeleton.degree)[:,0][np.where(np.array(T_longest_skeleton.degree)[:,1]==1)[0]]
    for end_point_id in range(len(end_points)):
        current_node_0 = end_points[end_point_id]
        current_node_1 = list(T_longest_skeleton.neighbors(current_node_0))
        current_node_2 = list(T_longest_skeleton.neighbors(current_node_1[0]))
        current_node_2.remove(current_node_0)
        current_node_3 = list(T_longest_skeleton.neighbors(current_node_2[0]))
        current_node_3.remove(current_node_1[0])
        if len(current_node_3) == 1:
            used_points = [current_node_0,current_node_1[0],current_node_2[0],current_node_3[0]]
        else:
            used_points = [current_node_0,current_node_1[0],current_node_2[0]]
        used_points_len = len(used_points)
        direction = np.array([0,0])
        for i,j in zip(range(used_points_len-1),range(1,used_points_len)):
            direction += skeleton_coords[used_points[i]]-skeleton_coords[used_points[j]]
        direction = direction/(used_points_len-1)
        add_coord = skeleton_coords[end_points[end_point_id]].astype('float')
        while True:
            add_coord += direction
            if not (0 <= round(add_coord[0]) < filament_mask_2D.shape[0] and \
                    0 <= round(add_coord[1]) < filament_mask_2D.shape[1]):
                break
            if filament_mask_2D[round(add_coord[0]), round(add_coord[1])] == 0:
                break
            add_coords.append(list(np.around(add_coord,0).astype('int')))
    if len(add_coords) != 0:
        skeleton_coords = np.r_[skeleton_coords,np.array(add_coords)]
        skeleton_coords = np.array(list(set(list(map(tuple, skeleton_coords)))))
    return skeleton_coords


def Get_Longest_Skeleton_Coords(filament_mask_2D):
    # selem = morphology.square(3)
    closing_mask = morphology.closing(filament_mask_2D, morphology.square(3))
    skeleton = morphology.skeletonize(closing_mask)    
    props = measure.regionprops(measure.label(skeleton))
    all_skeleton_coords = props[0].coords
    for skeleton_index in range(1,len(props)):
        all_skeleton_coords = np.r_[all_skeleton_coords,props[skeleton_index].coords]
    return all_skeleton_coords


def Get_Single_Filament_Skeleton(filament_mask_2D):
    all_skeleton_coords = Get_Longest_Skeleton_Coords(filament_mask_2D)
    if len(all_skeleton_coords)>4:
        all_skeleton_coords = Extend_Skeleton_Coords(all_skeleton_coords,filament_mask_2D)
    G_longest_skeleton,T_longest_skeleton = Graph_Infor_Connected(all_skeleton_coords)
    max_path,max_edges = Get_Max_Path_Weight(T_longest_skeleton)
    single_longest_skeleton_coords = all_skeleton_coords[max_path]
    skeleton_coords_2D = single_longest_skeleton_coords
    filament_skeleton = np.zeros_like(filament_mask_2D)
    for coord_i in range(len(skeleton_coords_2D)):
        filament_skeleton[skeleton_coords_2D[coord_i][0],skeleton_coords_2D[coord_i][1]]=1
    skeleton_coords_2D,small_sc = Trim_Skeleton_Coords_2D(skeleton_coords_2D)
    return skeleton_coords_2D,filament_skeleton,all_skeleton_coords


def Cal_B_Spline(SampInt,skeleton_coords_2D,fil_mask):
    random_coords = np.random.random(skeleton_coords_2D.shape)/1000000-np.random.random(skeleton_coords_2D.shape)/1000000
    skeleton_coords_2D = skeleton_coords_2D+random_coords
    x, y = skeleton_coords_2D[:,0], skeleton_coords_2D[:,1]
    nest = -1 
    tckp, up = splprep([x,y], nest = -1)
    xspline, yspline = splev(up, tckp)
    xprime, yprime = splev(up, tckp, der=1)
    logic_used_spline = (xspline>0) * (xspline < fil_mask.shape[0]-0.5) * (yspline>0) * (yspline < fil_mask.shape[1]-0.5)
    xspline = xspline[logic_used_spline]
    yspline = yspline[logic_used_spline]
    xprime = xprime[logic_used_spline]
    yprime = yprime[logic_used_spline]
    pts_mask = (fil_mask[np.round(xspline[0:-1:SampInt]).astype(int),np.round(yspline[0:-1:SampInt]).astype(int)])
    xspline = xspline[0:-1:SampInt][pts_mask]
    yspline = yspline[0:-1:SampInt][pts_mask]
    xprime = xprime[0:-1:SampInt][pts_mask]
    yprime = yprime[0:-1:SampInt][pts_mask]
    points = np.c_[yspline, xspline]
    fprime = np.c_[yprime, xprime]
    return points,fprime


def Profile_Builder(image,mask,point,derivative,shift=True,fold=False):
    '''
    This function is adapted from RadFil.
    Build the profile using array manipulation, instead of looping.
    '''
    # Read the point and double check whether it's inside the mask.
    x0, y0 = point
    if (not mask[int(round(y0)), int(round(x0))]):
        raise ValueError("The point is not in the mask.")

    # Create the grid to calculate where the profile cut crosses edges of the
    # pixels.
    shapex, shapey = image.shape[1], image.shape[0]
    edgex, edgey = np.arange(.5, shapex-.5, 1.), np.arange(.5, shapey-.5, 1.)

    # Extreme cases when the derivative is (1, 0) or (0, 1)
    if (derivative[0] == 0) or (derivative[1] == 0):
        if (derivative[0] == 0) and (derivative[1] == 0):
            raise ValueError("Both components of the derivative are zero; unable to derive a tangent.")
        elif (derivative[0] == 0):
            y_edgex = []
            edgex = []
            x_edgey = np.ones(len(edgey))*x0
        elif (derivative[1] == 0):
            y_edgex = np.ones(len(edgex))*y0
            x_edgey = []
            edgey = []

    ## The regular cases go here: calculate the crossing points of the cut and the grid.
    else:
        slope = -1./(derivative[1]/derivative[0])
        y_edgex = slope*(edgex - x0) + y0
        x_edgey = (edgey - y0)/slope + x0

        ### Mask out points outside the image.
        pts_maskx = ((np.round(x_edgey) >= 0.) & (np.round(x_edgey) < shapex))
        pts_masky = ((np.round(y_edgex) >= 0.) & (np.round(y_edgex) < shapey))

        edgex, edgey = edgex[pts_masky], edgey[pts_maskx]
        y_edgex, x_edgey = y_edgex[pts_masky], x_edgey[pts_maskx]

    # Sort the points to find the center of each segment inside a single pixel.
    ## This also deals with when the cut crosses at the 4-corner point(s).
    ## The sorting is done by sorting the x coordinates
    stack = sorted(list(set(zip(np.concatenate([edgex, x_edgey]),\
                       np.concatenate([y_edgex, edgey])))))
    coords_total = stack[:-1]+.5*np.diff(stack, axis = 0)
    ## extract the values from the image and the original mask
    #setup interpolation 
    xgrid=np.arange(0.5,image.shape[1]+0.5,1.0)
    ygrid=np.arange(0.5,image.shape[0]+0.5,1.0)
    interpolator = RegularGridInterpolator((xgrid,ygrid),image.T,bounds_error=False,fill_value=None)
    
    image_line=interpolator(coords_total)
    #image_line = image[np.round(centers[:, 1]).astype(int), np.round(centers[:, 0]).astype(int)]
    
    mask_line = mask[np.round(coords_total[:, 1]).astype(int), np.round(coords_total[:, 0]).astype(int)]
    #### select the part of the mask that includes the original point
    mask_p0 = (np.round(coords_total[:, 0]).astype(int) == int(round(x0)))&\
              (np.round(coords_total[:, 1]).astype(int) == int(round(y0)))
    mask_line = (morphology.label(mask_line) == morphology.label(mask_line)[mask_p0])
    
    # Extract the profile from the image.
    ## for the points within the original mask; to find the peak
    if derivative[1] < 0.:
        image_line0 = image_line[mask_line][::-1]
    else:
        image_line0 = image_line[mask_line]
    ## for the entire map
    #This is different from RadFil. It will improve the accecuracy of the profiles, especially for curving skeleton. 
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
    ## find the end points of the cuts (within the original mask)
    start, end = peak_finder[0], peak_finder[-1]
 
    ## find the peak here
    xpeak, ypeak = peak_finder[image_line0 == np.nanmax(image_line0)][0]
    ## the peak mask is used to determine where to unfold when shift = True
    mask_peak = (np.round(coords_total[:, 0]).astype(int) == int(round(xpeak)))&\
                (np.round(coords_total[:, 1]).astype(int) == int(round(ypeak)))
    
    mass_array = np.c_[image_line0,image_line0]
    xcom, ycom = np.around((np.c_[mass_array]*peak_finder).sum(0)\
            /image_line0.sum(),3).tolist()
    mask_com = (np.round(coords_total[:, 0]).astype(int) == int(round(xcom)))&\
                (np.round(coords_total[:, 1]).astype(int) == int(round(ycom)))
    ## plot the cut
#     axis.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth = 1.,alpha=1)

    # Shift. Peak
    if shift:
        final_dist = np.hypot(coords_total[:, 0]-xpeak, coords_total[:, 1]-ypeak)
        # unfold
        pos0 = np.where(mask_peak)[0][0]
        final_dist[:pos0] = -final_dist[:pos0]
    else:
        final_dist = np.hypot(coords_total[:, 0]-x0, coords_total[:, 1]-y0)
        # unfold
        pos0 = np.where(mask_p0)[0][0]
        final_dist[:pos0] = -final_dist[:pos0]

    # Fold
    if fold:
            final_dist = abs(final_dist)
            
    #Com. This is new part.   
    if shift:
        final_dist_com = np.hypot(coords_total[:, 0]-xcom, coords_total[:, 1]-ycom)
        # unfold
        if len(np.where(mask_com)[0]) == 0:
            pos0 = np.where(mask_p0)[0][0]
        else:
            pos0 = np.where(mask_com)[0][0]
        final_dist_com[:pos0] = -final_dist_com[:pos0]
    else:
        final_dist_com = np.hypot(coords_total[:, 0]-x0, coords_total[:, 1]-y0)
        # unfold
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

def Get_Sub_Mask(point,regions_data,related_ids_T,connected_ids_dict,clump_coords_dict,start_coords):
    if len(regions_data.shape) == 2:
        clump_id = regions_data[np.int64(np.around(point))[1],np.int64(np.around(point))[0]] -1 
        connected_ids_i = [clump_id] + connected_ids_dict[clump_id]
        fil_mask_sub_profiles = np.zeros_like(regions_data)
        for connected_id in connected_ids_i:
            if connected_id in related_ids_T:
                coords = clump_coords_dict[connected_id]
                fil_mask_sub_profiles[(coords[:,0]-start_coords[0],coords[:,1]-start_coords[1])] = 1
    elif len(regions_data.shape) == 3:
        clump_ids = regions_data[:,np.int64(np.around(point))[1],np.int64(np.around(point))[0]] -1 
        clump_ids = list(set(clump_ids))
        if -1 in clump_ids:
            clump_ids.remove(-1)
        connected_ids_i = []
        for clump_id in clump_ids:
            connected_ids_i += [clump_id] + connected_ids_dict[clump_id]
        fil_mask_sub_profiles = np.zeros_like(regions_data.sum(0))
        for connected_id in connected_ids_i:
            if connected_id in related_ids_T:
                if connected_id in related_ids_T:
                    coords = clump_coords_dict[connected_id]
                    fil_mask_sub_profiles[(coords[:,1]-start_coords[1],coords[:,2]-start_coords[2])] = 1
    return fil_mask_sub_profiles
    

def Cal_Dictionary_Cuts(SampInt,CalSub,regions_data,related_ids_T,connected_ids_dict,clump_coords_dict,\
                        skeleton_coords_2D,fil_image,fil_mask,dictionary_cuts,start_coords=None): 
    if len(skeleton_coords_2D)>3:
        points,fprime = Cal_B_Spline(SampInt,skeleton_coords_2D,fil_mask)
        points_updated = points.copy().tolist()
        fprime_updated = fprime.copy().tolist()
        for point_id in range(len(points)):
            fil_image_shape = fil_image.shape
            if np.round(points[point_id][0]) > 0 and np.round(points[point_id][1]) >0 \
                and np.round(points[point_id][0]) < fil_image_shape[1]-1 and \
                    np.round(points[point_id][1]) < fil_image_shape[0]-1: 
                if CalSub:
                    fil_mask = Get_Sub_Mask(points[point_id],regions_data,related_ids_T,connected_ids_dict,\
                                            clump_coords_dict,start_coords)
                profile = Profile_Builder(fil_image,fil_mask,points[point_id],fprime[point_id],shift=True,fold=False)
                dictionary_cuts['distance'].append(profile[0])
                dictionary_cuts['profile'].append(profile[1])
                dictionary_cuts['plot_peaks'].append(profile[2])
                dictionary_cuts['plot_cuts'].append(profile[3])
                mask_width = Dists_Array([profile[3][0]], [profile[3][1]])[0][0]
                dictionary_cuts['mask_width'].append(np.around(mask_width,3))
                dictionary_cuts['distance_com'].append(profile[4])
                dictionary_cuts['plot_coms'].append(profile[5])
            else:
                points_updated.remove(points[point_id].tolist())
                fprime_updated.remove(fprime[point_id].tolist())
        dictionary_cuts['points'].append(np.array(points_updated))
        dictionary_cuts['fprime'].append(np.array(fprime_updated))
    return dictionary_cuts


def Update_Max_Path_Record(max_path_record):
    max_path_lens = []
    for max_path_i in max_path_record:
        max_path_lens.append(len(max_path_i))
    max_path_record_array = np.array(max_path_record, dtype=object)
    max_path_record = max_path_record_array[np.argsort(max_path_lens)[::-1]].tolist()
    max_path_record_copy = max_path_record.copy()
    max_path_T = max_path_record_copy[0].copy()
    max_path_record_T = [max_path_record_copy[0].copy()]
    max_path_record_copy.remove(max_path_record_copy[0])
    while len(max_path_record_copy) > 0:
        len_1 = len(max_path_record_copy)
        for max_path_i in max_path_record_copy:
            for max_path_ii in max_path_i:
                if max_path_ii in max_path_T:
                    max_path_record_T.append(max_path_i)
                    max_path_record_copy.remove(max_path_i)
                    max_path_T += max_path_i
                    break
        len_2 = len(max_path_record_copy)
        if len_1 == len_2:
            max_path_record_T += max_path_record_copy
            break
    return max_path_record_T


def Search_Max_Path_And_Edges(paths_and_weights):
    max_path = []
    max_edges = []
    if len(paths_and_weights) != 0:
        max_weight = max([weight for path, weight in paths_and_weights])
        max_path = [path for path, weight in paths_and_weights if weight == max_weight][0]
        max_edges = [(max_path[i], max_path[i+1]) for i in range(len(max_path)-1)]
    return max_weight,max_path,max_edges
    

def Get_Max_Path_Intensity_Weighted_Fast(fil_mask,Tree,mask_coords,clump_numbers):
    min_weight = float('inf')
    edge_coords_1 = np.where(mask_coords[:,0] == mask_coords[:,0].min())[0].tolist()
    edge_coords_2 = np.where(mask_coords[:,0] == mask_coords[:,0].max())[0].tolist()
    edge_coords_3 = np.where(mask_coords[:,1] == mask_coords[:,1].min())[0].tolist()
    edge_coords_4 = np.where(mask_coords[:,1] == mask_coords[:,1].max())[0].tolist()
    edge_coords = list(set(edge_coords_1 + edge_coords_2 + edge_coords_3 + edge_coords_4))
    if clump_numbers < 300:
        fil_mask_erosion = morphology.binary_erosion(fil_mask, morphology.disk(1))
        fil_mask_dilation = morphology.binary_dilation(fil_mask, morphology.disk(1))
        contour_data = fil_mask_dilation*~fil_mask_erosion
        degree1_nodes = [node for node in Tree.nodes if Tree.degree(node) == 1 and \
                         contour_data[mask_coords[node][0],mask_coords[node][1]]]
        for (u, v, data) in Tree.edges(data=True):
            if data['weight'] < min_weight:
                min_weight = data['weight']
                min_weight_node = [u, v]
    #     min_weight_node.append(max_coord_index)
    #     min_weight_node = list(set(min_weight_node))
        paths_and_weights = []
        for source_node_id in min_weight_node:
            for target_node_id in degree1_nodes:
                path = nx.shortest_path(Tree, source_node_id, target_node_id)
                path_weight = 0
                for i in range(len(path)-1):
                    path_weight += 1/Tree[path[i]][path[i+1]]['weight']
                paths_and_weights.append((path, path_weight))
                max_weight_1,max_path_1,max_edges_1 = Search_Max_Path_And_Edges(paths_and_weights)
        paths_and_weights = []
        for target_node_id in degree1_nodes:
            path = nx.shortest_path(Tree, max_path_1[-1], target_node_id)
            in_path_logic = min_weight_node[0] in path 
            for min_weight_node_id in range(1,len(min_weight_node)):
                in_path_logic = in_path_logic and (min_weight_node[min_weight_node_id] in path)
            if in_path_logic:
                path_weight = 0
                for i in range(len(path)-1):
                    path_weight += 1/Tree[path[i]][path[i+1]]['weight']
                paths_and_weights.append((path, path_weight))
            for edge_coord in edge_coords:
                path = nx.shortest_path(Tree, target_node_id, edge_coord)
                path_weight = 0
                for i in range(len(path)-1):
                    path_weight += 1/Tree[path[i]][path[i+1]]['weight']
                paths_and_weights.append((path, path_weight))
    else:
        paths_and_weights = []
        for i in range(len(edge_coords)-1):
            for j in range(i+1, len(edge_coords)):
                path = nx.shortest_path(Tree, edge_coords[i], edge_coords[j])
                path_weight = 0
                for k in range(len(path)-1):
                    path_weight += 1/Tree[path[k]][path[k+1]]['weight']
                    paths_and_weights.append((path, path_weight))
    max_weight_2,max_path_2,max_edges_2 = Search_Max_Path_And_Edges(paths_and_weights)
    return max_path_2,max_edges_2


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
    

def Get_Max_Path_Intensity_Weighted(fil_mask, mask_coords, Tree, common_mask_coords_id=None, common_mask_coords_centers_id=None):
    """
    Get the maximum intensity-weighted path within a filament mask.

    This function searches for the most significant path in the tree built from
    the filament mask. The path weight is defined as the sum of inverse edge
    weights, so paths passing through higher-intensity regions are favored.

    Two search modes are supported:
    1. If ``common_mask_coords_id`` is given, candidate paths are searched from
       the specified nodes to tree endpoints.
    2. Otherwise, candidate paths are searched between all endpoint pairs.

    If ``common_mask_coords_centers_id`` is provided, paths passing through the
    specified center nodes are preferred. If no such path is found, the constraint
    is relaxed and all valid candidate paths are considered.

    Parameters
    ----------
    fil_mask : ndarray
        2D binary mask of the filament.
    mask_coords : ndarray
        Coordinates of mask nodes, aligned with node IDs in ``Tree``.
    Tree : networkx.Graph
        Tree structure describing the filament connectivity.
    common_mask_coords_id : list, optional
        Node IDs used as constrained starting nodes when searching candidate paths.
    common_mask_coords_centers_id : list, optional
        Node IDs of important center positions that candidate paths are preferred
        to pass through.

    Returns
    -------
    max_path : list
        Node sequence of the maximum intensity-weighted path.
    max_edges : list
        Edge sequence corresponding to ``max_path``.
    """

    # Fill holes inside the mask and extract the contour map
    fil_mask, contour_data = Fill_Mask_Holes(fil_mask)
    
    # Select endpoint nodes on the filament boundary
    # These nodes are leaf nodes in the tree and are used as path termini
    degree1_nodes = [node for node in Tree.nodes if Tree.degree(node) == 1 and 
                                        contour_data[mask_coords[node][0], mask_coords[node][1]]]
    
    paths_and_weights = []

    # Search paths from the specified common nodes to boundary endpoints
    if common_mask_coords_id is not None and len(common_mask_coords_id) > 0:
        for i in common_mask_coords_id:
            for j in range(len(degree1_nodes)):
                # Path from the constrained node to one boundary endpoint
                path = nx.shortest_path(Tree, i, degree1_nodes[j])

                # First try to keep paths passing through the required center node
                if common_mask_coords_centers_id[0] in path: 
                    path_weight = 0

                    # Accumulate inverse edge weights along the path
                    for k in range(len(path) - 1):
                        weight = Tree[path[k]][path[k + 1]]['weight']
                        if weight != 0:
                            path_weight += 1 / weight
                    paths_and_weights.append((path, path_weight))

        # If no path satisfies the center-node condition, remove this constraint
        if len(paths_and_weights) == 0:
            for i in common_mask_coords_id:
                for j in range(len(degree1_nodes)):
                    path = nx.shortest_path(Tree, i, degree1_nodes[j])
                    path_weight = 0
                    for k in range(len(path) - 1):
                        weight = Tree[path[k]][path[k + 1]]['weight']
                        if weight != 0:
                            path_weight += 1 / weight
                    paths_and_weights.append((path, path_weight))

    # Search paths between all endpoint pairs
    else:
        for i in range(len(degree1_nodes) - 1):
            for j in range(i + 1, len(degree1_nodes)):
                # Path connecting two boundary endpoints
                path = nx.shortest_path(Tree, degree1_nodes[i], degree1_nodes[j])

                # If center nodes are given, keep only paths crossing both ends of the center set
                if type(common_mask_coords_centers_id) is not type(None):
                    if common_mask_coords_centers_id[0] in path and common_mask_coords_centers_id[-1] in path:
                        path_weight = 0
                        for k in range(len(path) - 1):
                            weight = Tree[path[k]][path[k + 1]]['weight']
                            if weight != 0:
                                path_weight += 1 / weight
                        paths_and_weights.append((path, path_weight))
                else:
                    path_weight = 0

                    # Accumulate inverse edge weights for the full endpoint-to-endpoint path
                    for k in range(len(path) - 1):
                        weight = Tree[path[k]][path[k + 1]]['weight']
                        if weight != 0:
                            path_weight += 1 / weight
                    paths_and_weights.append((path, path_weight))

        # If no path satisfies the center-node condition, fall back to all endpoint pairs
        if len(paths_and_weights) == 0:
            for i in range(len(degree1_nodes) - 1):
                for j in range(i + 1, len(degree1_nodes)):
                    path = nx.shortest_path(Tree, degree1_nodes[i], degree1_nodes[j])
                    path_weight = 0
                    for k in range(len(path) - 1):
                        weight = Tree[path[k]][path[k + 1]]['weight']
                        if weight != 0:
                            path_weight += 1 / weight
                    paths_and_weights.append((path, path_weight))

    # Select the path with the largest accumulated intensity weight
    max_weight, max_path, max_edges = Search_Max_Path_And_Edges(paths_and_weights)
    
    return max_path, max_edges


def Get_Single_Filament_Skeleton_Weighted(fil_image, fil_mask, clump_numbers, common_sc_item=None, sub_centers_item=None, SmallSkeleton=6):
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
    common_mask_coords_centers_id = []
    
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
        if type(common_sc_item) != type(None):
            if tuple(mask_coords[i]) in map(tuple, common_sc_item) and i not in common_mask_coords_id:
                common_mask_coords_id.append(i)
        if type(sub_centers_item) != type(None):
            if tuple(mask_coords[i]) in map(tuple, sub_centers_item) and i not in common_mask_coords_centers_id:
                common_mask_coords_centers_id.append(i)
            
    # Create unique list of common coordinate IDs
    if type(common_sc_item) == type(None):
        common_mask_coords_id = None
    
    if type(sub_centers_item) == type(None) or len(common_mask_coords_centers_id)==0:
        common_mask_coords_centers_id = None
    else:
        sort_ids = np.array([{tuple(r): i for i, r in enumerate(mask_coords[common_mask_coords_centers_id])}\
                             [tuple(row)] for row in sub_centers_item])
        common_mask_coords_centers_id = np.array(common_mask_coords_centers_id)[sort_ids]

    # Find minimum spanning tree
    Tree = nx.minimum_spanning_tree(Graph_find_skeleton)
    
    # Find the longest path through the tree
    if clump_numbers < 100 or CalSubSK:
        max_path, max_edges = Get_Max_Path_Intensity_Weighted(fil_mask, mask_coords, Tree, \
                                                              common_mask_coords_id,common_mask_coords_centers_id)
    else:
        max_path, max_edges = Get_Max_Path_Intensity_Weighted_Fast(fil_mask, mask_coords, Tree, clump_numbers)

    # Extract coordinates for the maximum path
    skeleton_coords_2D = mask_coords[max_path]
        
    # Trim and refine the skeleton
    skeleton_coords_2D, small_sc = Trim_Skeleton_Coords_2D(skeleton_coords_2D, SmallSkeleton)
    # print('skeleton_coords_2D:',small_sc, skeleton_coords_2D)
    return skeleton_coords_2D, small_sc


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
    G_sorted_skeleton, T_sorted_skeleton = Graph_Infor(coords)
    
    # Find the longest path through the skeleton
    max_path, max_edges = Get_Max_Path_Weight(T_sorted_skeleton)
    
    sorted_skeleton_coords = coords[max_path]
    small_sc = len(sorted_skeleton_coords) <= SmallSkeleton
    return sorted_skeleton_coords, small_sc




