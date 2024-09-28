import time
import numpy as np
from skimage import filters, measure, morphology
from scipy.stats import multivariate_normal
from scipy import optimize, linalg
import scipy.ndimage as ndimage
from collections import defaultdict
from scipy.interpolate import splprep, splev, RegularGridInterpolator
import networkx as nx

from tqdm import tqdm


def Filament_Coords(origin_data, regions_data, data_wcs, clump_coords_dict, related_ids_T, CalSub=False):
    filament_coords = clump_coords_dict[related_ids_T[0]]
    for clump_id in related_ids_T[1:]:
        filament_coords = np.r_[filament_coords, clump_coords_dict[clump_id]]
    x_min = np.array(filament_coords[:, 0]).min()
    x_max = np.array(filament_coords[:, 0]).max()
    y_min = np.array(filament_coords[:, 1]).min()
    y_max = np.array(filament_coords[:, 1]).max()
    z_min = np.array(filament_coords[:, 2]).min()
    z_max = np.array(filament_coords[:, 2]).max()
    length = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    if np.int32(length * 0.1) % 2 == 0:
        length += np.int32(length * 0.1) + 5
    else:
        length += np.int32(length * 0.1) + 6
    filament_item = np.zeros([length, length, length])
    regions_data_T = np.zeros([length, length, length], dtype=np.int32)
    start_x = np.int64((length - (x_max - x_min)) / 2)
    start_y = np.int64((length - (y_max - y_min)) / 2)
    start_z = np.int64((length - (z_max - z_min)) / 2)
    filament_item[filament_coords[:, 0] - x_min + start_x, filament_coords[:, 1] - y_min + start_y, \
                  filament_coords[:, 2] - z_min + start_z] = \
        origin_data[filament_coords[:, 0], filament_coords[:, 1], filament_coords[:, 2]]
    regions_data_T[filament_coords[:, 0] - x_min + start_x, filament_coords[:, 1] - y_min + start_y, \
                   filament_coords[:, 2] - z_min + start_z] = \
        regions_data[filament_coords[:, 0], filament_coords[:, 1], filament_coords[:, 2]]
    start_coords = np.array([x_min - start_x, y_min - start_y, z_min - start_z])

    filament_item_mask_2D = np.zeros_like(filament_item.sum(0), dtype=np.int32)
    filament_item_mask_2D[filament_coords[:, 1] - y_min + start_y, filament_coords[:, 2] - z_min + start_z] = 1
    if CalSub == False:
        box_region = measure.regionprops(filament_item_mask_2D)
        lb_area = box_region[0].area
        data_wcs_item = data_wcs.deepcopy()
        data_wcs_item.wcs.crpix[0] -= start_coords[2]
        data_wcs_item.wcs.crpix[1] -= start_coords[1]
        data_wcs_item.wcs.crpix[2] -= start_coords[0]
    else:
        lb_area = None
        data_wcs_item = None

    return filament_coords, filament_item, data_wcs_item, regions_data_T, start_coords, filament_item_mask_2D, lb_area


def Get_Line_Coords_2D(point_a, point_b):
    coords = []
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    sort_index = np.argsort(np.abs(point_b - point_a))
    point_a_temp = point_a[sort_index]
    point_b_temp = point_b[sort_index]

    x1, y1 = point_a_temp[0], point_a_temp[1]
    x2, y2 = point_b_temp[0], point_b_temp[1]
    kx, ky = (x2 - x1), (y2 - y1)
    k_norm = np.sqrt(kx ** 2 + ky ** 2)
    for y in range(min(int(round(y1)), int(round(y2))), max(int(round(y1)), int(round(y2))) + 1):
        x = x1 + kx * (y - y1) / ky
        coords.append((int(round(x)), int(round(y))))
    index_0 = np.where(sort_index == 0)[0]
    index_1 = np.where(sort_index == 1)[0]
    line_coords = np.c_[np.array(coords)[:, index_0], np.array(coords)[:, index_1]].astype('int')
    return line_coords


def Get_Line_Coords_3D(point_a, point_b):
    coords = []
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    sort_index = np.argsort(np.abs(point_b - point_a))
    point_a_temp = point_a[sort_index]
    point_b_temp = point_b[sort_index]

    x1, y1, z1 = point_a_temp[0], point_a_temp[1], point_a_temp[2]
    x2, y2, z2 = point_b_temp[0], point_b_temp[1], point_b_temp[2]
    kx, ky, kz = (x2 - x1), (y2 - y1), (z2 - z1)
    k_norm = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    for z in range(min(int(round(z1)), int(round(z2))), max(int(round(z1)), int(round(z2))) + 1):
        x = x1 + kx * (z - z1) / kz
        y = y1 + ky * (z - z1) / kz
        coords.append((int(round(x)), int(round(y)), int(round(z))))
    index_0 = np.where(sort_index == 0)[0]
    index_1 = np.where(sort_index == 1)[0]
    index_2 = np.where(sort_index == 2)[0]
    coords_xy = np.c_[np.array(coords)[:, index_0], np.array(coords)[:, index_1]]
    line_coords = np.c_[coords_xy, np.array(coords)[:, index_2]].astype('int')
    return line_coords


def Get_DV(box_data, box_center):
    # 3D
    box_data_sum = box_data.sum(0)
    box_region = np.where(box_data_sum != 0)
    A11 = np.sum((box_region[0] - box_center[1]) ** 2 * \
                 box_data_sum[box_region])
    A12 = -np.sum((box_region[0] - box_center[1]) * \
                  (box_region[1] - box_center[2]) * \
                  box_data_sum[box_region])
    A21 = A12
    A22 = np.sum((box_region[1] - box_center[2]) ** 2 * \
                 box_data_sum[box_region])
    A = np.array([[A11, A12], [A21, A22]]) / len(box_region[0])
    D, V = np.linalg.eig(A)
    if D[0] < D[1]:
        D = D[[1, 0]]
        V = V[[1, 0]]
    if V[1][0] < 0 and V[0][0] > 0 and V[1][1] > 0:
        V = -V
    size_ratio = np.sqrt(D[0] / D[1])
    angle = np.around(np.arccos(V[0][0]) * 180 / np.pi - 90, 2)
    return D, V, size_ratio, angle


def Dists_Array(matrix_1, matrix_2):
    matrix_1 = np.array(matrix_1)
    matrix_2 = np.array(matrix_2)
    # num_1 = matrix_1.shape[0]
    # num_2 = matrix_2.shape[0]
    dist_1 = -2 * np.dot(matrix_1, matrix_2.T)
    dist_2 = np.sum(np.square(matrix_1), axis=1, keepdims=True)
    dist_3 = np.sum(np.square(matrix_2), axis=1)
    random_dist = np.random.random(dist_1.shape) / 1000000
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
    return Graph, Tree


def Graph_Infor_Connected(points):
    #     points = np.array(sorted(points, key=lambda x: x[0], reverse=False))
    dist_matrix = Dists_Array(points, points)
    mask_coords_in_dm = np.where(np.logical_and(dist_matrix > 0.5, dist_matrix < 2))
    Graph = nx.Graph()
    for i, j in zip(mask_coords_in_dm[0], mask_coords_in_dm[1]):
        weight_ij = 1  # dist_matrix[i,j]
        Graph.add_edge(i, j, weight=weight_ij)
    Tree = nx.minimum_spanning_tree(Graph)
    return Graph, Tree


def Graph_Infor_SubStructure(origin_data, filament_mask_2D, filament_centers_LBV, filament_clumps_id,
                             connected_ids_dict):
    points_V = filament_centers_LBV[:, 2]
    points_LB = np.c_[filament_centers_LBV[:, 0], filament_centers_LBV[:, 1]]
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
                line_coords = Get_Line_Coords_2D(points_LB[i], points_LB[j])
                mask_2D_ids = filament_mask_2D[line_coords[:, 1], line_coords[:, 0]]  # mask_2D_ids = [0,1]
                #                 weight_ij = origin_data[filament_centers_LBV[i][2],filament_centers_LBV[i][1],\
                #                 filament_centers_LBV[i][0]] + \
                #                 origin_data[filament_centers_LBV[j][2],filament_centers_LBV[j][1],filament_centers_LBV[j][0]]
                line_coords = Get_Line_Coords_3D(filament_centers_LBV[i], filament_centers_LBV[j])
                weight_ij = origin_data[line_coords[:, 2], line_coords[:, 1], line_coords[:, 0]].mean()
                if 0 not in mask_2D_ids:
                    weight = dist_matrix[i, j] * np.abs(points_V[i] - points_V[j]) / (weight_ij)
                    Graph.add_edge(i, j, weight=weight)
    Tree = nx.minimum_spanning_tree(Graph)
    return Graph, Tree


def Get_Max_Path_Node(T):
    # Node
    degree1_nodes = [node for node in T.nodes if T.degree(node) == 1]
    max_path = []
    max_num_nodes = -float('inf')
    for i in range(len(degree1_nodes) - 1):
        for j in range(i + 1, len(degree1_nodes)):
            if nx.has_path(T, degree1_nodes[i], degree1_nodes[j]):
                path = nx.shortest_path(T, degree1_nodes[i], degree1_nodes[j])
                num_nodes = len(path)
                if num_nodes > max_num_nodes:
                    max_num_nodes = num_nodes
                    max_path = path
    max_edges = [(max_path[i], max_path[i + 1]) for i in range(len(max_path) - 1)]
    return max_path, max_edges


def Get_Max_Path_Weight(T):
    # Weight
    degree1_nodes = [node for node in T.nodes if T.degree(node) == 1]
    paths_and_weights = []
    max_path = []
    max_edges = []
    for i in range(len(degree1_nodes) - 1):
        for j in range(i + 1, len(degree1_nodes)):
            if nx.has_path(T, degree1_nodes[i], degree1_nodes[j]):
                path = nx.shortest_path(T, degree1_nodes[i], degree1_nodes[j])
                path_weight = 0
                for k in range(len(path) - 1):
                    path_weight += T[path[k]][path[k + 1]]['weight']
                paths_and_weights.append((path, path_weight))
    if len(paths_and_weights) != 0:
        max_weight = max([weight for path, weight in paths_and_weights])
        max_path = [path for path, weight in paths_and_weights if weight == max_weight][0]
        max_edges = [(max_path[i], max_path[i + 1]) for i in range(len(max_path) - 1)]
    return max_path, max_edges


def Get_Max_Path_Weight_SubStructure(origin_data, filament_centers_LBV, T):
    # Weight
    points_V = filament_centers_LBV[:, 2]
    points_LB = np.c_[filament_centers_LBV[:, 0], filament_centers_LBV[:, 1]]
    dist_matrix = Dists_Array(points_LB, points_LB)
    filament_centers_LBV = np.int64(np.around(filament_centers_LBV))

    degree1_nodes = [node for node in T.nodes if T.degree(node) == 1]
    paths_and_weights = []
    max_path = []
    max_edges = []
    for i in range(len(degree1_nodes) - 1):
        for j in range(i + 1, len(degree1_nodes)):
            if nx.has_path(T, degree1_nodes[i], degree1_nodes[j]):
                path = nx.shortest_path(T, degree1_nodes[i], degree1_nodes[j])
                path_weight = 0
                for k in range(len(path) - 1):
                    #                     weight_ij = \
                    #                     origin_data[filament_centers_LBV[path[k]][2],filament_centers_LBV[path[k]][1],\
                    #                     filament_centers_LBV[path[k]][0]] + \
                    #                     origin_data[filament_centers_LBV[path[k+1]][2],filament_centers_LBV[path[k+1]][1],\
                    #                     filament_centers_LBV[path[k+1]][0]]
                    line_coords = Get_Line_Coords_3D(filament_centers_LBV[path[k]], filament_centers_LBV[path[k + 1]])
                    weight_ij = origin_data[line_coords[:, 2], line_coords[:, 1], line_coords[:, 0]].mean()
                    path_weight += dist_matrix[path[k], path[k + 1]] * np.abs(
                        points_V[path[k]] - points_V[path[k + 1]]) * weight_ij
                #                     path_weight += T[path[k]][path[k+1]]['weight']
                paths_and_weights.append((path, path_weight))
    if len(paths_and_weights) != 0:
        max_weight = max([weight for path, weight in paths_and_weights])
        max_path = [path for path, weight in paths_and_weights if weight == max_weight][0]
        max_edges = [(max_path[i], max_path[i + 1]) for i in range(len(max_path) - 1)]
    return max_path, max_edges


def Get_Max_Path_Recursion(origin_data, filament_centers_LBV, max_path_record, max_edges_record, G, T):
    max_path, max_edges = Get_Max_Path_Weight_SubStructure(origin_data, filament_centers_LBV, T)
    max_path_record.append(max_path)
    max_edges_record.append(max_edges)
    new_T = T.copy()
    for i in range(len(max_path) - 1):
        new_T.remove_edge(max_path[i], max_path[i + 1])

    sub_degrees = new_T.degree
    nodes_T = new_T.nodes
    new_T_2 = new_T.copy()
    nodes = nodes_T
    for node in nodes:
        if sub_degrees[node] == 0:
            new_T_2.remove_node(node)

    if new_T_2.nodes != 0:
        subgraphs = list(nx.connected_components(new_T_2))
        for subgraph in subgraphs:
            new_T_3 = G.subgraph(subgraph)
            T = new_T_3.copy()
            for sub_edge in T.edges:
                if sub_edge not in new_T_2.edges:
                    T.remove_edge(sub_edge[0], sub_edge[1])
            max_path_record, max_edges_record = \
                Get_Max_Path_Recursion(origin_data, filament_centers_LBV, max_path_record, max_edges_record, G, T)
    return max_path_record, max_edges_record


def Extend_Skeleton_Coords(skeleton_coords, filament_mask_2D):
    add_coords = []
    #     G_longest_skeleton,T_longest_skeleton = Graph_Infor_Connected(skeleton_coords)
    G_longest_skeleton, T_longest_skeleton = Graph_Infor(skeleton_coords)
    end_points = np.array(T_longest_skeleton.degree)[:, 0][np.where(np.array(T_longest_skeleton.degree)[:, 1] == 1)[0]]
    for end_point_id in range(len(end_points)):
        current_node_0 = end_points[end_point_id]
        current_node_1 = list(T_longest_skeleton.neighbors(current_node_0))
        current_node_2 = list(T_longest_skeleton.neighbors(current_node_1[0]))
        current_node_2.remove(current_node_0)
        current_node_3 = list(T_longest_skeleton.neighbors(current_node_2[0]))
        current_node_3.remove(current_node_1[0])
        if len(current_node_3) == 1:
            used_points = [current_node_0, current_node_1[0], current_node_2[0], current_node_3[0]]
        else:
            used_points = [current_node_0, current_node_1[0], current_node_2[0]]
        used_points_len = len(used_points)
        direction = np.array([0, 0])
        for i, j in zip(range(used_points_len - 1), range(1, used_points_len)):
            direction += skeleton_coords[used_points[i]] - skeleton_coords[used_points[j]]
        direction = direction / (used_points_len - 1)
        add_coord = skeleton_coords[end_points[end_point_id]].astype('float')
        while True:
            add_coord += direction
            if not (0 <= round(add_coord[0]) < filament_mask_2D.shape[0] and \
                    0 <= round(add_coord[1]) < filament_mask_2D.shape[1]):
                break
            if filament_mask_2D[round(add_coord[0]), round(add_coord[1])] == 0:
                break
            add_coords.append(list(np.around(add_coord, 0).astype('int')))
    if len(add_coords) != 0:
        skeleton_coords = np.r_[skeleton_coords, np.array(add_coords)]
        skeleton_coords = np.array(list(set(list(map(tuple, skeleton_coords)))))
    return skeleton_coords


def Get_Longest_Skeleton_Coords(filament_mask_2D):
    # selem = morphology.square(3)
    closing_mask = morphology.closing(filament_mask_2D, morphology.square(3))
    skeleton = morphology.skeletonize(closing_mask)
    props = measure.regionprops(measure.label(skeleton))
    all_skeleton_coords = props[0].coords
    for skeleton_index in range(1, len(props)):
        all_skeleton_coords = np.r_[all_skeleton_coords, props[skeleton_index].coords]
    return all_skeleton_coords


def Get_Single_Filament_Skeleton(filament_mask_2D):
    all_skeleton_coords = Get_Longest_Skeleton_Coords(filament_mask_2D)
    if len(all_skeleton_coords) > 4:
        all_skeleton_coords = Extend_Skeleton_Coords(all_skeleton_coords, filament_mask_2D)
    G_longest_skeleton, T_longest_skeleton = Graph_Infor_Connected(all_skeleton_coords)
    max_path, max_edges = Get_Max_Path_Weight(T_longest_skeleton)
    single_longest_skeleton_coords = all_skeleton_coords[max_path]
    skeleton_coords_2D = single_longest_skeleton_coords
    filament_skeleton = np.zeros_like(filament_mask_2D)
    for coord_i in range(len(skeleton_coords_2D)):
        filament_skeleton[skeleton_coords_2D[coord_i][0], skeleton_coords_2D[coord_i][1]] = 1
    skeleton_coords_2D,small_sc = Trim_Skeleton_Coords_2D(skeleton_coords_2D, filament_mask_2D)
    return skeleton_coords_2D,filament_skeleton, all_skeleton_coords


def Cal_B_Spline(SampInt, skeleton_coords_2D, fil_mask):
    random_coords = np.random.random(skeleton_coords_2D.shape) / 1000000 - np.random.random(
        skeleton_coords_2D.shape) / 1000000
    skeleton_coords_2D = skeleton_coords_2D + random_coords
    x, y = skeleton_coords_2D[:, 0], skeleton_coords_2D[:, 1]
    nest = -1
    tckp, up = splprep([x, y], nest=-1)
    xspline, yspline = splev(up, tckp)
    xprime, yprime = splev(up, tckp, der=1)
    logic_used_spline = (xspline > 0) * (xspline < fil_mask.shape[0] - 0.5) * (yspline > 0) * (
                yspline < fil_mask.shape[1] - 0.5)
    xspline = xspline[logic_used_spline]
    yspline = yspline[logic_used_spline]
    xprime = xprime[logic_used_spline]
    yprime = yprime[logic_used_spline]
    pts_mask = (fil_mask[np.round(xspline[0:-1:SampInt]).astype(int), np.round(yspline[0:-1:SampInt]).astype(int)])
    xspline = xspline[0:-1:SampInt][pts_mask]
    yspline = yspline[0:-1:SampInt][pts_mask]
    xprime = xprime[0:-1:SampInt][pts_mask]
    yprime = yprime[0:-1:SampInt][pts_mask]
    points = np.c_[yspline, xspline]
    fprime = np.c_[yprime, xprime]
    return points, fprime


def Profile_Builder(image, mask, point, derivative, shift=True, fold=False):
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
        slope = -1. / (derivative[1] / derivative[0])
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
    ## extract the values from the image and the original mask
    # setup interpolation
    xgrid = np.arange(0.5, image.shape[1] + 0.5, 1.0)
    ygrid = np.arange(0.5, image.shape[0] + 0.5, 1.0)
    interpolator = RegularGridInterpolator((xgrid, ygrid), image.T, bounds_error=False, fill_value=None)

    image_line = interpolator(coords_total)
    # image_line = image[np.round(centers[:, 1]).astype(int), np.round(centers[:, 0]).astype(int)]

    mask_line = mask[np.round(coords_total[:, 1]).astype(int), np.round(coords_total[:, 0]).astype(int)]
    #### select the part of the mask that includes the original point
    mask_p0 = (np.round(coords_total[:, 0]).astype(int) == int(round(x0))) & \
              (np.round(coords_total[:, 1]).astype(int) == int(round(y0)))
    mask_line = (morphology.label(mask_line) == morphology.label(mask_line)[mask_p0])

    # Extract the profile from the image.
    ## for the points within the original mask; to find the peak
    if derivative[1] < 0.:
        image_line0 = image_line[mask_line][::-1]
    else:
        image_line0 = image_line[mask_line]
    ## for the entire map
    # This is different from RadFil. It will improve the accecuracy of the profiles, especially for curving skeleton.
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
    mask_peak = (np.round(coords_total[:, 0]).astype(int) == int(round(xpeak))) & \
                (np.round(coords_total[:, 1]).astype(int) == int(round(ypeak)))

    mass_array = np.c_[image_line0, image_line0]
    xcom, ycom = np.around((np.c_[mass_array] * peak_finder).sum(0) \
                           / image_line0.sum(), 3).tolist()
    mask_com = (np.round(coords_total[:, 0]).astype(int) == int(round(xcom))) & \
               (np.round(coords_total[:, 1]).astype(int) == int(round(ycom)))
    ## plot the cut
    #     axis.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth = 1.,alpha=1)

    # Shift. Peak
    if shift:
        final_dist = np.hypot(coords_total[:, 0] - xpeak, coords_total[:, 1] - ypeak)
        # unfold
        pos0 = np.where(mask_peak)[0][0]
        final_dist[:pos0] = -final_dist[:pos0]
    else:
        final_dist = np.hypot(coords_total[:, 0] - x0, coords_total[:, 1] - y0)
        # unfold
        pos0 = np.where(mask_p0)[0][0]
        final_dist[:pos0] = -final_dist[:pos0]

    # Fold
    if fold:
        final_dist = abs(final_dist)

    # Com. This is new part.
    if shift:
        final_dist_com = np.hypot(coords_total[:, 0] - xcom, coords_total[:, 1] - ycom)
        # unfold
        pos0 = np.where(mask_com)[0][0]
        final_dist_com[:pos0] = -final_dist_com[:pos0]
    else:
        final_dist_com = np.hypot(coords_total[:, 0] - x0, coords_total[:, 1] - y0)
        # unfold
        pos0 = np.where(mask_p0)[0][0]
        final_dist_com[:pos0] = -final_dist_com[:pos0]

    # Fold
    if fold:
        final_dist = abs(final_dist)

    peak = np.around([xpeak, ypeak]).astype(int)
    com = np.array([xcom, ycom])
    return final_dist, image_line_T, peak, (start, end), final_dist_com, com


def Get_Sub_Mask(point, regions_data, related_ids_T, connected_ids_dict, clump_coords_dict, start_coords):
    if len(regions_data.shape) == 2:
        clump_id = regions_data[np.int64(np.around(point))[1], np.int64(np.around(point))[0]] - 1
        connected_ids_i = [clump_id] + connected_ids_dict[clump_id]
        fil_mask_sub_profiles = np.zeros_like(regions_data)
        for connected_id in connected_ids_i:
            if connected_id in related_ids_T:
                coords = clump_coords_dict[connected_id]
                fil_mask_sub_profiles[(coords[:, 0] - start_coords[0], \
                                       coords[:, 1] - start_coords[1])] = 1
    elif len(regions_data.shape) == 3:
        clump_ids = regions_data[:, np.int64(np.around(point))[1], np.int64(np.around(point))[0]] - 1
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
                    fil_mask_sub_profiles[(coords[:, 1] - start_coords[1], coords[:, 2] - start_coords[2])] = 1
    return fil_mask_sub_profiles


def Cal_Dictionary_Cuts(SampInt, CalSub, regions_data, related_ids_T, connected_ids_dict, clump_coords_dict, \
                        skeleton_coords_2D, fil_image, fil_mask, dictionary_cuts, start_coords=None):
    if len(skeleton_coords_2D) > 3:
        points, fprime = Cal_B_Spline(SampInt, skeleton_coords_2D, fil_mask)
        points_updated = points.copy().tolist()
        fprime_updated = fprime.copy().tolist()
        for point_id in range(len(points)):
            fil_image_shape = fil_image.shape
            if np.round(points[point_id][0]) > 0 and np.round(points[point_id][1]) > 0 \
                    and np.round(points[point_id][0]) < fil_image_shape[1] - 1 and \
                    np.round(points[point_id][1]) < fil_image_shape[0] - 1:
                if CalSub:
                    fil_mask = Get_Sub_Mask(points[point_id], regions_data, related_ids_T, connected_ids_dict, \
                                            clump_coords_dict, start_coords)
                profile = Profile_Builder(fil_image, fil_mask, points[point_id], fprime[point_id], shift=True,
                                          fold=False)
                dictionary_cuts['distance'].append(profile[0])
                dictionary_cuts['profile'].append(profile[1])
                dictionary_cuts['plot_peaks'].append(profile[2])
                dictionary_cuts['plot_cuts'].append(profile[3])
                mask_width = Dists_Array([profile[3][0]], [profile[3][1]])[0][0]
                dictionary_cuts['mask_width'].append(np.around(mask_width, 3))
                dictionary_cuts['distance_com'].append(profile[4])
                dictionary_cuts['plot_coms'].append(profile[5])
            else:
                points_updated.remove(points[point_id].tolist())
                fprime_updated.remove(fprime[point_id].tolist())
        dictionary_cuts['points'].append(np.array(points_updated))
        dictionary_cuts['fprime'].append(np.array(fprime_updated))
    return dictionary_cuts


def Update_Dictionary_Cuts(dictionary_cuts, start_coords):
    if len(dictionary_cuts['points']) > 0:
        dictionary_cuts['points'][-1] = dictionary_cuts['points'][-1] + start_coords[1:][::-1]
        len_new = len(dictionary_cuts['points'][-1])
        for key in ['plot_peaks', 'plot_cuts', 'plot_coms']:
            dictionary_cuts[key][-len_new:] = \
                list(np.array(dictionary_cuts[key])[-len_new:] + start_coords[1:][::-1])
    return dictionary_cuts


# def Get_Max_Path_Intensity_Weighted(fil_mask,Tree,mask_coords):
#     #Node
#     fil_mask_erosion = morphology.binary_erosion(fil_mask, morphology.selem.disk(1))
#     fil_mask_dilation = morphology.binary_dilation(fil_mask, morphology.selem.disk(1))
#     contour_data = fil_mask_dilation*~fil_mask_erosion
#     degree1_nodes = [node for node in Tree.nodes if Tree.degree(node) == 1 and \
#                         contour_data[mask_coords[node][0],mask_coords[node][1]]]
#     max_path = []
#     max_num_nodes = -float('inf')
#     for i in range(len(degree1_nodes)-1):
#         for j in range(i+1, len(degree1_nodes)):
#             if nx.has_path(Tree, degree1_nodes[i], degree1_nodes[j]):
#                 path = nx.shortest_path(Tree, degree1_nodes[i], degree1_nodes[j])
#                 num_nodes = len(path)
#                 if num_nodes > max_num_nodes:
#                     max_num_nodes = num_nodes
#                     max_path = path
#     max_edges = [(max_path[i], max_path[i+1]) for i in range(len(max_path)-1)]
#     return max_path,max_edges

def Get_Common_Skeleton(filament_clumps_id, max_path_i, max_path_used, skeleton_coords_record, \
                        start_coords, clump_coords_dict):
    common_clump_id = None
    common_sc_item = None
    subpart_id_used = None
    if len(max_path_used) != 1:
        common_path_id = -1
        break_logic = False
        for subpart_id_used in np.int32(np.linspace(len(max_path_used) - 2, 0, len(max_path_used) - 1)):
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
        array2_tuples = set(map(tuple, clump_coords_dict[common_clump_id][:, 1:]))
        common_elements_tuples = array1_tuples & array2_tuples
        common_skeleton_coords = np.array(list(common_elements_tuples))
        common_skeleton_coord_ids = []
        for common_skeleton_coord in common_skeleton_coords:
            index = np.where(np.all(skeleton_coords_i == common_skeleton_coord, axis=1))[0]
            common_skeleton_coord_ids.append(index[0])
        common_skeleton_coords = common_skeleton_coords[np.argsort(common_skeleton_coord_ids)]
        if len(common_skeleton_coords) > 3:
            third_len = len(common_skeleton_coords) // 3 + 1
            common_skeleton_coords = common_skeleton_coords[third_len:2 * third_len]
            common_sc_item = common_skeleton_coords - start_coords[1:]
    return common_clump_id, common_sc_item


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


def Trim_Skeleton_Coords_2D(skeleton_coords_2D, fil_mask, CalSubSK=False, SmallSkeleton=6):
    id_item = 1
    skeleton_coords_2D_len = 3
    fil_mask_shape = fil_mask.shape
    xres = fil_mask_shape[0]
    yres = fil_mask_shape[1]
    while id_item < skeleton_coords_2D_len - 1:
        x_center = skeleton_coords_2D[id_item][0]
        y_center = skeleton_coords_2D[id_item][1]
        x_arange = np.arange(max(0, x_center - 1), min(xres, x_center + 2))
        y_arange = np.arange(max(0, y_center - 1), min(yres, y_center + 2))
        [x, y] = np.meshgrid(x_arange, y_arange)
        neighbor_coords = np.column_stack([x.flat, y.flat])
        in_neighbor_number = 0
        for skeleton_coord in skeleton_coords_2D[id_item - 1:id_item + 3]:
            if tuple(skeleton_coord) in list(map(tuple, neighbor_coords)):
                in_neighbor_number += 1
        # 3*3
        remove_number = 0
        for id_item_T in range(id_item + 3, skeleton_coords_2D_len - 2):
            id_item_TT = id_item_T - remove_number
            if tuple(skeleton_coords_2D[id_item_TT]) in list(map(tuple, neighbor_coords)):
                remove_number += 1
                skeleton_coords_2D = np.r_[skeleton_coords_2D[:id_item_TT], skeleton_coords_2D[id_item_TT + 1:]]
        if in_neighbor_number == 4:
            skeleton_coords_2D = np.r_[skeleton_coords_2D[:id_item + 1], skeleton_coords_2D[id_item + 2:]]
        id_item += 1
        skeleton_coords_2D_len = len(skeleton_coords_2D)
    # 5*5
    id_item = 1
    while id_item < skeleton_coords_2D_len - 1:
        x_center = skeleton_coords_2D[id_item][0]
        y_center = skeleton_coords_2D[id_item][1]
        x_arange = np.arange(max(0, x_center - 2), min(xres, x_center + 3))
        y_arange = np.arange(max(0, y_center - 2), min(yres, y_center + 3))
        [x, y] = np.meshgrid(x_arange, y_arange)
        neighbor_coords = np.column_stack([x.flat, y.flat])
        remove_number = 0
        for id_item_T in range(id_item + 4, skeleton_coords_2D_len - 2):
            id_item_TT = id_item_T - remove_number
            if tuple(skeleton_coords_2D[id_item_TT]) in list(map(tuple, neighbor_coords)):
                remove_number += 1
                skeleton_coords_2D = np.r_[skeleton_coords_2D[:id_item_TT], skeleton_coords_2D[id_item_TT + 1:]]
        skeleton_coords_2D_len = len(skeleton_coords_2D)
        id_item += 1
    # skeleton_coords_2D 延长并取最长
    skeleton_mask = np.zeros_like(fil_mask)
    skeleton_mask[(skeleton_coords_2D[:, 0], skeleton_coords_2D[:, 1])] = 1
    props = measure.regionprops(measure.label(skeleton_mask))
    lengths = [prop.major_axis_length for prop in props]
    longest_skeleton_index = lengths.index(max(lengths))
    skeleton_coords_2D = props[longest_skeleton_index].coords
    for skeleton_index in range(0, len(props)):
        if len(props[skeleton_index].coords) > SmallSkeleton * 2 and skeleton_index != longest_skeleton_index:
            skeleton_coords_2D = np.r_[skeleton_coords_2D, props[skeleton_index].coords]
    if len(skeleton_coords_2D) > SmallSkeleton:
        small_sc = False
        if not CalSubSK:
            skeleton_coords_2D = Extend_Skeleton_Coords(skeleton_coords_2D, fil_mask)
    else:
        small_sc = True
        print('Small skeleton_coords_2D!')
    G_longest_skeleton, T_longest_skeleton = Graph_Infor(skeleton_coords_2D)
    max_path, max_edges = Get_Max_Path_Node(T_longest_skeleton)
    skeleton_coords_2D = skeleton_coords_2D[max_path]
    return skeleton_coords_2D, small_sc


def Search_Max_Path_And_Edges(paths_and_weights):
    max_path = []
    max_edges = []
    if len(paths_and_weights) != 0:
        max_weight = max([weight for path, weight in paths_and_weights])
        max_path = [path for path, weight in paths_and_weights if weight == max_weight][0]
        max_edges = [(max_path[i], max_path[i + 1]) for i in range(len(max_path) - 1)]
    return max_weight, max_path, max_edges


def Get_Max_Path_Intensity_Weighted(fil_mask, Tree, mask_coords, common_mask_coords_id=None):
    # Weight
    fil_mask_erosion = morphology.binary_erosion(fil_mask, morphology.disk(1))
    fil_mask_dilation = morphology.binary_dilation(fil_mask, morphology.disk(1))
    contour_data = fil_mask_dilation * ~fil_mask_erosion
    degree1_nodes = [node for node in Tree.nodes if Tree.degree(node) == 1 and \
                     contour_data[mask_coords[node][0], mask_coords[node][1]]]
    paths_and_weights = []
    if type(common_mask_coords_id) != type(None):
        for i in common_mask_coords_id:
            for j in range(len(degree1_nodes)):
                path = nx.shortest_path(Tree, i, degree1_nodes[j])
                path_weight = 0
                for k in range(len(path) - 1):
                    if Tree[path[k]][path[k + 1]]['weight'] != 0:
                        path_weight += 1 / Tree[path[k]][path[k + 1]]['weight']
                paths_and_weights.append((path, path_weight))
    else:
        for i in range(len(degree1_nodes) - 1):
            for j in range(i + 1, len(degree1_nodes)):
                #             if nx.has_path(Tree, degree1_nodes[i], degree1_nodes[j]):
                path = nx.shortest_path(Tree, degree1_nodes[i], degree1_nodes[j])
                path_weight = 0
                for k in range(len(path) - 1):
                    if Tree[path[k]][path[k + 1]]['weight'] != 0:
                        path_weight += 1 / Tree[path[k]][path[k + 1]]['weight']
                paths_and_weights.append((path, path_weight))
    max_weight, max_path, max_edges = Search_Max_Path_And_Edges(paths_and_weights)
    return max_path, max_edges


def Get_Max_Path_Intensity_Weighted_Fast(fil_mask, Tree, mask_coords, clump_numbers):
    min_weight = float('inf')
    edge_coords_1 = np.where(mask_coords[:, 0] == mask_coords[:, 0].min())[0].tolist()
    edge_coords_2 = np.where(mask_coords[:, 0] == mask_coords[:, 0].max())[0].tolist()
    edge_coords_3 = np.where(mask_coords[:, 1] == mask_coords[:, 1].min())[0].tolist()
    edge_coords_4 = np.where(mask_coords[:, 1] == mask_coords[:, 1].max())[0].tolist()
    edge_coords = list(set(edge_coords_1 + edge_coords_2 + edge_coords_3 + edge_coords_4))
    if clump_numbers < 1000:
        fil_mask_erosion = morphology.binary_erosion(fil_mask, morphology.disk(1))
        fil_mask_dilation = morphology.binary_dilation(fil_mask, morphology.disk(1))
        contour_data = fil_mask_dilation * ~fil_mask_erosion
        degree1_nodes = [node for node in Tree.nodes if Tree.degree(node) == 1 and \
                         contour_data[mask_coords[node][0], mask_coords[node][1]]]
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
                for i in range(len(path) - 1):
                    path_weight += 1 / Tree[path[i]][path[i + 1]]['weight']
                paths_and_weights.append((path, path_weight))
                max_weight_1, max_path_1, max_edges_1 = Search_Max_Path_And_Edges(paths_and_weights)
        paths_and_weights = []
        for target_node_id in degree1_nodes:
            path = nx.shortest_path(Tree, max_path_1[-1], target_node_id)
            in_path_logic = min_weight_node[0] in path
            for min_weight_node_id in range(1, len(min_weight_node)):
                in_path_logic = in_path_logic and (min_weight_node[min_weight_node_id] in path)
            if in_path_logic:
                path_weight = 0
                for i in range(len(path) - 1):
                    path_weight += 1 / Tree[path[i]][path[i + 1]]['weight']
                paths_and_weights.append((path, path_weight))
            for edge_coord in edge_coords:
                path = nx.shortest_path(Tree, target_node_id, edge_coord)
                path_weight = 0
                for i in range(len(path) - 1):
                    path_weight += 1 / Tree[path[i]][path[i + 1]]['weight']
                paths_and_weights.append((path, path_weight))
    else:
        paths_and_weights = []
        for i in range(len(edge_coords) - 1):
            for j in range(i + 1, len(edge_coords)):
                path = nx.shortest_path(Tree, edge_coords[i], edge_coords[j])
                path_weight = 0
                for k in range(len(path) - 1):
                    path_weight += 1 / Tree[path[k]][path[k + 1]]['weight']
                    paths_and_weights.append((path, path_weight))
    max_weight_2, max_path_2, max_edges_2 = Search_Max_Path_And_Edges(paths_and_weights)
    return max_path_2, max_edges_2


def Get_Single_Filament_Skeleton_Weighted(fil_image, fil_mask, clump_numbers, common_sc_item=None, SmallSkeleton=6):
    CalSubSK = type(common_sc_item) != type(None)
    fil_image_filtered = ndimage.uniform_filter(fil_image, size=3)
    regions_list = measure.regionprops(np.array(fil_mask, dtype='int'))
    mask_coords = regions_list[0].coords
    dist_matrix = Dists_Array(mask_coords, mask_coords)
    mask_coords_in_dm = np.where(np.logical_and(dist_matrix > 0.5, dist_matrix < 2))
    Graph_find_skeleton = nx.Graph()
    common_mask_coords_id = []
    for i, j in zip(mask_coords_in_dm[0], mask_coords_in_dm[1]):
        weight_ij = fil_image_filtered[mask_coords[i][0], mask_coords[i][1]] + \
                    fil_image_filtered[mask_coords[j][0], mask_coords[j][1]]
        if weight_ij == 0:
            weight_ij = 0
        else:
            weight_ij = dist_matrix[i, j] / weight_ij
        Graph_find_skeleton.add_edge(i, j, weight=weight_ij)
        if CalSubSK:
            if tuple(mask_coords[i]) in map(tuple, common_sc_item):
                common_mask_coords_id.append(i)
    if CalSubSK:
        common_mask_coords_id = list(set(common_mask_coords_id))
    else:
        common_mask_coords_id = None
    Tree = nx.minimum_spanning_tree(Graph_find_skeleton)
    if clump_numbers < 100 or CalSubSK:
        max_path, max_edges = Get_Max_Path_Intensity_Weighted(fil_mask, Tree, mask_coords, common_mask_coords_id)
    else:
        max_path, max_edges = Get_Max_Path_Intensity_Weighted_Fast(fil_mask, Tree, mask_coords, clump_numbers)
    skeleton_coords_2D = mask_coords[max_path]
    skeleton_coords_2D, small_sc = Trim_Skeleton_Coords_2D(skeleton_coords_2D, fil_mask, CalSubSK, SmallSkeleton)
    return skeleton_coords_2D, small_sc


def Cal_Lengh_Width_Ratio(CalSub, regions_data, related_ids_T, connected_ids_dict, clump_coords_dict, \
                          filament_mask_2D, filament_data, clumps_number, SkeletonType='Intensity'):
    samp_int = 1
    dictionary_cuts = defaultdict(list)
    fil_image = filament_data.sum(0)
    fil_mask = filament_mask_2D.astype(bool)
    regions = measure.regionprops(measure.label(fil_mask, connectivity=2))
    if len(regions) > 1:
        max_area = regions[0].area
        max_region = regions[0]
        for region in regions:
            coords = region.coords
            fil_image[coords[:, 0], coords[:, 1]] = 0
            fil_mask[coords[:, 0], coords[:, 1]] = False
            if region.area > max_area:
                max_area = region.area
                max_region = region
        fil_image[max_region.coords[:, 0], max_region.coords[:, 1]] = \
            filament_data.sum(0)[max_region.coords[:, 0], max_region.coords[:, 1]]
        fil_mask[max_region.coords[:, 0], max_region.coords[:, 1]] = True
    if SkeletonType == 'Morphology':
        skeleton_coords_2D, filament_skeleton, all_skeleton_coords = Get_Single_Filament_Skeleton(fil_mask)
    elif SkeletonType == 'Intensity':
        all_skeleton_coords = None
        skeleton_coords_2D, small_sc = Get_Single_Filament_Skeleton_Weighted(fil_image, fil_mask, clumps_number)
    else:
        print('Please choose the skeleton_type between Morphology and Intensity')

    if not small_sc:
        dictionary_cuts = Cal_Dictionary_Cuts(samp_int, CalSub, regions_data, related_ids_T, connected_ids_dict,
                                              clump_coords_dict, \
                                              skeleton_coords_2D, fil_image, fil_mask, dictionary_cuts)
        start_coords = np.array(dictionary_cuts['plot_cuts'])[:, 0]
        end_coords = np.array(dictionary_cuts['plot_cuts'])[:, 1]
        width_dists = np.diagonal(Dists_Array(start_coords, end_coords))
        #         width_dist_mean = np.mean(width_dists)
        width_dist_mean = np.median(width_dists[1:-1])
        lengh_dist = len(dictionary_cuts['points'][0]) * samp_int
        lengh_width_ratio = lengh_dist / width_dist_mean
    else:
        lengh_dist = 1
        lengh_width_ratio = 1
    return dictionary_cuts, lengh_dist, lengh_width_ratio, skeleton_coords_2D, all_skeleton_coords


def Get_LBV_Table(coords):
    x_min = np.array(coords[0]).min()
    x_max = np.array(coords[0]).max()
    y_min = np.array(coords[1]).min()
    y_max = np.array(coords[1]).max()
    z_min = np.array(coords[2]).min()
    z_max = np.array(coords[2]).max()
    v_delta = x_max - x_min + 1
    box_data = np.zeros([y_max - y_min + 3, z_max - z_min + 3])
    box_data[coords[1] - y_min + 1, coords[2] - z_min + 1] = 1
    box_label = measure.label(box_data)
    box_region = measure.regionprops(box_label)
    lb_area = box_region[0].area
    coords_range = [x_min, x_max, y_min, y_max, z_min, z_max]
    return coords_range, lb_area, v_delta, box_data


def Cal_Velocity_Map(filament_item, skeleton_coords_2D, data_wcs_item):
    filament_item_shape = filament_item.shape
    l_min, l_max = 0, filament_item_shape[2] - 1
    b_min, b_max = 0, filament_item_shape[1] - 1
    v_min, v_max = 0, filament_item_shape[0] - 1
    if data_wcs_item.naxis == 4:
        lbv_start = data_wcs_item.all_pix2world(l_min, b_min, v_min, 0, 0)
        lbv_end = data_wcs_item.all_pix2world(l_max, b_max, v_max, 0, 0)
    elif data_wcs_item.naxis == 3:
        lbv_start = data_wcs_item.all_pix2world(l_min, b_min, v_min, 0)
        lbv_end = data_wcs_item.all_pix2world(l_max, b_max, v_max, 0)
    lbv_item_start = [np.around(lbv_start[0], 2), np.around(lbv_start[1], 2), np.around(lbv_start[2] / 1000, 2)]
    lbv_item_end = [np.around(lbv_end[0], 2), np.around(lbv_end[1], 2), np.around(lbv_end[2] / 1000, 2)]
    velocity_range = np.linspace(lbv_item_start[2], lbv_item_end[2], filament_item_shape[0])
    velocity_map_item = np.tensordot(filament_item, velocity_range, axes=((0,), (0,))) / np.sum(filament_item, axis=0)
    velocity_map_item = np.nan_to_num(velocity_map_item)
    v_skeleton_com_i = velocity_map_item[(skeleton_coords_2D[:, 0], skeleton_coords_2D[:, 1])]
    v_skeleton_com_delta = np.around(v_skeleton_com_i.max() - v_skeleton_com_i.min(), 3)
    return lbv_item_start, lbv_item_end, velocity_map_item, v_skeleton_com_delta



