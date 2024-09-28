import time
import numpy as np
from tqdm import tqdm


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


def Dist_Line_Point(point_a, point_b, point_c):
    #     vector_ab_x = point_b[0] - point_a[0]
    vector_ab_y = point_b[1] - point_a[1]
    vector_ab_z = point_b[2] - point_a[2]
    #     vector_ac_x = point_c[0] - point_a[0]
    vector_ac_y = point_c[1] - point_a[1]
    vector_ac_z = point_c[2] - point_a[2]
    #     element = vector_ab_x * vector_ac_x + vector_ab_y * vector_ac_y + vector_ab_z * vector_ac_z
    #     denominator_0 = np.sqrt(vector_ab_x * vector_ab_x + vector_ab_y * vector_ab_y + vector_ab_z * vector_ab_z)
    #     denominator_1 = np.sqrt(vector_ac_x * vector_ac_x + vector_ac_y * vector_ac_y + vector_ac_z * vector_ac_z)
    element = vector_ab_y * vector_ac_y + vector_ab_z * vector_ac_z
    denominator_0 = np.sqrt(vector_ab_y * vector_ab_y + vector_ab_z * vector_ab_z)
    denominator_1 = np.sqrt(vector_ac_y * vector_ac_y + vector_ac_z * vector_ac_z)

    theta = np.arccos(element / (denominator_0 * denominator_1))
    dist_ab_c = np.abs(denominator_1 * np.sin(theta))
    return dist_ab_c


def Get_Line_Logic(centers, center_id, nearest_centers_id, nearest_center_id, regions_data, AllowItems):
    line_data_record = []
    line_zero_record = []
    point_a = np.around(centers[center_id])
    point_b = np.around(centers[nearest_center_id])
    line_coords_3D = Get_Line_Coords_3D(point_a, point_b)
    for line_coord_3D in line_coords_3D:
        line_data = regions_data[:, line_coord_3D[1], line_coord_3D[2]]
        #         print('line_data:',line_data)
        nearest_centers_id_T = nearest_centers_id.copy()
        nearest_centers_id_T.append(center_id)
        line_data_intersection = set(line_data - 1) & set(nearest_centers_id_T)
        if len(line_data_intersection) == 0:
            line_zero_record.append(line_coord_3D)
        else:
            if center_id in line_data_intersection or nearest_center_id in line_data_intersection:
                line_data_record.append(center_id)
                line_data_record.append(nearest_center_id)
            else:
                line_data_record += list(line_data_intersection)
    line_data_set = list(set(line_data_record))
    line_logic = len(line_zero_record) < 1 and len(line_data_set) < AllowItems + 3
    return line_logic, line_data_set


def Cal_Delta_Angle(center_i, center_j, angle):
    l_delta = center_i[2] - center_j[2]
    b_delta = center_i[1] - center_j[1]
    center_angle = 90 if l_delta == 0 else np.around(np.degrees(np.arctan(b_delta / l_delta)), 2)
    cos_alpha = np.cos(np.radians(center_angle - angle))
    alpha = np.degrees(np.arccos(cos_alpha))
    delta_angle = alpha if alpha <= 90 else 180 - alpha
    return delta_angle


def Pedal_Point(pline_1, pline_2, pitem):
    if pline_1[0] != pline_2[0]:
        k, b = np.linalg.solve([[pline_1[0], 1], [pline_2[0], 1]], [pline_1[1], pline_2[1]])
        x = np.divide(((pline_2[0] - pline_1[0]) * pitem[0] + (pline_2[1] - pline_1[1]) * pitem[1] - \
                       b * (pline_2[1] - pline_1[1])), (pline_2[0] - pline_1[0] + k * (pline_2[1] - pline_1[1])))
        y = k * x + b
    else:
        x = pline_1[0]
        y = pitem[1]
    pedal_point = np.array([x, y])
    return pedal_point


def Between_Item_Logic(point_1, point_2, pedal_point):
    dist_line_1 = ((np.array(point_1) - point_2) ** 2).sum() ** (1 / 2)
    dist_line_2 = ((point_1 - pedal_point) ** 2).sum() ** (1 / 2)
    dist_line_3 = ((point_2 - pedal_point) ** 2).sum() ** (1 / 2)
    if dist_line_2 > dist_line_1 or dist_line_3 > dist_line_1:
        return False
    else:
        return True


def Get_Crossing_Items(regions_data, centers, center_id, nearest_centers_id, nearest_center_id, \
                       crossing_items, line_data_set, TolDistance, AllowItems):
    line_data_dict = {}
    min_center_id = np.min([center_id, nearest_center_id])
    for nearest_center_id_t in nearest_centers_id:
        if nearest_center_id_t != center_id and nearest_center_id_t != nearest_center_id:
            dist_ab_c = Dist_Line_Point(centers[center_id], centers[nearest_center_id], centers[nearest_center_id_t])
            if dist_ab_c < TolDistance:
                line_logic_0, line_data_0 = Get_Line_Logic(centers, center_id, \
                                                           nearest_centers_id, nearest_center_id_t, regions_data,
                                                           AllowItems)
                line_logic_1, line_data_1 = Get_Line_Logic(centers, nearest_center_id, \
                                                           nearest_centers_id, nearest_center_id_t, regions_data,
                                                           AllowItems)
                if line_logic_0 and line_logic_1:
                    line_data_dict[nearest_center_id_t] = list(set(line_data_0 + line_data_1))
                    pline_1 = [centers[center_id][2], centers[center_id][1]]
                    pline_2 = [centers[nearest_center_id][2], centers[nearest_center_id][1]]
                    pitem = [centers[nearest_center_id_t][2], centers[nearest_center_id_t][1]]
                    pedal_point = Pedal_Point(pline_1, pline_2, pitem)
                    between_item_logic = Between_Item_Logic(pline_1, pline_2, pedal_point)
                    if between_item_logic and nearest_center_id_t not in crossing_items[min_center_id][0]:
                        crossing_items[min_center_id][0] += [nearest_center_id_t]
                    elif nearest_center_id_t not in crossing_items[min_center_id][1]:
                        crossing_items[min_center_id][1] += [nearest_center_id_t]
    # 所有在line_data_set内的都需要满足TolDistance才允许保留
    remove_flag = 0
    if len(line_data_set) - len(crossing_items[min_center_id][0]) > 2:
        remove_flag = 1
        nearest_center_id_temps = crossing_items[min_center_id][0] + crossing_items[min_center_id][1]
        for nearest_center_id_t in nearest_center_id_temps:
            if nearest_center_id_t in crossing_items[min_center_id][0]:
                crossing_items[min_center_id][0].remove(nearest_center_id_t)
            else:
                crossing_items[min_center_id][1].remove(nearest_center_id_t)
    # items中的item对应的line_data_i需要在items中
    items = list(
        set([center_id] + [nearest_center_id] + crossing_items[min_center_id][0] + crossing_items[min_center_id][1]))
    for nearest_center_id_t in line_data_dict.keys():
        for line_data_i in line_data_dict[nearest_center_id_t]:
            if line_data_i not in items:
                if nearest_center_id_t in crossing_items[min_center_id][0]:
                    crossing_items[min_center_id][0].remove(nearest_center_id_t)
    #                 elif nearest_center_id_t in crossing_items[min_center_id][1]:
    #                     crossing_items[min_center_id][1].remove(nearest_center_id_t)
    return crossing_items, remove_flag


def Add_Items_To_Related_Ids(related_ids, crossing_items, new_items_dict, item_flag):
    #     item_flag=0:between,item_flag=1,line
    keys = list(crossing_items.keys())
    for key in keys:
        if len(crossing_items[key][item_flag]) != 0:
            valid_item = []
            items_set = set(crossing_items[key][item_flag])
            for item in items_set:
                if item not in related_ids[key]:
                    valid_item.append(item)
            if len(valid_item) != 0:
                new_items_dict[key] = list(set(valid_item))
                related_ids[key] += valid_item
    return related_ids, new_items_dict


def Get_Related_Ids_RR(regions_data, centers, rr_centers_id, connected_ids_dict, edges, angles, TolAngle, TolDistance,
                       AllowItems):
    related_ids = {}
    line_items = {}
    between_items = {}
    crossing_items = {}
    for center_id in rr_centers_id:
        related_ids[center_id] = []
        crossing_items[center_id] = [[], []]
    for center_id in rr_centers_id:
        if edges[center_id] == 0:  # or edges[center_id] == 1
            center_i = centers[center_id]
            nearest_centers_id = connected_ids_dict[center_id]
            for nearest_center_id in nearest_centers_id:
                if edges[nearest_center_id] == 0:
                    center_j = centers[nearest_center_id]
                    delta_angle_1 = Cal_Delta_Angle(center_i, center_j, angles[center_id])
                    delta_angle_2 = Cal_Delta_Angle(center_i, center_j, angles[nearest_center_id])
                    line_logic, line_data_set = Get_Line_Logic(centers, center_id, nearest_centers_id,
                                                               nearest_center_id, regions_data, AllowItems)
                    if delta_angle_1 < TolAngle and delta_angle_2 < TolAngle and line_logic and center_id < nearest_center_id:
                        related_ids[center_id].append(nearest_center_id)
                        nearest_centers_id_j = connected_ids_dict[nearest_center_id]
                        nearest_centers_id_ij = list(set(nearest_centers_id + nearest_centers_id_j))
                        crossing_items, remove_flag = Get_Crossing_Items(regions_data, centers, center_id, \
                                                                         nearest_centers_id_ij, nearest_center_id,
                                                                         crossing_items, line_data_set, TolDistance,
                                                                         AllowItems)
                        if remove_flag:
                            related_ids[center_id].remove(nearest_center_id)
                        else:
                            line_centers_id = list(set(crossing_items[center_id][1].copy()))
                            for line_center_id in line_centers_id:
                                nearest_centers_id_j = connected_ids_dict[line_center_id]
                                nearest_centers_id_ij = nearest_centers_id_j + [line_center_id]
                                nearest_centers_id_ij = list(set(nearest_centers_id_ij))
                                crossing_items, remove_flag = Get_Crossing_Items(regions_data, centers, center_id, \
                                                                                 nearest_centers_id_ij,
                                                                                 nearest_center_id, crossing_items,
                                                                                 line_data_set, TolDistance, AllowItems)
                                if remove_flag:
                                    related_ids[center_id].remove(nearest_center_id)
                                    break
    related_ids, between_items = Add_Items_To_Related_Ids(related_ids, crossing_items, between_items, 0)
    related_ids, line_items = Add_Items_To_Related_Ids(related_ids, crossing_items, line_items, 1)
    return related_ids, between_items, line_items


def Update_Related_Ids(related_ids):
    related_ids_1 = {}
    related_ids_2 = {}
    keys = np.array(list(related_ids.keys()))
    for key in keys:
        related_ids_1[key] = []
    key_used = []
    loop_i = 1
    for key_0 in keys:
        for key_1 in keys:
            for key_1_item in related_ids[key_1]:
                if (key_0 in related_ids[key_1] or key_1 in related_ids[key_0] or key_1_item in related_ids[key_0]) \
                        and key_1 not in key_used:
                    related_ids_1[key_0] += related_ids[key_0]
                    related_ids_1[key_0] += related_ids[key_1]
                    related_ids_1[key_0].append(key_1)
                    key_used.append(key_1)
        loop_i += 1
    for key in related_ids_1.keys():
        if len(related_ids_1[key]) != 0:
            related_ids_2[key] = list(set(related_ids_1[key]))
    #             related_ids_2[key].remove(key)
    return related_ids_2


