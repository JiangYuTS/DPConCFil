import time
import numpy as np
from tqdm import tqdm
import copy


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


def Estimate_Direction_Consistency(centers,angles,connected_ids_dict,center_id_i,center_id_j,\
                                   dist_con_items,related_ids,TolAngle,TolDistance):
    center_i = centers[center_id_i]
    center_j = centers[center_id_j]
    # con_center_ids_i = connected_ids_dict[center_id_i]
    con_center_ids_j = connected_ids_dict[center_id_j]
    delta_angle_1 = Cal_Delta_Angle(center_i, center_j, angles[center_id_i])
    delta_angle_2 = Cal_Delta_Angle(center_i, center_j, angles[center_id_j])
    angle_logic = False
    if delta_angle_1 < TolAngle and delta_angle_2 < TolAngle:
        angle_logic = True
        related_ids[center_id_i].append(center_id_j)
        # for con_center_id_j in con_center_ids_j:
        #     delta_vbl_1 = np.abs(centers[center_id_i] - centers[con_center_id_j])
            # delta_vbl_2 = np.abs(centers[center_id_j] - centers[con_center_id_j])
            # if  np.argmax(delta_vbl_1) == 0 :#or np.argmax(delta_vbl_2) == 0:
            #     con_center_ids_ij.remove(con_center_id_j)
        dist_con_items = Estimate_Position_Consistency(centers,center_id_i,center_id_j,con_center_ids_j,dist_con_items,TolDistance)
    return dist_con_items,related_ids,angle_logic
    

def Estimate_Position_Consistency(centers,center_id_i,center_id_j,con_center_ids_ij,dist_con_items,TolDistance,WeightFactor=[1,1,1]):
    for con_center_id_ij in con_center_ids_ij:
        if center_id_i != center_id_j and con_center_id_ij != center_id_i and con_center_id_ij != center_id_j:
            dist_ab_c = Dist_Line_Point(centers[center_id_i], centers[center_id_j], centers[con_center_id_ij])
            if dist_ab_c < TolDistance:
                pline_1 = [centers[center_id_i][2], centers[center_id_i][1]]
                pline_2 = [centers[center_id_j][2], centers[center_id_j][1]]
                pitem = [centers[con_center_id_ij][2], centers[con_center_id_ij][1]]
                pedal_point = Pedal_Point(pline_1, pline_2, pitem)
                between_item_logic = Between_Item_Logic(pline_1, pline_2, pedal_point)
                if between_item_logic and con_center_id_ij not in dist_con_items[center_id_i][0]:
                    dist_con_items[center_id_i][0] += [con_center_id_ij]
                elif con_center_id_ij not in dist_con_items[center_id_i][1]:
                    delta_vbl_1 = np.abs(centers[center_id_i] - centers[con_center_id_ij])*WeightFactor
                    if  np.argmax(delta_vbl_1) != 0 :
                        dist_con_items[center_id_i][1] += [con_center_id_ij]
    return dist_con_items


def Estimate_Position_Consistency_2(centers,center_id_i,center_id_j,line_center_id,con_center_ids_ij,dist_con_items,\
                                    TolDistance,WeightFactor=[1,1,1]):
    for con_center_id_ij in con_center_ids_ij:
        if con_center_id_ij != center_id_i and con_center_id_ij != center_id_j and con_center_id_ij != line_center_id:
            dist_ab_c = Dist_Line_Point(centers[center_id_i], centers[center_id_j], centers[con_center_id_ij])
            if dist_ab_c < TolDistance:
                pline_1 = [centers[center_id_i][2], centers[center_id_i][1]]
                pline_2 = [centers[line_center_id][2], centers[line_center_id][1]]
                pitem = [centers[con_center_id_ij][2], centers[con_center_id_ij][1]]
                pedal_point = Pedal_Point(pline_1, pline_2, pitem)
                between_item_logic_1 = Between_Item_Logic(pline_1, pline_2, pedal_point)
                pline_1 = [centers[center_id_j][2], centers[center_id_j][1]]
                pline_2 = [centers[line_center_id][2], centers[line_center_id][1]]
                pitem = [centers[con_center_id_ij][2], centers[con_center_id_ij][1]]
                pedal_point = Pedal_Point(pline_1, pline_2, pitem)
                between_item_logic_2 = Between_Item_Logic(pline_1, pline_2, pedal_point)
                if between_item_logic_1 or between_item_logic_2:
                    continue
                elif con_center_id_ij not in dist_con_items[center_id_j][1]:
                    delta_vbl_1 = np.abs(centers[center_id_j] - centers[con_center_id_ij])*WeightFactor
                    if  np.argmax(delta_vbl_1) != 0 :
                        dist_con_items[center_id_j][1] += [con_center_id_ij]
    return dist_con_items


def Get_Related_Ids_RR(regions_data,centers,rr_centers_id,connected_ids_dict_lists,edges,angles,TolAngle,TolDistance):
    related_ids = {}
    line_items = {}
    between_items = {}
    dist_con_items = {}
    connected_ids_dict_1 = connected_ids_dict_lists[0]
    connected_ids_dict_2 = connected_ids_dict_lists[1]
    connected_ids_dict_3 = connected_ids_dict_lists[2]
    for center_id in rr_centers_id:
        related_ids[center_id] = []
        dist_con_items[center_id] = [[], []]
    for center_id_i in rr_centers_id:
        if edges[center_id_i] == 0:  # or edges[center_id_i] == 1
            con_center_ids_i = connected_ids_dict_1[center_id_i]
            for center_id_j in con_center_ids_i:
                if edges[center_id_j] == 0 :
                    dist_con_items,related_ids,angle_logic = Estimate_Direction_Consistency(centers,angles,connected_ids_dict_1,center_id_i,\
                                                                                            center_id_j,dist_con_items,related_ids,\
                                                                                            TolAngle,TolDistance)
                    if angle_logic:
                        con_center_ids_i = connected_ids_dict_3[center_id_i]
                        con_center_ids_j = connected_ids_dict_3[center_id_j]
                        for between_id in dist_con_items[center_id_i][0]:
                            if between_id not in con_center_ids_i:
                                dist_con_items[center_id_i][0].remove(between_id)
                        line_center_ids = list(set(con_center_ids_j) & set(dist_con_items[center_id_i][1]))
                        for line_center_id in line_center_ids:
                            con_center_ids_j_2 = connected_ids_dict_3[line_center_id]
                            dist_con_items = Estimate_Position_Consistency(centers,center_id_i,center_id_j,\
                                                           con_center_ids_j_2,dist_con_items,TolDistance)
                            # dist_con_items = Estimate_Position_Consistency(centers,center_id_j,line_center_id,\
                            #                                con_center_ids_j_2,dist_con_items,TolDistance)
                            dist_con_items = Estimate_Position_Consistency_2(centers,center_id_i,center_id_j,line_center_id,\
                                                           con_center_ids_j_2,dist_con_items,TolDistance)

                            for con_center_id_j_2 in con_center_ids_j_2:
                                dist_con_items,related_ids,angle_logic = Estimate_Direction_Consistency(centers,angles,\
                                                                            connected_ids_dict_3,center_id_j,con_center_id_j_2,\
                                                                            dist_con_items,related_ids,TolAngle,TolDistance)
    related_ids, between_items = Add_Items_To_Related_Ids(related_ids, dist_con_items, between_items, 0)
    related_ids, line_items = Add_Items_To_Related_Ids(related_ids, dist_con_items, line_items, 1)
    return related_ids, dist_con_items


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
    return related_ids_2

    
def Add_Isolated_Con_Neighbor(related_ids,con_ids_dict_enhanced):
    related_ids_add_enhanced = copy.deepcopy(related_ids)
    for key in related_ids.keys():
        for clump_id in related_ids[key]:
            connected_ids_enhanced = con_ids_dict_enhanced[clump_id]
            for connected_id_enhanced in connected_ids_enhanced:
                if connected_id_enhanced not in related_ids[key]:
                    add_flag = False
                    for connected_id_enhanced_2 in con_ids_dict_enhanced[connected_id_enhanced]:
                        if connected_id_enhanced_2 not in related_ids[key]:
                            add_flag = False
                            break
                        else:
                            add_flag = True
                    if add_flag:
                        related_ids_add_enhanced[key] += [connected_id_enhanced]
    for i in range(len(related_ids_add_enhanced)):
        len_0 = len(related_ids_add_enhanced)
        related_ids_add_enhanced = Update_Related_Ids(related_ids_add_enhanced)
        len_1 = len(related_ids_add_enhanced)
        if len_0 == len_1:
            break
    return related_ids_add_enhanced



