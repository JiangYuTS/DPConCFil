import numpy as np
import copy


def Dist_Line_Point(point_a, point_b, point_c):
    """
    Calculate the perpendicular distance from point_c to the line defined by point_a and point_b.
    This function appears to be focused on only the y and z dimensions (ignoring x).
    
    Parameters:
        point_a: First point defining the line
        point_b: Second point defining the line
        point_c: The point to calculate distance from the line
        
    Returns:
        dist_ab_c: The perpendicular distance from point_c to line AB
    """
    # Calculate vector components from point A to B
    vector_ab_y = point_b[1] - point_a[1]
    vector_ab_z = point_b[2] - point_a[2]
    
    # Calculate vector components from point A to C
    vector_ac_y = point_c[1] - point_a[1]
    vector_ac_z = point_c[2] - point_a[2]
    
    # Calculate dot product of the two vectors (only considering y and z components)
    element = vector_ab_y * vector_ac_y + vector_ab_z * vector_ac_z
    
    # Calculate magnitude of vectors
    denominator_0 = np.sqrt(vector_ab_y * vector_ab_y + vector_ab_z * vector_ab_z)
    denominator_1 = np.sqrt(vector_ac_y * vector_ac_y + vector_ac_z * vector_ac_z)
    
    # Calculate the angle between the two vectors using the dot product formula
    theta = np.arccos(element / (denominator_0 * denominator_1))
    
    # Calculate the perpendicular distance using trigonometry
    dist_ab_c = np.abs(denominator_1 * np.sin(theta))
    return dist_ab_c


def Cal_Delta_Angle(center_i, center_j, angle):
    """
    Calculate the angular difference between the given angle and the angle formed by two centers.
    
    Parameters:
        center_i: Coordinates of the first center
        center_j: Coordinates of the second center
        angle: Reference angle in degrees
        
    Returns:
        delta_angle: The angular difference in degrees (0-90°)
    """
    # Calculate the difference in longitude and latitude between centers
    l_delta = center_i[2] - center_j[2]  # Longitude difference
    b_delta = center_i[1] - center_j[1]  # Latitude difference
    
    # Calculate the angle of the line connecting the two centers
    # Default to 90° if l_delta is zero to avoid division by zero
    center_angle = 90 if l_delta == 0 else np.around(np.degrees(np.arctan(b_delta / l_delta)), 2)
    
    # Calculate the cosine of the angular difference
    cos_alpha = np.cos(np.radians(center_angle - angle))
    
    # Convert back to degrees
    alpha = np.degrees(np.arccos(cos_alpha))
    
    # Ensure the angle is in the range [0, 90] degrees
    delta_angle = alpha if alpha <= 90 else 180 - alpha
    return delta_angle


def Pedal_Point(pline_1, pline_2, pitem):
    """
    Calculate the pedal point (perpendicular projection) of a point onto a line.
    
    Parameters:
        pline_1: First point defining the line (longitude, latitude)
        pline_2: Second point defining the line (longitude, latitude)
        pitem: Point to project onto the line (longitude, latitude)
        
    Returns:
        pedal_point: The coordinates of the projection point on the line
    """
    # Check if the line is not vertical (points have different x-coordinates)
    if pline_1[0] != pline_2[0]:
        # Calculate line equation parameters (y = kx + b) by solving linear system
        k, b = np.linalg.solve([[pline_1[0], 1], [pline_2[0], 1]], [pline_1[1], pline_2[1]])
        
        # Calculate x-coordinate of the pedal point using the formula for perpendicular projection
        x = np.divide(((pline_2[0] - pline_1[0]) * pitem[0] + (pline_2[1] - pline_1[1]) * pitem[1] - \
                       b * (pline_2[1] - pline_1[1])), (pline_2[0] - pline_1[0] + k * (pline_2[1] - pline_1[1])))
        
        # Calculate y-coordinate using the line equation
        y = k * x + b
    else:
        # If the line is vertical, the pedal point has the same x-coordinate as the line
        # and the same y-coordinate as the point
        x = pline_1[0]
        y = pitem[1]
    
    # Create array with the pedal point coordinates
    pedal_point = np.array([x, y])
    return pedal_point


def Between_Item_Logic(point_1, point_2, pedal_point):
    """
    Check if a pedal point is located between two points on a line segment.
    
    Parameters:
        point_1: First endpoint of the line segment
        point_2: Second endpoint of the line segment
        pedal_point: Point to check if it's between the endpoints
        
    Returns:
        Boolean: True if the pedal point is between the endpoints, False otherwise
    """
    # Calculate distances between all points
    dist_line_1 = ((np.array(point_1) - point_2) ** 2).sum() ** (1 / 2)  # Distance between endpoints
    dist_line_2 = ((point_1 - pedal_point) ** 2).sum() ** (1 / 2)  # Distance from point_1 to pedal_point
    dist_line_3 = ((point_2 - pedal_point) ** 2).sum() ** (1 / 2)  # Distance from point_2 to pedal_point
    
    # If either distance from endpoints to pedal point is greater than the total segment length,
    # then the pedal point lies outside the line segment
    if dist_line_2 > dist_line_1 or dist_line_3 > dist_line_1:
        return False
    else:
        return True
        

def Add_Items_To_Related_Ids(related_ids, crossing_items, new_items_dict, item_flag):
    """
    Add new items to related_ids dictionary based on crossing items.
    
    Parameters:
        related_ids: Dictionary mapping center IDs to lists of related center IDs
        crossing_items: Dictionary with items to be added
        new_items_dict: Dictionary to store newly added items
        item_flag: Flag indicating which type of items to add (0: between items, 1: line items)
        
    Returns:
        related_ids: Updated dictionary with newly added items
        new_items_dict: Dictionary containing only the newly added items
    """
    # Get all keys from the crossing_items dictionary
    keys = list(crossing_items.keys())
    
    for key in keys:
        # Check if there are items to add for the current key
        if len(crossing_items[key][item_flag]) != 0:
            valid_item = []
            # Create a set of items to avoid duplicates
            items_set = set(crossing_items[key][item_flag])
            
            # Find items not already in related_ids
            for item in items_set:
                if item not in related_ids[key]:
                    valid_item.append(item)
            
            # Add valid items to both dictionaries
            if len(valid_item) != 0:
                new_items_dict[key] = list(set(valid_item))
                related_ids[key] += valid_item
    
    return related_ids, new_items_dict


def Estimate_Direction_Consistency(centers, angles, connected_ids_dict, center_id_i, center_id_j, 
                                   dist_con_items, related_ids, TolAngle, TolDistance):
    """
    Estimate if two centers have consistent directions and update related structures.
    
    Parameters:
        centers: Array containing coordinates of all centers
        angles: Array containing angles of all centers
        connected_ids_dict: Dictionary of connected center IDs
        center_id_i: ID of the first center
        center_id_j: ID of the second center
        dist_con_items: Dictionary tracking connected items by distance
        related_ids: Dictionary mapping center IDs to related center IDs
        TolAngle: Tolerance angle threshold for direction consistency
        TolDistance: Tolerance distance threshold for position consistency
        
    Returns:
        dist_con_items: Updated distance connection items
        related_ids: Updated related IDs dictionary
        angle_logic: Boolean indicating if the centers have consistent directions
    """
    # Get center coordinates
    center_i = centers[center_id_i]
    center_j = centers[center_id_j]
    
    # Get connected center IDs for center_j
    con_center_ids_j = connected_ids_dict[center_id_j]
    
    # Calculate angular differences between the centers and their angles
    delta_angle_1 = Cal_Delta_Angle(center_i, center_j, angles[center_id_i])
    delta_angle_2 = Cal_Delta_Angle(center_i, center_j, angles[center_id_j])
    
    angle_logic = False
    # Check if both angular differences are within tolerance
    if delta_angle_1 < TolAngle and delta_angle_2 < TolAngle:
        angle_logic = True
        # Add center_j to related IDs of center_i
        related_ids[center_id_i].append(center_id_j)
        
        # Estimate position consistency for the connected centers
        dist_con_items = Estimate_Position_Consistency(centers, center_id_i, center_id_j, 
                                                      con_center_ids_j, dist_con_items, TolDistance)
    
    return dist_con_items, related_ids, angle_logic
    

def Estimate_Position_Consistency(centers, center_id_i, center_id_j, con_center_ids_ij, 
                                 dist_con_items, TolDistance, WeightFactor=[1,1,1]):
    """
    Estimate if positions of centers are consistent based on distance criteria.
    
    Parameters:
        centers: Array containing coordinates of all centers
        center_id_i: ID of the first center
        center_id_j: ID of the second center
        con_center_ids_ij: IDs of centers connected to both center_i and center_j
        dist_con_items: Dictionary tracking connected items by distance
        TolDistance: Tolerance threshold for distance
        WeightFactor: Weight factors for different dimensions [x,y,z]
        
    Returns:
        dist_con_items: Updated distance connection items
    """
    # Check each connected center
    for con_center_id_ij in con_center_ids_ij:
        # Skip if the connected center is one of the endpoints
        if center_id_i != center_id_j and con_center_id_ij != center_id_i and con_center_id_ij != center_id_j:
            # Calculate the perpendicular distance from the connected center to the line
            dist_ab_c = Dist_Line_Point(centers[center_id_i], centers[center_id_j], centers[con_center_id_ij])
            
            # If distance is within tolerance, continue checking
            if dist_ab_c < TolDistance:
                # Prepare points for pedal point calculation (longitude, latitude)
                pline_1 = [centers[center_id_i][2], centers[center_id_i][1]]
                pline_2 = [centers[center_id_j][2], centers[center_id_j][1]]
                pitem = [centers[con_center_id_ij][2], centers[con_center_id_ij][1]]
                
                # Calculate pedal point (projection of connected center onto the line)
                pedal_point = Pedal_Point(pline_1, pline_2, pitem)
                
                # Check if the pedal point is between the endpoints
                between_item_logic = Between_Item_Logic(pline_1, pline_2, pedal_point)
                
                # Add to between items if pedal point is between endpoints
                if between_item_logic and con_center_id_ij not in dist_con_items[center_id_i][0]:
                    dist_con_items[center_id_i][0] += [con_center_id_ij]
                # Otherwise, add to line items with additional checks
                elif con_center_id_ij not in dist_con_items[center_id_i][1]:
                    # Calculate weighted difference between centers
                    delta_vbl_1 = np.abs(centers[center_id_i] - centers[con_center_id_ij]) * WeightFactor
                    # Check if the maximum difference is not in the x dimension
                    if np.argmax(delta_vbl_1) != 0:
                        dist_con_items[center_id_i][1] += [con_center_id_ij]
    
    return dist_con_items


def Estimate_Position_Consistency_2(centers, center_id_i, center_id_j, line_center_id, con_center_ids_ij, 
                                   dist_con_items, TolDistance, WeightFactor=[1,1,1]):
    """
    Extended version of position consistency estimation for three centers forming a triangle.
    
    Parameters:
        centers: Array containing coordinates of all centers
        center_id_i: ID of the first center
        center_id_j: ID of the second center
        line_center_id: ID of the third center forming a line with others
        con_center_ids_ij: IDs of centers connected to these centers
        dist_con_items: Dictionary tracking connected items by distance
        TolDistance: Tolerance threshold for distance
        WeightFactor: Weight factors for different dimensions [x,y,z]
        
    Returns:
        dist_con_items: Updated distance connection items
    """
    # Check each connected center
    for con_center_id_ij in con_center_ids_ij:
        # Skip if the connected center is one of the triangle vertices
        if (con_center_id_ij != center_id_i and con_center_id_ij != center_id_j 
            and con_center_id_ij != line_center_id):
            # Calculate perpendicular distance from connected center to the line
            dist_ab_c = Dist_Line_Point(centers[center_id_i], centers[center_id_j], centers[con_center_id_ij])
            
            # If distance is within tolerance, continue checking
            if dist_ab_c < TolDistance:
                # Check if the connected center is between center_i and line_center_id
                pline_1 = [centers[center_id_i][2], centers[center_id_i][1]]
                pline_2 = [centers[line_center_id][2], centers[line_center_id][1]]
                pitem = [centers[con_center_id_ij][2], centers[con_center_id_ij][1]]
                pedal_point = Pedal_Point(pline_1, pline_2, pitem)
                between_item_logic_1 = Between_Item_Logic(pline_1, pline_2, pedal_point)
                
                # Check if the connected center is between center_j and line_center_id
                pline_1 = [centers[center_id_j][2], centers[center_id_j][1]]
                pline_2 = [centers[line_center_id][2], centers[line_center_id][1]]
                pitem = [centers[con_center_id_ij][2], centers[con_center_id_ij][1]]
                pedal_point = Pedal_Point(pline_1, pline_2, pitem)
                between_item_logic_2 = Between_Item_Logic(pline_1, pline_2, pedal_point)
                
                # Skip if the connected center is between any pair of triangle vertices
                if between_item_logic_1 or between_item_logic_2:
                    continue
                # Otherwise, check for adding to line items
                elif con_center_id_ij not in dist_con_items[center_id_j][1]:
                    # Calculate weighted difference between centers
                    delta_vbl_1 = np.abs(centers[center_id_j] - centers[con_center_id_ij]) * WeightFactor
                    # Check if the maximum difference is not in the x dimension
                    if np.argmax(delta_vbl_1) != 0:
                        dist_con_items[center_id_j][1] += [con_center_id_ij]
    
    return dist_con_items


def Get_Related_Ids_RR(regions_data, centers, rr_centers_id, connected_ids_dict_lists, 
                      edges, angles, TolAngle, TolDistance):
    """
    Identify related centers based on position and direction consistency.
    
    Parameters:
        regions_data: Data for regions
        centers: Array containing coordinates of all centers
        rr_centers_id: IDs of centers to analyze
        connected_ids_dict_lists: Lists of dictionaries with connected center IDs
        edges: Edge information for centers
        angles: Array containing angles of all centers
        TolAngle: Tolerance angle threshold for direction consistency
        TolDistance: Tolerance distance threshold for position consistency
        
    Returns:
        related_ids: Dictionary mapping center IDs to related center IDs
        dist_con_items: Dictionary of connected items classified by distance
    """
    # Initialize dictionaries
    related_ids = {}
    line_items = {}
    between_items = {}
    dist_con_items = {}
    
    # Extract connected IDs dictionaries from the list
    connected_ids_dict_1 = connected_ids_dict_lists[0]
    connected_ids_dict_2 = connected_ids_dict_lists[1]
    connected_ids_dict_3 = connected_ids_dict_lists[2]
    
    # Initialize related_ids and dist_con_items for each center
    for center_id in rr_centers_id:
        related_ids[center_id] = []
        dist_con_items[center_id] = [[], []]  # [between_items, line_items]
    
    # Analyze each center for relationships
    for center_id_i in rr_centers_id:
        # Only analyze centers with edge value 0
        if edges[center_id_i] == 0:
            con_center_ids_i = connected_ids_dict_1[center_id_i]
            
            # Check each connected center
            for center_id_j in con_center_ids_i:
                if edges[center_id_j] == 0:
                    # Estimate direction consistency between centers
                    dist_con_items, related_ids, angle_logic = Estimate_Direction_Consistency(
                        centers, angles, connected_ids_dict_1, center_id_i, center_id_j,
                        dist_con_items, related_ids, TolAngle, TolDistance
                    )
                    
                    # If directions are consistent, do more detailed analysis
                    if angle_logic:
                        # Get expanded connected centers
                        con_center_ids_i = connected_ids_dict_3[center_id_i]
                        con_center_ids_j = connected_ids_dict_3[center_id_j]
                        
                        # Clean up between items
                        for between_id in dist_con_items[center_id_i][0]:
                            if between_id not in con_center_ids_i:
                                dist_con_items[center_id_i][0].remove(between_id)
                        
                        # Find common connected centers (line centers)
                        line_center_ids = list(set(con_center_ids_j) & set(dist_con_items[center_id_i][1]))
                        
                        # For each line center, analyze position consistency
                        for line_center_id in line_center_ids:
                            con_center_ids_j_2 = connected_ids_dict_3[line_center_id]
                            
                            # Estimate position consistency for different center combinations
                            dist_con_items = Estimate_Position_Consistency(
                                centers, center_id_i, center_id_j,
                                con_center_ids_j_2, dist_con_items, TolDistance
                            )
                            
                            dist_con_items = Estimate_Position_Consistency_2(
                                centers, center_id_i, center_id_j, line_center_id,
                                con_center_ids_j_2, dist_con_items, TolDistance
                            )
                            
                            # Check direction consistency for additional centers
                            for con_center_id_j_2 in con_center_ids_j_2:
                                dist_con_items, related_ids, angle_logic = Estimate_Direction_Consistency(
                                    centers, angles, connected_ids_dict_3, center_id_j, con_center_id_j_2,
                                    dist_con_items, related_ids, TolAngle, TolDistance
                                )
    
    # Add between items and line items to related_ids
    related_ids, between_items = Add_Items_To_Related_Ids(related_ids, dist_con_items, between_items, 0)
    related_ids, line_items = Add_Items_To_Related_Ids(related_ids, dist_con_items, line_items, 1)
    
    return related_ids, dist_con_items


def Update_Related_Ids(related_ids):
    """
    Update related IDs by merging connected components.
    
    Parameters:
        related_ids: Dictionary mapping center IDs to related center IDs
        
    Returns:
        related_ids_2: Updated dictionary with merged related IDs
    """
    # Initialize dictionaries
    related_ids_1 = {}
    related_ids_2 = {}
    keys = np.array(list(related_ids.keys()))
    
    # Initialize related_ids_1 with empty lists
    for key in keys:
        related_ids_1[key] = []
    
    key_used = []
    loop_i = 1
    
    # Merge related IDs for each key
    for key_0 in keys:
        for key_1 in keys:
            for key_1_item in related_ids[key_1]:
                # Check if there's a connection between key_0 and key_1
                if ((key_0 in related_ids[key_1] or key_1 in related_ids[key_0] or 
                    key_1_item in related_ids[key_0]) and key_1 not in key_used):
                    # Merge related IDs
                    related_ids_1[key_0] += related_ids[key_0]
                    related_ids_1[key_0] += related_ids[key_1]
                    related_ids_1[key_0].append(key_1)
                    key_used.append(key_1)
        loop_i += 1
    
    # Remove empty lists and create unique sets
    for key in related_ids_1.keys():
        if len(related_ids_1[key]) != 0:
            related_ids_2[key] = list(set(related_ids_1[key]))
    
    return related_ids_2

    
def Add_Isolated_Con_Neighbor(related_ids, con_ids_dict_enhanced):
    """
    Add isolated connected neighbors to related IDs.
    
    Parameters:
        related_ids: Dictionary mapping center IDs to related center IDs
        con_ids_dict_enhanced: Enhanced dictionary of connected center IDs
        
    Returns:
        related_ids_add_enhanced: Updated dictionary with added isolated neighbors
    """
    # Deep copy to avoid modifying the original dictionary
    related_ids_add_enhanced = copy.deepcopy(related_ids)
    
    # For each key in related_ids
    for key in related_ids.keys():
        # Check each clump ID related to this key
        for clump_id in related_ids[key]:
            # Get connected IDs for this clump
            connected_ids_enhanced = con_ids_dict_enhanced[clump_id]
            
            # Check each connected ID
            for connected_id_enhanced in connected_ids_enhanced:
                # If not already related to the key
                if connected_id_enhanced not in related_ids[key]:
                    add_flag = False
                    
                    # Check if all IDs connected to this ID are already related to the key
                    for connected_id_enhanced_2 in con_ids_dict_enhanced[connected_id_enhanced]:
                        if connected_id_enhanced_2 not in related_ids[key]:
                            add_flag = False
                            break
                        else:
                            add_flag = True
                    
                    # If all connected IDs are already related, add this ID too
                    if add_flag:
                        related_ids_add_enhanced[key] += [connected_id_enhanced]
    
    # Iteratively update related IDs until no more changes occur
    for i in range(len(related_ids_add_enhanced)):
        len_0 = len(related_ids_add_enhanced)
        related_ids_add_enhanced = Update_Related_Ids(related_ids_add_enhanced)
        len_1 = len(related_ids_add_enhanced)
        
        # Break if no change in dictionary size
        if len_0 == len_1:
            break
    
    return related_ids_add_enhanced



