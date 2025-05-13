import time
import os
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from skimage import measure, morphology
from collections import defaultdict
from tqdm import tqdm

# Importing related modules from the same package
from . import Filament_Class_Funs_Identification as FCFI  # Functions for filament identification
from . import Filament_Class_Funs_Analysis as FCFA        # Functions for filament analysis
from . import Filament_Class_Funs_Table as FCFT           # Functions for creating output tables


class FilamentInfor(object):
    """
    A class for identifying and analyzing filamentary structures in astronomical data.
    
    This class processes clump data to identify filamentary structures based on geometric
    and spatial relationships between clumps. It calculates various properties of these
    filaments, including their length, width, orientation, and velocity gradients.
    """
    
    def __init__(self, clumpsObj=None, parameters=None, save_files=None, SkeletonType=None):
        """
        Initialize the FilamentInfor object.
        
        Parameters:
        -----------
        clumpsObj : object
            Object containing clump information and data
        parameters : list
            List of parameters [TolAngle, TolDistance, LWRatio] for filament identification
        save_files : list
            List of filenames for saving results
        SkeletonType : str
            Type of skeleton algorithm to use ('Morphology' or 'Intensity')
        """
        self.clumpsObj = clumpsObj
        self.parameters = parameters
        if type(parameters) != type(None):
            # Extract individual parameters from the list
            self.TolAngle = parameters[0]     # Tolerance angle for filament detection
            self.TolDistance = parameters[1]  # Tolerance distance for filament detection
            self.LWRatio = parameters[2]      # Length-to-width ratio threshold
        self.save_files = save_files
        self.AllowItems = 4          # Minimum number of items allowed in a filament
        self.AllowClumps = 2         # Minimum number of clumps allowed in a filament
        self.CalSub = False          # Flag for calculating substructures
        self.SmallSkeleton = 6       # Threshold for small skeleton detection
        self.SkeletonType = SkeletonType  # Type of skeleton algorithm to use

        # Initialize lists for storing substructure information
        self.substructure_num = []       # Number of substructures
        self.substructure_ids = []       # IDs of substructures
        self.skeleton_coords_record = [] # Coordinates of skeletons

        # Initialize filament data array (commented out)
        # filament_data_T = np.zeros_like(clumpsObj.origin_data)
        # self.filament_data_T = filament_data_T

    def Filament_Clumps_Relation(self):
        """
        Identify related clumps that may form filamentary structures based on 
        geometric relationships (angles and distances).
        
        This function analyzes the spatial relationship between clumps and identifies
        potential filamentary structures by grouping clumps that align within the
        specified tolerance angle and distance parameters.
        
        Sets:
        -----
        self.related_ids : dict
            Dictionary of related clump IDs that form potential filaments
        self.related_ids_add_enhanced : dict
            Enhanced dictionary with isolated connected neighbors added
        """
        # Extract parameters and clump information
        TolAngle = self.TolAngle
        TolDistance = self.TolDistance
        regions_data = self.clumpsObj.regions_data
        centers = self.clumpsObj.centers
        edges = self.clumpsObj.edges
        angles = self.clumpsObj.angles
        rc_dict = self.clumpsObj.rc_dict
        connected_ids_dict_lists = self.clumpsObj.connected_ids_dict_lists
        
        # Dictionary to store related clump IDs
        related_ids = {}
        
        # Loop through all keys in rc_dict (region-clump dictionary)
        for key in tqdm(rc_dict.keys()):
            rr_centers_id = rc_dict[key]
            if len(rr_centers_id) > 0:
                # Get centers and angles of related regions
                rr_centers = np.array(centers)[np.array(rr_centers_id)]
                rr_angles = np.array(angles)[np.array(rr_centers_id)]
                
                # Get related IDs based on angle and distance criteria
                related_ids_temp, dist_con_items = FCFI.Get_Related_Ids_RR(
                    regions_data, centers, rr_centers_id, connected_ids_dict_lists, 
                    edges, angles, TolAngle, TolDistance)
        
                # Iteratively update related IDs until convergence
                for i in range(len(related_ids_temp)):
                    len_0 = len(related_ids_temp)
                    related_ids_temp = FCFI.Update_Related_Ids(related_ids_temp)
                    len_1 = len(related_ids_temp)
                    if len_0 == len_1:
                        break
                
                # Add related IDs to the main dictionary
                for key_1 in related_ids_temp.keys():
                    related_ids[key_1] = related_ids_temp[key_1]
        
        # Enhance the related IDs by adding isolated connected neighbors
        related_ids_add_enhanced = FCFI.Add_Isolated_Con_Neighbor(related_ids, connected_ids_dict_lists[1])

        # Store the results
        self.related_ids = related_ids_add_enhanced
        self.related_ids_add_enhanced = related_ids_add_enhanced.copy()

    def Filament_Infor_I(self, related_ids_T):
        """
        Calculate properties of a single filament identified by related_ids_T.
        
        This function computes various properties of a filament, including its
        coordinates, center of mass, length, width ratio, orientation angle,
        and velocity gradient.
        
        Parameters:
        -----------
        related_ids_T : list
            List of clump IDs that form the filament
        
        Sets:
        -----
        Multiple attributes describing the filament's properties (see code)
        """
        # Extract necessary data from clumpsObj
        data_wcs = self.clumpsObj.data_wcs
        data_ranges_lbv = self.clumpsObj.data_ranges_lbv
        origin_data = self.clumpsObj.origin_data
        regions_data = self.clumpsObj.regions_data
        connected_ids_dict = self.clumpsObj.connected_ids_dict
        clump_coords_dict = self.clumpsObj.clump_coords_dict
        SkeletonType = self.SkeletonType
        CalSub = self.CalSub

        # Get filament coordinates and related data
        filament_coords, filament_item, data_wcs_item, regions_data_T, start_coords, filament_item_mask_2D, lb_area = \
            FCFA.Filament_Coords(origin_data, regions_data, data_wcs, clump_coords_dict, related_ids_T, CalSub)

        # Calculate center of mass using intensity weighting
        od_mass = origin_data[filament_coords[:, 0], filament_coords[:, 1], filament_coords[:, 2]]
        mass_array = np.c_[od_mass, od_mass, od_mass]
        filament_com = np.around((mass_array * filament_coords).sum(0) / od_mass.sum(), 3).tolist()
        
        # Convert center of mass from pixel coordinates to world coordinates (WCS)
        if data_wcs.naxis == 4:
            filament_com_wcs_T = data_wcs.all_pix2world(filament_com[2], filament_com[1], filament_com[0], 0, 0)
        elif data_wcs.naxis == 3:
            filament_com_wcs_T = data_wcs.all_pix2world(filament_com[2], filament_com[1], filament_com[0], 0)
        else:
            print('Please check the naxis of data_wcs, 3 or 4.')
        
        # Format world coordinates
        filament_com_wcs = np.around(np.c_[filament_com_wcs_T[0], filament_com_wcs_T[1], \
                                           filament_com_wcs_T[2] / 1000][0], 3).tolist()
        
        # Calculate center of mass relative to the start coordinates
        filament_com_item = [filament_com[0] - start_coords[0], filament_com[1] - start_coords[1],
                             filament_com[2] - start_coords[2]]
        
        # Get filament dimensions, volume, size ratio, and orientation angle
        D, V, size_ratio, angle = FCFA.Get_DV(filament_item, filament_com_item)

        # Calculate length, width ratio, and skeleton coordinates
        dc_no_sub, lengh, lw_ratio, skeleton_coords_2D, all_skeleton_coords = \
            FCFA.Cal_Lengh_Width_Ratio(False, regions_data_T, related_ids_T, connected_ids_dict, clump_coords_dict, \
                                       filament_item_mask_2D, filament_item, len(related_ids_T), SkeletonType)
        
        # Update dictionary cuts with start coordinates
        dc_no_sub = FCFA.Update_Dictionary_Cuts(dc_no_sub, start_coords)

        # Calculate velocity map and velocity gradient
        lbv_item_start, lbv_item_end, velocity_map_item, v_skeleton_com_delta = \
            FCFA.Cal_Velocity_Map(filament_item, skeleton_coords_2D, data_wcs_item)

        # Store all calculated properties as attributes
        self.dc_no_sub = dc_no_sub
        self.clumps_number = len(related_ids_T)
        self.filament_com = filament_com
        self.filament_com_wcs = filament_com_wcs
        self.filament_length = lengh
        self.filament_ratio = lw_ratio
        self.filament_angle = angle
        # self.filament_data = filament_data
        self.filament_coords = filament_coords
        self.lb_area = lb_area
        self.skeleton_coords_2D = skeleton_coords_2D + start_coords[1:]
        if type(all_skeleton_coords) != type(None):
            all_skeleton_coords = all_skeleton_coords + start_coords[1:]
            self.all_skeleton_coords = all_skeleton_coords
        self.start_coords = start_coords
        self.filament_item = filament_item
        self.regions_data_T = regions_data_T
        self.filament_com_item = filament_com_item
        self.filament_item_mask_2D = filament_item_mask_2D
        self.data_wcs_item = data_wcs_item
        self.velocity_map_item = velocity_map_item
        self.v_grad = v_skeleton_com_delta

    def Filament_Infor_All(self, related_ids=None):
        """
        Calculate properties for all identified filaments.
        
        This function processes all filaments identified by the Filament_Clumps_Relation
        method, computes their properties, and filters them based on the length-to-width
        ratio and minimum number of clumps criteria.
        
        Parameters:
        -----------
        related_ids : dict, optional
            Dictionary of related clump IDs that form filaments. If None,
            Filament_Clumps_Relation will be called to generate it.
            
        Returns:
        --------
        filament_infor_all : defaultdict
            Dictionary containing all filament properties
        
        Sets:
        -----
        Multiple attributes storing filament properties for all filaments
        """
        LWRatio = self.LWRatio

        # Get related IDs if not provided
        if type(related_ids) == type(None):
            self.Filament_Clumps_Relation()
            related_ids = self.related_ids
        
        # Initialize dictionary to store all filament information
        filament_infor_all = defaultdict(list)
        
        # Initialize array to store filament regions data
        filament_regions_data = np.zeros_like(self.clumpsObj.regions_data, dtype=np.int32)
        
        keys = list(related_ids.keys())
        k = 1  # Counter for filament regions
        
        # Process each set of related IDs (potential filament)
        for key in tqdm(keys):
            related_ids_T = related_ids[key]
            
            # Calculate properties for this filament
            self.Filament_Infor_I(related_ids_T)
            
            # Check if filament has multiple disjoint regions
            regions = measure.regionprops(measure.label(self.filament_item > 0, connectivity=3))
            if len(regions) > 1:
                # If multiple regions exist, use only the largest one
                max_area = 0
                for region in regions:
                    if region.area > max_area:
                        max_area = region.area
                        max_region = region
                
                # Update related IDs based on the largest region
                coords = max_region.coords + self.start_coords
                filament_clump_ids_usable = list(set(self.clumpsObj.\
                        regions_data[(coords[:, 0], coords[:, 1], coords[:, 2])].astype(int) - 1))
                related_ids[key] = filament_clump_ids_usable
                related_ids_T = related_ids[key]
                
                # Recalculate properties with updated related IDs
                self.Filament_Infor_I(related_ids_T)

            # Filter filaments based on length-to-width ratio and number of clumps
            filament_ratio = self.filament_ratio
            if filament_ratio < LWRatio or len(related_ids_T) < self.AllowClumps:
                del related_ids[key]
            else:
                # Store properties for this filament
                filament_infor_all['filament_com'] += [self.filament_com]
                filament_infor_all['filament_com_wcs'] += [self.filament_com_wcs]
                filament_infor_all['clumps_number'] += [self.clumps_number]
                filament_infor_all['filament_length'] += [self.filament_length]
                filament_infor_all['filament_ratio'] += [self.filament_ratio]
                filament_infor_all['filament_angle'] += [self.filament_angle]
                filament_infor_all['lb_areas'] += [self.lb_area]
                filament_infor_all['v_grads'] += [self.v_grad]
                
                # Mark this filament in the regions data array
                filament_regions_data[
                    self.filament_coords[:, 0], self.filament_coords[:, 1], self.filament_coords[:, 2]] = k
                    
                filament_infor_all['start_coords'] += [self.start_coords]
                filament_infor_all['skeleton_coords_2D'] += [self.skeleton_coords_2D]
                k += 1

        # Store the final related IDs
        filament_infor_all['related_ids'] = related_ids
        self.related_ids = related_ids
        
        # Store filament properties as attributes for easy access
        self.filament_com_all = filament_infor_all['filament_com']
        self.filament_com_wcs_all = filament_infor_all['filament_com_wcs']
        self.clumps_number_all = filament_infor_all['clumps_number']
        self.filament_length_all = filament_infor_all['filament_length']
        self.filament_ratio_all = filament_infor_all['filament_ratio']
        self.filament_angle_all = filament_infor_all['filament_angle']
        self.filament_lb_area_all = filament_infor_all['lb_areas']
        self.filament_v_grad_all = filament_infor_all['v_grads']
        self.start_coords_all = filament_infor_all['start_coords']
        self.skeleton_coords_2D_all = filament_infor_all['skeleton_coords_2D']
        self.filament_regions_data = filament_regions_data

        return filament_infor_all

    def Get_Item_Dictionary_Cuts(self, filament_clumps_id, dictionary_cuts=None, SampInt=1, Substructure=False):
        """
        Calculate cross-sections (cuts) along the filament's skeleton.
        
        This function analyzes the filament's skeleton and calculates perpendicular
        cross-sections (cuts) at regular intervals along the skeleton. It can also
        identify and analyze substructures within the filament.
        
        Parameters:
        -----------
        filament_clumps_id : list
            List of clump IDs that form the filament
        dictionary_cuts : dict, optional
            Dictionary to store cut information
        SampInt : int, optional
            Sampling interval for cuts along the skeleton (default: 1)
        Substructure : bool, optional
            Flag to calculate substructures (default: False)
            
        Returns:
        --------
        dictionary_cuts : dict
            Dictionary containing information about cuts along the skeleton
            
        Sets:
        -----
        self.dictionary_cuts : dict
            Same as the return value
        """
        self.SampInt = SampInt
        self.Substructure = Substructure
        SkeletonType = self.SkeletonType
        SmallSkeleton = self.SmallSkeleton
        
        # Extract necessary data from clumpsObj
        centers = self.clumpsObj.centers
        origin_data = self.clumpsObj.origin_data
        regions_data = self.clumpsObj.regions_data
        data_wcs = self.clumpsObj.data_wcs
        connected_ids_dict = self.clumpsObj.connected_ids_dict
        clump_coords_dict = self.clumpsObj.clump_coords_dict
        filament_coords = self.filament_coords

        # Initialize lists for path and edge records
        filament_centers_LBV = []
        max_path_record = []
        max_edges_record = []

        # Convert clump center coordinates to LBV format and sort them
        for index in filament_clumps_id:
            filament_centers_LBV.append([centers[index][2], centers[index][1], centers[index][0]])
        sorted_id = sorted(range(len(filament_centers_LBV)), key=lambda k: filament_centers_LBV[k], reverse=False)
        filament_centers_LBV = (np.array(filament_centers_LBV)[sorted_id])
        filament_clumps_id = np.array(filament_clumps_id)[sorted_id]

        # Create 2D mask of the filament
        filament_mask_2D = np.zeros((regions_data.shape[1], regions_data.shape[2]), dtype=np.int16)
        filament_mask_2D[filament_coords[:, 1], filament_coords[:, 2]] = 1
        fil_mask = filament_mask_2D.astype(bool)
        
        # Build graph and tree for substructure analysis
        Graph, Tree = FCFA.Graph_Infor_SubStructure(origin_data, fil_mask, filament_centers_LBV, filament_clumps_id, \
                                                    self.clumpsObj.connected_ids_dict)
        
        # Recursively find maximum paths through the graph
        max_path_record, max_edges_record = FCFA.Get_Max_Path_Recursion(origin_data, filament_centers_LBV, \
                                                                        max_path_record, max_edges_record, Graph, Tree)
        max_path_record = FCFA.Update_Max_Path_Record(max_path_record)

        # Initialize variables for substructure analysis
        self.CalSub = False
        max_path_used = []
        skeleton_coords_record = []
        substructure_num_i = 0
        
        # Process substructures if requested
        if Substructure:
            CalSub = True
            substructure_ids_T = []
            
            # Process each substructure (path)
            for subpart_id in range(0, len(max_path_record)):
                max_path_i = max_path_record[subpart_id]
                max_path_used.append(max_path_i)
                related_ids_T = np.array(filament_clumps_id)[max_path_i]

                if type(dictionary_cuts) != type(None) and len(related_ids_T) > 0:
                    # Get coordinates and data for this substructure
                    filament_coords, filament_item, data_wcs_item, regions_data_T, start_coords, filament_item_mask_2D, lb_area = \
                        FCFA.Filament_Coords(origin_data, regions_data, data_wcs, clump_coords_dict, related_ids_T,
                                             CalSub)

                    # Create 2D projection and mask
                    fil_image = filament_item.sum(0)
                    fil_mask = filament_item_mask_2D.astype(bool)
                    
                    # Get common skeleton coordinates
                    common_clump_id, common_sc_item = FCFA.Get_Common_Skeleton(filament_clumps_id, related_ids_T,\
                                                                               max_path_i, max_path_used,
                                                                               skeleton_coords_record, start_coords,
                                                                               clump_coords_dict)
                    
                    # Get skeleton coordinates based on the specified method
                    if SkeletonType == 'Morphology':
                        skeleton_coords_2D, filament_skeleton, all_skeleton_coords = FCFA.Get_Single_Filament_Skeleton(
                            fil_mask)
                    elif SkeletonType == 'Intensity':
                        all_skeleton_coords = None
                        clumps_number = len(related_ids_T)
                        skeleton_coords_2D, small_sc = FCFA.Get_Single_Filament_Skeleton_Weighted(fil_image, fil_mask, \
                                                                                                  clumps_number,
                                                                                                  common_sc_item,
                                                                                                  SmallSkeleton)
                    else:
                        print('Please choose the skeleton_type between Morphology and Intensity')

                    # Record skeleton coordinates
                    skeleton_coords_record.append(skeleton_coords_2D + start_coords[1:])
                    
                    # Calculate dictionary cuts if the skeleton is not too small
                    if not small_sc:
                        # Add small random numbers to avoid exact coincidence of coordinates
                        skeleton_coords_2D = skeleton_coords_2D + np.random.random(skeleton_coords_2D.shape) / 10000
                        
                        # Calculate cuts along the skeleton
                        dictionary_cuts = FCFA.Cal_Dictionary_Cuts(SampInt, CalSub, regions_data_T, related_ids_T, \
                                                                   connected_ids_dict, clump_coords_dict,
                                                                   skeleton_coords_2D, \
                                                                   fil_image, fil_mask, dictionary_cuts, start_coords)
                        dictionary_cuts = FCFA.Update_Dictionary_Cuts(dictionary_cuts, start_coords)

                        # Record substructure information
                        substructure_ids_T += [list(related_ids_T)]
                        substructure_num_i += 1
                        
            # If no substructures were found, use the entire filament as one substructure
            if substructure_num_i == 0:
                substructure_num_i = 1
                substructure_ids_T = [list(filament_clumps_id)]
                
            # Record substructure information
            self.substructure_num += [substructure_num_i]
            self.substructure_ids += [substructure_ids_T]
            self.skeleton_coords_record += [skeleton_coords_record]
            
            # Use dc_no_sub if no cuts were calculated
            if len(dictionary_cuts['points']) == 0:
                dictionary_cuts = self.dc_no_sub
        else:
            # If not calculating substructures, use existing cuts or calculate new ones based on SampInt
            if SampInt == 1:
                dictionary_cuts = self.dc_no_sub
            else:
                # Extract necessary data
                start_coords = self.start_coords
                regions_data_T = self.regions_data_T
                skeleton_coords_2D = self.skeleton_coords_2D
                fil_image = self.filament_item.sum(0)
                fil_mask = self.filament_item_mask_2D.astype(bool)
                
                # Adjust skeleton coordinates and add small random numbers
                skeleton_coords_2D = skeleton_coords_2D - start_coords[1:] + np.random.random(
                    skeleton_coords_2D.shape) / 10000
                    
                # Calculate dictionary cuts
                dictionary_cuts = FCFA.Cal_Dictionary_Cuts(SampInt, self.CalSub, regions_data_T, filament_clumps_id, \
                                                           connected_ids_dict, clump_coords_dict, skeleton_coords_2D,
                                                           fil_image, fil_mask, dictionary_cuts)
                dictionary_cuts = FCFA.Update_Dictionary_Cuts(dictionary_cuts, start_coords)
                
        # Store and return the dictionary cuts
        self.dictionary_cuts = dictionary_cuts
        return dictionary_cuts

    def Filament_Detect(self, related_ids=None):
        """
        Main method to detect filaments and save results to files.
        
        This function coordinates the filament detection process, calculates
        filament properties, creates output tables, and saves results to files.
        
        Parameters:
        -----------
        related_ids : dict, optional
            Dictionary of related clump IDs that form filaments. If None,
            Filament_Infor_All will generate it.
            
        Returns:
        --------
        filament_infor_all : defaultdict
            Dictionary containing all filament properties
        Filament_Table_Pix : astropy.table.Table
            Table with filament properties in pixel coordinates
        Filament_Table_WCS : astropy.table.Table
            Table with filament properties in world coordinates
        """
        # Record start time
        start_1 = time.time()
        start_2 = time.ctime()
        
        # Extract filenames from save_files
        save_files = self.save_files
        mask_name = save_files[0]
        filament_table_pix_name = save_files[1]
        filament_table_wcs_name = save_files[2]
        filament_infor_name = save_files[3]
        
        # Check if output directory exists
        path_items = filament_table_pix_name.split('/')
        exist_logic = False
        if len(path_items) > 1:
            if not os.path.exists(filament_table_pix_name[:-len(path_items[-1])]):
                print('The path of outcat_name does not exist.')
            else:
                exist_logic = True
        elif len(path_items) == 1:
            if not os.path.exists(filament_table_pix_name):
                print('The path of outcat_name does not exist.')
            else:
                exist_logic = True
                
        # Proceed if output directory exists
        if exist_logic:
            # Calculate filament information
            filament_infor_all = self.Filament_Infor_All(related_ids)
            
            # Save filament information to NPZ file
            np.savez(filament_infor_name, filament_infor_all=filament_infor_all)
            
            # Get center of mass for all filaments
            coms_vbl = self.filament_com_all
            self.filament_number = len(coms_vbl)
            
            # Save results if filaments were found
            if len(coms_vbl) != 0:
                # Get filament regions data
                filament_regions_data = self.filament_regions_data
                
                # Create tables for pixel and WCS coordinates
                Filament_Table_Pix = FCFT.Table_Interface_Pix(self)
                Filament_Table_WCS = FCFT.Table_Interface_WCS(self)
                
                # Save results to files
                fits.writeto(mask_name, filament_regions_data, overwrite=True)
                Filament_Table_Pix.write(filament_table_pix_name, overwrite=True)
                Filament_Table_WCS.write(filament_table_wcs_name, overwrite=True)
                
                # Record end time and elapsed time
                end_1 = time.time()
                end_2 = time.ctime()
                delta_time = np.around(end_1 - start_1, 2)
                
                # Create and save parameter and time record
                par_time_record = np.hstack(
                    [[self.TolAngle, self.TolDistance, self.LWRatio, start_2, end_2, delta_time]])
                par_time_record = Table(par_time_record,
                                        names=['TolAngle', 'TolDistance', 'LWRatio', 'Start', 'End', 'DTime'])
                par_time_record.write(filament_table_pix_name[:-4] + '_DPConCFil_record.csv', overwrite=True)
                
                # Store elapsed time and print summary
                self.delta_time = delta_time
                print('Number:', len(coms_vbl))
                print('Time:', delta_time)
                
                return filament_infor_all, Filament_Table_Pix, Filament_Table_WCS
            else:
                # If no filaments were found, still record time information
                end_1 = time.time()
                end_2 = time.ctime()
                delta_time = np.around(end_1 - start_1, 2)
                
                # Create and save parameter and time record
                par_time_record = np.hstack(
                    [[self.TolAngle, self.TolDistance, self.LWRatio, start_2, end_2, delta_time]])
                par_time_record = Table(par_time_record,
                                        names=['TolAngle', 'TolDistance', 'LWRatio', 'Start', 'End', 'DTime'])
                par_time_record.write(filament_table_pix_name[:-4] + '_DPConCFil_record.csv', overwrite=True)
                
                # Store elapsed time and print summary
                self.delta_time = delta_time
                print('Number:', len(coms_vbl))
                print('Time:', delta_time)
                
                return None, None, None