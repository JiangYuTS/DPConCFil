import time
import os
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from skimage import measure, morphology
from collections import defaultdict
from tqdm import tqdm

from . import Filament_Class_Funs_Identification as FCFI
from . import Filament_Class_Funs_Analysis as FCFA
from . import Filament_Class_Funs_Table as FCFT


class FilamentInfor(object):
    def __init__(self, clumpsObj=None, parameters=None, save_files=None, SkeletonType=None):
        self.clumpsObj = clumpsObj
        self.parameters = parameters
        if type(parameters) != type(None):
            self.TolAngle = parameters[0]
            self.TolDistance = parameters[1]
            self.LWRatio = parameters[2]
        self.save_files = save_files
        self.AllowItems = 4
        self.AllowClumps = 2
        self.CalSub = False
        self.SmallSkeleton = 6
        self.SkeletonType = SkeletonType

        self.substructure_num = []
        self.substructure_ids = []

        filament_data_T = np.zeros_like(clumpsObj.origin_data)
        self.filament_data_T = filament_data_T

    def Filament_Clumps_Relation(self):
        TolAngle = self.TolAngle
        TolDistance = self.TolDistance
        AllowItems = self.AllowItems
        regions_data = self.clumpsObj.regions_data
        centers = self.clumpsObj.centers
        edges = self.clumpsObj.edges
        angles = self.clumpsObj.angles
        rc_dict = self.clumpsObj.rc_dict
        # clump_coords_dict = self.clumpsObj.clump_coords_dict
        connected_ids_dict = self.clumpsObj.connected_ids_dict

        related_ids_record = {}
        between_items_record = {}
        line_items_record = {}
        for key in tqdm(rc_dict.keys()):
            rr_centers_id = rc_dict[key]
            rr_centers = np.array(centers)[np.array(rr_centers_id)]
            rr_angles = np.array(angles)[np.array(rr_centers_id)]
            if len(rr_centers) > 1:
                related_ids, between_items, line_items = FCFI.Get_Related_Ids_RR \
                    (regions_data, centers, rr_centers_id, connected_ids_dict, edges, angles, TolAngle, TolDistance,
                     AllowItems)
                for i in range(len(related_ids)):
                    len_0 = len(related_ids)
                    related_ids = FCFI.Update_Related_Ids(related_ids)
                    len_1 = len(related_ids)
                    if len_0 == len_1:
                        break
                for key_1 in related_ids.keys():
                    related_ids_record[key_1] = related_ids[key_1]
                for key_2 in between_items.keys():
                    between_items_record[key_2] = between_items[key_2]
                for key_3 in line_items.keys():
                    line_items_record[key_3] = line_items[key_3]

        self.related_ids = related_ids_record
        self.between_items_record = between_items_record
        self.line_items_record = line_items_record

    def Filament_Infor_I(self, related_ids_T):
        data_wcs = self.clumpsObj.data_wcs
        data_ranges_lbv = self.clumpsObj.data_ranges_lbv
        origin_data = self.clumpsObj.origin_data
        regions_data = self.clumpsObj.regions_data
        connected_ids_dict = self.clumpsObj.connected_ids_dict
        clump_coords_dict = self.clumpsObj.clump_coords_dict
        SkeletonType = self.SkeletonType
        CalSub = self.CalSub

        filament_coords, filament_item, data_wcs_item, regions_data_T, start_coords, filament_item_mask_2D, lb_area = \
            FCFA.Filament_Coords(origin_data, regions_data, data_wcs, clump_coords_dict, related_ids_T, CalSub)

        od_mass = origin_data[filament_coords[:, 0], filament_coords[:, 1], filament_coords[:, 2]]
        mass_array = np.c_[od_mass, od_mass, od_mass]
        filament_com = np.around((mass_array * filament_coords).sum(0) / od_mass.sum(), 3).tolist()
        if data_wcs.naxis == 4:
            filament_com_wcs_T = data_wcs.all_pix2world(filament_com[2], filament_com[1], filament_com[0], 0, 0)
        elif data_wcs.naxis == 3:
            filament_com_wcs_T = data_wcs.all_pix2world(filament_com[2], filament_com[1], filament_com[0], 0)
        filament_com_wcs = np.around(np.c_[filament_com_wcs_T[0], filament_com_wcs_T[1], \
                                           filament_com_wcs_T[2] / 1000][0], 3).tolist()
        filament_com_item = [filament_com[0] - start_coords[0], filament_com[1] - start_coords[1],
                             filament_com[2] - start_coords[2]]
        D, V, size_ratio, angle = FCFA.Get_DV(filament_item, filament_com_item)

        #         filament_data = np.zeros_like(origin_data)
        filament_data = self.filament_data_T.copy()
        filament_data[filament_coords[:, 0], filament_coords[:, 1], filament_coords[:, 2]] = od_mass

        dc_no_sub, lengh, lw_ratio, skeleton_coords_2D, all_skeleton_coords = \
            FCFA.Cal_Lengh_Width_Ratio(False, regions_data_T, related_ids_T, connected_ids_dict, clump_coords_dict, \
                                       filament_item_mask_2D, filament_item, len(related_ids_T), SkeletonType)
        dc_no_sub = FCFA.Update_Dictionary_Cuts(dc_no_sub, start_coords)

        lbv_item_start, lbv_item_end, velocity_map_item, v_skeleton_com_delta = \
            FCFA.Cal_Velocity_Map(filament_item, skeleton_coords_2D, data_wcs_item)

        self.dc_no_sub = dc_no_sub
        self.clumps_number = len(related_ids_T)
        self.filament_com = filament_com
        self.filament_com_wcs = filament_com_wcs
        self.filament_length = lengh
        self.filament_ratio = lw_ratio
        self.filament_angle = angle
        self.filament_data = filament_data
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

    def Filament_Infor_All(self):
        LWRatio = self.LWRatio
        # data_wcs = self.clumpsObj.data_wcs
        # origin_data = self.clumpsObj.origin_data
        regions_data = self.clumpsObj.regions_data

        self.Filament_Clumps_Relation()
        related_ids = self.related_ids
        filament_infor_all = defaultdict(list)
        filament_regions_data = np.zeros_like(regions_data, dtype=np.int32)
        keys = list(related_ids.keys())
        k = 1
        for key in tqdm(keys):
            related_ids_T = related_ids[key]
            self.Filament_Infor_I(related_ids_T)
            filament_data = self.filament_data
            regions = measure.regionprops(measure.label(filament_data > 0, connectivity=3))
            if len(regions) > 1:
                max_area = 0
                for region in regions:
                    if region.area > max_area:
                        max_area = region.area
                        max_region = region
                coords = max_region.coords
                filament_clump_ids_usable = list(
                    set(regions_data[(coords[:, 0], coords[:, 1], coords[:, 2])].astype(int) - 1))
                related_ids[key] = filament_clump_ids_usable
                related_ids_T = related_ids[key]
                self.Filament_Infor_I(related_ids_T)

            filament_ratio = self.filament_ratio
            if filament_ratio < LWRatio or len(related_ids_T) < self.AllowClumps:
                del related_ids[key]
            else:
                filament_infor_all['filament_com'] += [self.filament_com]
                filament_infor_all['filament_com_wcs'] += [self.filament_com_wcs]
                filament_infor_all['clumps_number'] += [self.clumps_number]
                filament_infor_all['filament_length'] += [self.filament_length]
                filament_infor_all['filament_ratio'] += [self.filament_ratio]
                filament_infor_all['filament_angle'] += [self.filament_angle]
                filament_infor_all['lb_areas'] += [self.lb_area]
                #                 filament_infor_all['v_spans'] += [self.v_span]
                filament_infor_all['v_grads'] += [self.v_grad]
                filament_regions_data[
                    self.filament_coords[:, 0], self.filament_coords[:, 1], self.filament_coords[:, 2]] = k
                filament_infor_all['start_coords'] += [self.start_coords]
                filament_infor_all['skeleton_coords_2D'] += [self.skeleton_coords_2D]
                k += 1

        filament_infor_all['related_ids'] = related_ids
        self.related_ids = related_ids
        self.filament_com_all = filament_infor_all['filament_com']
        self.filament_com_wcs_all = filament_infor_all['filament_com_wcs']
        self.clumps_number_all = filament_infor_all['clumps_number']
        self.filament_length_all = filament_infor_all['filament_length']
        self.filament_ratio_all = filament_infor_all['filament_ratio']
        self.filament_angle_all = filament_infor_all['filament_angle']
        self.filament_lb_area_all = filament_infor_all['lb_areas']
        #         self.filament_v_span_all = filament_infor_all['v_spans']
        self.filament_v_grad_all = filament_infor_all['v_grads']
        self.start_coords_all = filament_infor_all['start_coords']
        self.skeleton_coords_2D_all = filament_infor_all['skeleton_coords_2D']
        self.filament_regions_data = filament_regions_data

        return filament_infor_all

    def Get_Item_Dictionary_Cuts(self, filament_clumps_id, dictionary_cuts=None, SampInt=1, Substructure=False):
        self.SampInt = SampInt
        self.Substructure = Substructure
        SkeletonType = self.SkeletonType
        SmallSkeleton = self.SmallSkeleton
        centers = self.clumpsObj.centers
        origin_data = self.clumpsObj.origin_data
        regions_data = self.clumpsObj.regions_data
        data_wcs = self.clumpsObj.data_wcs
        connected_ids_dict = self.clumpsObj.connected_ids_dict
        clump_coords_dict = self.clumpsObj.clump_coords_dict
        filament_coords = self.filament_coords

        filament_centers_LBV = []
        max_path_record = []
        max_edges_record = []

        for index in filament_clumps_id:
            filament_centers_LBV.append([centers[index][2], centers[index][1], centers[index][0]])
        sorted_id = sorted(range(len(filament_centers_LBV)), key=lambda k: filament_centers_LBV[k], reverse=False)
        filament_centers_LBV = (np.array(filament_centers_LBV)[sorted_id])
        filament_clumps_id = np.array(filament_clumps_id)[sorted_id]

        filament_mask_2D = np.zeros((regions_data.shape[1], regions_data.shape[2]), dtype=np.int16)
        filament_mask_2D[filament_coords[:, 1], filament_coords[:, 2]] = 1
        fil_mask = filament_mask_2D.astype(bool)
        Graph, Tree = FCFA.Graph_Infor_SubStructure(origin_data, fil_mask, filament_centers_LBV, filament_clumps_id, \
                                                    self.clumpsObj.connected_ids_dict)
        max_path_record, max_edges_record = FCFA.Get_Max_Path_Recursion(origin_data, filament_centers_LBV, \
                                                                        max_path_record, max_edges_record, Graph, Tree)
        max_path_record = FCFA.Update_Max_Path_Record(max_path_record)

        self.CalSub = False
        max_path_used = []
        skeleton_coords_record = []
        substructure_num_i = 0
        if Substructure:
            CalSub = True
            substructure_ids_T = []
            for subpart_id in range(0, len(max_path_record)):
                max_path_i = max_path_record[subpart_id]
                max_path_used.append(max_path_i)
                related_ids_T = np.array(filament_clumps_id)[max_path_i]

                if type(dictionary_cuts) != type(None) and len(related_ids_T)>0:
                    filament_coords, filament_item, data_wcs_item, regions_data_T, start_coords, filament_item_mask_2D, lb_area = \
                        FCFA.Filament_Coords(origin_data, regions_data, data_wcs, clump_coords_dict, related_ids_T,
                                             CalSub)

                    fil_image = filament_item.sum(0)
                    fil_mask = filament_item_mask_2D.astype(bool)

                    common_clump_id, common_sc_item = FCFA.Get_Common_Skeleton(filament_clumps_id, \
                                                                               max_path_i, max_path_used,
                                                                               skeleton_coords_record, start_coords,
                                                                               clump_coords_dict)
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

                    skeleton_coords_record.append(skeleton_coords_2D + start_coords[1:])
                    if not small_sc:
                        #                         skeleton_coords_record.append(skeleton_coords_2D+start_coords[1:])
                        skeleton_coords_2D = skeleton_coords_2D + np.random.random(skeleton_coords_2D.shape) / 10000
                        dictionary_cuts = FCFA.Cal_Dictionary_Cuts(SampInt, self.CalSub, regions_data_T, related_ids_T, \
                                                                   connected_ids_dict, clump_coords_dict,
                                                                   skeleton_coords_2D, \
                                                                   fil_image, fil_mask, dictionary_cuts, start_coords)
                        dictionary_cuts = FCFA.Update_Dictionary_Cuts(dictionary_cuts, start_coords)

                        substructure_ids_T += [list(related_ids_T)]
                        substructure_num_i += 1
            if substructure_num_i == 0:
                substructure_num_i = 1
                substructure_ids_T = [list(filament_clumps_id)]
            self.substructure_num += [substructure_num_i]
            self.substructure_ids += [substructure_ids_T]
            if len(dictionary_cuts['points']) == 0:
                dictionary_cuts = self.dc_no_sub
        else:
            if SampInt == 1:
                dictionary_cuts = self.dc_no_sub
            else:
                start_coords = self.start_coords
                regions_data_T = self.regions_data_T
                skeleton_coords_2D = self.skeleton_coords_2D
                fil_image = self.filament_item.sum(0)
                fil_mask = self.filament_item_mask_2D.astype(bool)
                skeleton_coords_2D = skeleton_coords_2D - start_coords[1:] + np.random.random(
                    skeleton_coords_2D.shape) / 10000
                dictionary_cuts = FCFA.Cal_Dictionary_Cuts(SampInt, self.CalSub, regions_data_T, filament_clumps_id, \
                                                           connected_ids_dict, clump_coords_dict, skeleton_coords_2D,
                                                           fil_image, fil_mask, dictionary_cuts)
                dictionary_cuts = FCFA.Update_Dictionary_Cuts(dictionary_cuts, start_coords)
        self.dictionary_cuts = dictionary_cuts
        return dictionary_cuts

    def Filament_Detect(self):
        start_1 = time.time()
        start_2 = time.ctime()
        save_files = self.save_files
        mask_name = save_files[0]
        filament_table_pix_name = save_files[1]
        filament_table_wcs_name = save_files[2]
        filament_infor_name = save_files[3]
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
        if exist_logic:
            filament_infor_all = self.Filament_Infor_All()
            np.savez(filament_infor_name, filament_infor_all=filament_infor_all)
            coms_vbl = self.filament_com_all
            self.filament_number = len(coms_vbl)
            if len(coms_vbl) != 0:
                filament_regions_data = self.filament_regions_data
                Filament_Table_Pix = FCFT.Table_Interface_Pix(self)
                Filament_Table_WCS = FCFT.Table_Interface_WCS(self)
                fits.writeto(mask_name, filament_regions_data, overwrite=True)
                Filament_Table_Pix.write(filament_table_pix_name, overwrite=True)
                Filament_Table_WCS.write(filament_table_wcs_name, overwrite=True)
                end_1 = time.time()
                end_2 = time.ctime()
                delta_time = np.around(end_1 - start_1, 2)
                par_time_record = np.hstack(
                    [[self.TolAngle, self.TolDistance, self.LWRatio, start_2, end_2, delta_time]])
                par_time_record = Table(par_time_record,
                                        names=['TolAngle', 'TolDistance', 'LWRatio', 'Start', 'End', 'DTime'])
                par_time_record.write(filament_table_pix_name[:-4] + '_DPConFil_record.csv', overwrite=True)
                self.delta_time = delta_time
                print('Number:', len(coms_vbl))
                print('Time:', delta_time)
                return filament_infor_all, Filament_Table_Pix, Filament_Table_WCS
            else:
                end_1 = time.time()
                end_2 = time.ctime()
                delta_time = np.around(end_1 - start_1, 2)
                par_time_record = np.hstack(
                    [[self.TolAngle, self.TolDistanc, self.LWRatio, start_2, end_2, delta_time]])
                par_time_record = Table(par_time_record,
                                        names=['TolAngle', 'TolDistanc', 'LWRatio', 'Start', 'End', 'DTime'])
                par_time_record.write(filament_table_pix_name[:-4] + '_DPConFil_record.csv', overwrite=True)
                self.delta_time = delta_time
                print('Number:', len(coms_vbl))
                print('Time:', delta_time)
                return None, None, None
