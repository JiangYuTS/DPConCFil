import time
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from skimage import measure,morphology
from collections import defaultdict
from tqdm import tqdm

from . import Filament_Class_Funs_Identification as FCFI
from . import Filament_Class_Funs_Analysis as FCFA
from . import Filament_Class_Funs_Table as FCFT

class FilamentInfor(object):
    def __init__(self, clumpsObj=None,parameters=None,save_files=None,SkeletonType=None):
        self.clumpsObj = clumpsObj
        self.parameters = parameters
        if type(parameters) != type(None):
            self.TolAngle = parameters[0]
            self.TolDistance = parameters[1]
            self.LWRatio = parameters[2]
        self.save_files = save_files        
        self.AllowItems = 3
        self.AllowClumps = 2
        self.CalSub = False
        self.SkeletonType = SkeletonType
        
        self.substructure_num = []
        self.substructure_ids = []
        
    def Filament_Clumps_Relation(self):  
        TolAngle = self.TolAngle
        TolDistance = self.TolDistance
        AllowItems = self.AllowItems
        regions_data = self.clumpsObj.regions_data
        centers = self.clumpsObj.centers
        edges = self.clumpsObj.edges
        angles = self.clumpsObj.angles
        rc_dict = self.clumpsObj.rc_dict
        clump_coords_dict = self.clumpsObj.clump_coords_dict
        connected_ids_dict = self.clumpsObj.connected_ids_dict
        
        related_center_record = {}
        between_items_record = {}
        line_items_record = {}
        for key in tqdm(rc_dict.keys()):
            rr_centers_id = rc_dict[key]
            rr_centers = np.array(centers)[np.array(rr_centers_id)]
            rr_angles = np.array(angles)[np.array(rr_centers_id)]
            if len(rr_centers)>1:            
                related_center,between_items,line_items = FCFI.Get_Related_Center_RR\
                            (regions_data,centers,rr_centers_id,connected_ids_dict,edges,angles,TolAngle,TolDistance,AllowItems)
                for i in range(len(related_center)):
                    len_0 = len(related_center)
                    related_center = FCFI.Update_Related_Center(related_center)
                    len_1 = len(related_center)
                    if len_0 == len_1:
                        break
                for key_1 in related_center.keys():
                    related_center_record[key_1]=related_center[key_1]
                for key_2 in between_items.keys():
                    between_items_record[key_2]=between_items[key_2]
                for key_3 in line_items.keys():
                    line_items_record[key_3]=line_items[key_3]
                    
        self.related_center = related_center_record
        self.between_items_record = between_items_record
        self.line_items_record = line_items_record
        
    def Filament_Infor_I(self,related_center_ids):
        data_wcs = self.clumpsObj.data_wcs
        data_ranges_lbv = self.clumpsObj.data_ranges_lbv
        origin_data = self.clumpsObj.origin_data
        regions_data = self.clumpsObj.regions_data
        connected_ids_dict = self.clumpsObj.connected_ids_dict
        clump_coords_dict = self.clumpsObj.clump_coords_dict
        SkeletonType = self.SkeletonType
        CalSub = self.CalSub

        filament_infor_i = {}
        filament_coords = [[],[],[]]
        filament_mask_3D = np.zeros_like(regions_data)
        filament_data = np.zeros_like(origin_data)
        for core_id in related_center_ids:
            coords = clump_coords_dict[core_id]
            filament_mask_3D[coords] = 1
            filament_data[coords] = origin_data[coords]
            filament_coords[0] += list(coords[0])
            filament_coords[1] += list(coords[1])
            filament_coords[2] += list(coords[2])
        od_mass = origin_data[filament_coords[0],filament_coords[1],filament_coords[2]]
        mass_array = np.c_[od_mass,od_mass,od_mass]
        filament_com = np.around((mass_array*np.array(filament_coords).T).sum(0)\
                    /od_mass.sum(),3).tolist()
        
        if data_wcs.naxis==4:
            filament_com_wcs_T = data_wcs.all_pix2world(filament_com[2],filament_com[1],filament_com[0],0,0)
        elif data_wcs.naxis==3:
            filament_com_wcs_T = data_wcs.all_pix2world(filament_com[2],filament_com[1],filament_com[0],0)
        filament_com_wcs = np.around(np.c_[filament_com_wcs_T[0],filament_com_wcs_T[1],filament_com_wcs_T[2]/1000][0],3).tolist()
        coords_range,lb_area,v_delta,box_data = FCFA.Get_LBV_Table(filament_coords)
        x_min = coords_range[0]
        x_max = coords_range[1]
        y_min = coords_range[2]
        y_max = coords_range[3]
        z_min = coords_range[4]
        z_max = coords_range[5]
        length = np.max([x_max-x_min,y_max-y_min,z_max-z_min])+5
        filament_item =  np.zeros([length,length,length])
        start_x = np.int((length - (x_max-x_min))/2)
        start_y = np.int((length - (y_max-y_min))/2)
        start_z = np.int((length - (z_max-z_min))/2)
        filament_item[filament_coords[0]-x_min+start_x,filament_coords[1]-y_min+start_y,filament_coords[2]-z_min+start_z] = od_mass
        start_coords = [x_min-start_x,y_min-start_y,z_min-start_z]
        filament_com_item = [filament_com[0]-start_coords[0],filament_com[1]-start_coords[1],filament_com[2]-start_coords[2]]
        D,V,size_ratio,angle = FCFA.Get_DV(filament_item,filament_com_item)
        
        filament_mask_2D = np.zeros_like(filament_data.sum(0))
        filament_mask_2D[filament_coords[1],filament_coords[2]]=1
        filament_item_mask_2D = np.zeros_like(filament_item.sum(0))
        filament_item_mask_2D[filament_coords[1]-y_min+start_y,filament_coords[2]-z_min+start_z]=1

        dc_no_sub,lengh,lw_ratio,skeleton_coords_2D,all_skeleton_coords = \
                FCFA.Cal_Lengh_Width_Ratio(CalSub,regions_data,related_center_ids,connected_ids_dict,clump_coords_dict,\
                                   filament_mask_2D,filament_data,len(related_center_ids),SkeletonType)
        
        self.dc_no_sub = dc_no_sub
        self.clumps_number = len(related_center_ids)
        self.filament_com = filament_com
        self.filament_com_wcs = filament_com_wcs
        self.filament_length = lengh
        self.filament_ratio = lw_ratio
        self.filament_angle = angle
        self.filament_data = filament_data
        self.filament_coords = filament_coords
        self.lb_area = lb_area
        self.v_span = v_delta
        self.filament_mask_2D = filament_mask_2D
        self.skeleton_coords_2D = skeleton_coords_2D
        if type(all_skeleton_coords) != type(None):
            self.all_skeleton_coords = all_skeleton_coords
        self.start_coords = start_coords
        self.filament_item = filament_item
        self.filament_com_item = filament_com_item
        self.filament_item_mask_2D = filament_item_mask_2D
        self.item_skeleton_coords_2D = skeleton_coords_2D
        
        velocity_map_i,v_skeleton_com_delta = FCFA.Cal_V_Com_Delta(filament_data,skeleton_coords_2D,data_ranges_lbv)
        self.v_grad = v_skeleton_com_delta
        self.velocity_map = velocity_map_i
        
    def Filament_Infor_All(self):
        LWRatio = self.LWRatio
        data_wcs = self.clumpsObj.data_wcs
        origin_data = self.clumpsObj.origin_data
        regions_data = self.clumpsObj.regions_data

        self.Filament_Clumps_Relation()
        related_center = self.related_center
        filament_infor_all = defaultdict(list)
        filament_regions_data = np.zeros_like(regions_data,dtype='uint16')
        keys = list(related_center.keys()) 
        k = 1
        for key in tqdm(keys): 
            related_center_ids = related_center[key]
            self.Filament_Infor_I(related_center_ids)
            filament_data = self.filament_data
            regions = measure.regionprops(measure.label(filament_data>0,connectivity=3))
            if len(regions)>1:
                max_area = 0
                for region in regions:
                    if region.area > max_area:
                        max_area = region.area
                        max_region = region
                coords = max_region.coords
                filament_clump_ids_usable = list(set(regions_data[(coords[:,0],coords[:,1],coords[:,2])].astype(int)-1))
                related_center[key] = filament_clump_ids_usable
                related_center_ids = related_center[key]
                self.Filament_Infor_I(related_center_ids)

            filament_ratio = self.filament_ratio
            if filament_ratio < LWRatio or len(related_center_ids) < self.AllowClumps:
                del related_center[key]
            else:
                filament_infor_all['filament_com'] += [self.filament_com]
                filament_infor_all['filament_com_wcs'] += [self.filament_com_wcs]
                filament_infor_all['clumps_number'] += [self.clumps_number]
                filament_infor_all['filament_length'] += [self.filament_length]
                filament_infor_all['filament_ratio'] += [self.filament_ratio]
                filament_infor_all['filament_angle'] += [self.filament_angle]
                filament_infor_all['lb_areas'] += [self.lb_area]
                filament_infor_all['v_spans'] += [self.v_span]
                filament_infor_all['v_grads'] += [self.v_grad]
                filament_regions_data[self.filament_coords[0],self.filament_coords[1],self.filament_coords[2]] = k
                filament_infor_all['start_coords'] += [self.start_coords]
                filament_infor_all['skeleton_coords_2D'] += [self.skeleton_coords_2D]
                k += 1
                
        filament_infor_all['related_center'] = related_center
        self.related_center = related_center
        self.filament_com_all = filament_infor_all['filament_com']
        self.filament_com_wcs_all = filament_infor_all['filament_com_wcs']
        self.clumps_number_all = filament_infor_all['clumps_number']
        self.filament_length_all = filament_infor_all['filament_length']
        self.filament_ratio_all = filament_infor_all['filament_ratio']
        self.filament_angle_all = filament_infor_all['filament_angle']
        self.filament_lb_area_all = filament_infor_all['lb_areas']
        self.filament_v_span_all = filament_infor_all['v_spans']
        self.filament_v_grad_all = filament_infor_all['v_grads']
        self.start_coords_all = filament_infor_all['start_coords']
        self.skeleton_coords_2D_all = filament_infor_all['skeleton_coords_2D']
        self.filament_regions_data = filament_regions_data
        
        return filament_infor_all

    def Get_Item_Dictionary_Cuts(self,filament_clumps_id,dictionary_cuts=None,SampInt=1,Substructure=False):
        self.SampInt = SampInt
        self.Substructure = Substructure
        centers = self.clumpsObj.centers
        origin_data = self.clumpsObj.origin_data
        regions_data = self.clumpsObj.regions_data
        connected_ids_dict = self.clumpsObj.connected_ids_dict
        clump_coords_dict = self.clumpsObj.clump_coords_dict
        fil_image = self.filament_data.sum(0)
        fil_mask = self.filament_mask_2D.astype(bool)
        
        filament_centers_LBV = []
        max_path_record = []
        max_edges_record = []
        
        for index in filament_clumps_id:
            filament_centers_LBV.append([centers[index][2],centers[index][1],centers[index][0]])
        sorted_id = sorted(range(len(filament_centers_LBV)), key=lambda k: filament_centers_LBV[k], reverse=False)
        filament_centers_LBV = (np.array(filament_centers_LBV)[sorted_id])
        filament_clumps_id = np.array(filament_clumps_id)[sorted_id]
        
        Graph,Tree = FCFA.Graph_Infor_SubStructure(origin_data,fil_mask,filament_centers_LBV,filament_clumps_id,self.clumpsObj.connected_ids_dict)
        max_path_record,max_edges_record = FCFA.Get_Max_Path_Recursion(origin_data,filament_centers_LBV,max_path_record,max_edges_record,Graph,Tree)
        self.substructure_num += [len(max_path_record)]
        if Substructure:
            CalSub = True
            substructure_ids_T = []
            for subpart_id in range(0,len(max_path_record)):
                related_center_ids = np.array(filament_clumps_id)[max_path_record[subpart_id]]
                substructure_ids_T += [list(related_center_ids)]
                if type(dictionary_cuts) != type(None):
                    filamentObj_sub = FilamentInfor(self.clumpsObj,self.parameters,self.save_files,self.SkeletonType)
                    filamentObj_sub.Filament_Infor_I(related_center_ids)
                    sub_filament_data = filamentObj_sub.filament_data
                    sub_filament_mask_2D = filamentObj_sub.filament_mask_2D
                    skeleton_coords_2D = filamentObj_sub.skeleton_coords_2D
                    fil_image = sub_filament_data.sum(0)
                    fil_mask = sub_filament_mask_2D.astype(bool)
                    skeleton_coords_2D = skeleton_coords_2D+np.random.random(skeleton_coords_2D.shape)/10000
                    dictionary_cuts = FCFA.Cal_Dictionary_Cuts(SampInt,CalSub,regions_data,related_center_ids,
                             connected_ids_dict,clump_coords_dict,skeleton_coords_2D,fil_image,fil_mask,dictionary_cuts)
            self.substructure_ids += [substructure_ids_T]
        else:
            if SampInt == 1:
                dictionary_cuts = self.dc_no_sub
            else:
                skeleton_coords_2D = self.skeleton_coords_2D
                skeleton_coords_2D = skeleton_coords_2D+np.random.random(skeleton_coords_2D.shape)/10000
                dictionary_cuts = FCFA.Cal_Dictionary_Cuts(SampInt,self.CalSub,regions_data,filament_clumps_id,\
                             connected_ids_dict,clump_coords_dict,skeleton_coords_2D,fil_image,fil_mask,dictionary_cuts)

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

        filament_infor_all = self.Filament_Infor_All()
        np.savez(filament_infor_name,filament_infor_all = filament_infor_all)
        coms_vbl = self.filament_com_all
        self.filament_number = len(coms_vbl)
        if len(coms_vbl)!=0:
            filament_regions_data = self.filament_regions_data
            Filament_Table_Pix = FCFT.Table_Interface_Pix(self)
            Filament_Table_WCS = FCFT.Table_Interface_WCS(self)
            fits.writeto(mask_name, filament_regions_data, overwrite=True)
            Filament_Table_Pix.write(filament_table_pix_name,overwrite=True)  
            Filament_Table_WCS.write(filament_table_wcs_name,overwrite=True)  
            end_1 = time.time()
            end_2 = time.ctime()
            delta_time = np.around(end_1-start_1,2)
            time_record = np.hstack([[start_2, end_2, delta_time]])
            time_record = Table(time_record, names=['Start', 'End', 'DTime'])
            time_record.write(mask_name[:-9] + 'time_record.csv',overwrite=True)    
            self.delta_time = delta_time
            print('Number:', len(coms_vbl))
            print('Time:', delta_time)
            return filament_infor_all,Filament_Table_Pix,Filament_Table_WCS
        else:
            print('Number:', len(coms_vbl))
            return None,None,None



