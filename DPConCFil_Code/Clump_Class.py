import warnings
import time
import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy.table import Table
from skimage import measure, morphology
import os

import FacetClumps
from FacetClumps.Detect_Files import Detect as DF_FacetClumps
from FacetClumps.Cal_Tables_From_Mask_Funs import Cal_Tables_From_Mask as DF_FacetClumps_Mask

from . import Clump_Class_Funs


class ClumpInfor(object):
    def __init__(self, file_name=None, mask_name=None, outcat_name=None, outcat_wcs_name=None):
        self.file_name = file_name
        self.mask_name = mask_name
        self.outcat_name = outcat_name
        self.outcat_wcs_name = outcat_wcs_name

        data_header = fits.getheader(file_name)
        #         data_header.remove('VELREF')
        origin_data = fits.getdata(file_name)
        origin_data = np.squeeze(origin_data)
        origin_data[np.isnan(origin_data)] = -999
        # data_header['CTYPE3'] = 'VELO'
        data_wcs = wcs.WCS(data_header)

        self.data_header = data_header
        self.data_wcs = data_wcs
        self.origin_data = origin_data
        origin_data_shape = origin_data.shape
        self.origin_data_shape = origin_data_shape

        data_ranges_lbv = Clump_Class_Funs.Get_Data_Ranges_WCS(self.origin_data, self.data_wcs)
        self.data_ranges_lbv = data_ranges_lbv
        delta_v = (data_ranges_lbv[2][1] - data_ranges_lbv[2][0]) / origin_data_shape[0]
        self.delta_v = np.around(delta_v, 3)
        

    def Cal_Infor_From_Mask_Or_Algorithm(self, mask_or_algorithm='FacetClumps', parameters=None):
        self.parameters = parameters
        if mask_or_algorithm == 'FacetClumps':
            did_tables_FacetClumps = DF_FacetClumps(self.file_name, self.parameters, self.mask_name, self.outcat_name,
                                                    self.outcat_wcs_name)
            self.did_tables = did_tables_FacetClumps

        elif mask_or_algorithm == 'mask':
            # did_tables_mask = DF_FacetClumps_Mask(file_name,mask_name,outcat_name,outcat_wcs_name)
            did_tables_mask = Clump_Class_Funs.Detect_From_Regions(self)
            # self.did_tables = did_tables_mask

        else:
            raise TypeError("Please choose an algorithm or give a mask.")

    
    def Get_Clumps_Infor(self, sr_origin=False, fit_flag=True, ErosedK=2):
        if not os.path.exists(self.mask_name):
            print('The mask file does not exist.')
        else:
            origin_data = self.origin_data
            regions_data = fits.getdata(self.mask_name)
            outcat_table = Table.read(self.outcat_name)
            outcat_wcs_table = Table.read(self.outcat_wcs_name)

            centers = np.array([outcat_table['Cen3'] - 1, outcat_table['Cen2'] - 1, outcat_table['Cen1'] - 1]).T
            peaks = np.array([outcat_table['Peak3'] - 1, outcat_table['Peak2'] - 1, outcat_table['Peak1'] - 1]).T
            centers_wcs = np.array([outcat_wcs_table['Cen1'], outcat_wcs_table['Cen2'], outcat_wcs_table['Cen3']]).T
            edges = outcat_table['Edge']
            angles = outcat_table['Angle']

            regions_list = measure.regionprops(regions_data)
            input_data = [origin_data,regions_data]
            for i in range(2):
                regions_data_erosed = regions_data*morphology.erosion(regions_data>0, morphology.ball(ErosedK+i))
                input_data += [regions_data_erosed]
            
            if sr_origin:
                RMS, Threshold = self.parameters[0], self.parameters[1]
                srs_list_0, srs_array_0 = FacetClumps.FacetClumps_3D_Funs.Get_Regions_FacetClumps(origin_data, RMS, Threshold, np.array([0]))
                srs_list, srs_array = FacetClumps.FacetClumps_3D_Funs.Get_Regions_FacetClumps(origin_data, RMS, Threshold, srs_array_0)
                rc_dict = Clump_Class_Funs.Build_RC_Dict_Simplified(peaks, srs_array, srs_list)
            else :
                srs_array = measure.label(regions_data > 0, connectivity=3)
                srs_list = measure.regionprops(srs_array,intensity_image=origin_data)
                rc_dict = Clump_Class_Funs.Build_RC_Dict_Simplified(peaks, srs_array, srs_list)
            sr_coords_dict = {}
            for index in range(len(srs_list)):
                sr_coords = srs_list[index].coords
                sr_coords_dict[index] = sr_coords

            if fit_flag:
                centers,angles,clump_coords_dict,connected_ids_dict_lists = \
                    Clump_Class_Funs.Gaussian_Fit_Infor(input_data,regions_list,centers,edges,angles)
                if self.data_wcs.world_n_dim == 3:
                    cen1, cen2, cen3 = self.data_wcs.all_pix2world(centers[:, 2], centers[:, 1], centers[:, 0], 0)
                    centers_wcs = np.column_stack([np.around(cen1, 3), np.around(cen2, 3), np.around(cen3 / 1000, 3)])
                elif self.data_wcs.world_n_dim == 4:
                    cen1, cen2, cen3, temp_c = self.data_wcs.all_pix2world(centers[:, 2], centers[:, 1], centers[:, 0], 0,0)
                    centers_wcs = np.column_stack([np.around(cen1, 3), np.around(cen2, 3), np.around(cen3 / 1000, 3)])
                np.set_printoptions(suppress=True)
            else:
                centers_T,angles_T,clump_coords_dict,connected_ids_dict_lists = \
                    Clump_Class_Funs.Gaussian_Fit_Infor(input_data,regions_list,centers,edges,angles)
                
            self.fit_flag = fit_flag
            self.regions_data = regions_data
            self.regions_list = regions_list
            self.centers = centers
            self.centers_wcs = centers_wcs
            self.peaks = peaks
            self.edges = edges
            self.angles = angles
            self.signal_regions_array = srs_array
            self.signal_regions_list = srs_list
            self.sr_coords_dict = sr_coords_dict
            self.rc_dict = rc_dict
            self.clump_coords_dict = clump_coords_dict
            self.connected_ids_dict = connected_ids_dict_lists[0]
            self.connected_ids_dict_lists = connected_ids_dict_lists






