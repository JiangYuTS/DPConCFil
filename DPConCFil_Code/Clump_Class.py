import warnings
import time
import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy.table import Table
from skimage import measure, morphology
import os

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
        data_header['CTYPE3'] = 'VELO'
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
            #             did_tables_mask = DF_FacetClumps_Mask(file_name,mask_name,outcat_name,outcat_wcs_name)
            did_tables_mask = Clump_Class_Funs.Detect_From_Regions(self)
            self.did_tables = did_tables_mask

        else:
            raise TypeError("Please choose an algorithm or give a mask.")

    def Get_Clumps_Infor(self, fit_flag=True):
        if not os.path.exists(self.mask_name):
            print('The mask file does not exist.')
        else:
            regions_data = fits.getdata(self.mask_name)
            outcat_table = Table.read(self.outcat_name)
            outcat_wcs_table = Table.read(self.outcat_wcs_name)

            centers = np.array([outcat_table['Cen3'] - 1, outcat_table['Cen2'] - 1, outcat_table['Cen1'] - 1]).T
            peaks = np.array([outcat_table['Peak3'] - 1, outcat_table['Peak2'] - 1, outcat_table['Peak1'] - 1]).T
            centers_wcs = np.array([outcat_wcs_table['Cen1'], outcat_wcs_table['Cen2'], outcat_wcs_table['Cen3']]).T
            edges = outcat_table['Edge']
            angles = outcat_table['Angle']

            regions_label = measure.label(regions_data > 0, connectivity=3)
            regions = measure.regionprops(regions_label)
            new_regions, temp_regions_array, rc_dict = Clump_Class_Funs.Build_RC_Dict(peaks, regions_label, regions)

            if fit_flag:
                centers, angles, clump_coords_dict, connected_ids_dict = \
                    Clump_Class_Funs.Gaussian_Fit_Infor(self.origin_data, regions_data, centers, edges, angles)
                if self.data_wcs.world_n_dim == 3:
                    cen1, cen2, cen3 = self.data_wcs.all_pix2world(centers[:, 2], centers[:, 1], centers[:, 0], 0)
                    centers_wcs = np.column_stack([np.around(cen1, 3), np.around(cen2, 3), np.around(cen3 / 1000, 3)])
                elif self.data_wcs.world_n_dim == 4:
                    cen1, cen2, cen3, temp_c = self.data_wcs.all_pix2world(centers[:, 2], centers[:, 1], centers[:, 0], 0,
                                                                           0)
                    centers_wcs = np.column_stack([np.around(cen1, 3), np.around(cen2, 3), np.around(cen3 / 1000, 3)])
                np.set_printoptions(suppress=True)
            else:
                centers_T, angles_T, clump_coords_dict, connected_ids_dict = \
                    Clump_Class_Funs.Gaussian_Fit_Infor(self.origin_data, regions_data, centers, edges, angles)
            self.fit_flag = fit_flag
            self.regions_data = regions_data
            self.centers = centers
            self.centers_wcs = centers_wcs
            self.edges = edges
            self.angles = angles
            self.rc_dict = rc_dict
            self.clump_coords_dict = clump_coords_dict
            self.connected_ids_dict = connected_ids_dict

