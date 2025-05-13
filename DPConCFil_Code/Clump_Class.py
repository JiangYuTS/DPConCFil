import warnings
import time
import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy.table import Table
from skimage import measure, morphology
import os

# Import the FacetClumps package - specialized for detecting clumps in astronomical data
import FacetClumps
from FacetClumps.Detect_Files import Detect as DF_FacetClumps
from FacetClumps.Cal_Tables_From_Mask_Funs import Cal_Tables_From_Mask as DF_FacetClumps_Mask

# Import local module for clump classification functions
from . import Clump_Class_Funs


class ClumpInfor(object):
    """
    Class for analyzing and extracting information about clumps in astronomical data.
    This class handles FITS file processing, clump detection, and property calculation.
    """
    def __init__(self, file_name=None, mask_name=None, outcat_name=None, outcat_wcs_name=None):
        """
        Initialize the ClumpInfor object with file paths and load initial data.
        
        Parameters:
        -----------
        file_name : str
            Path to the input FITS data file containing astronomical observations
        mask_name : str
            Path to save or load the mask file that identifies clump regions
        outcat_name : str
            Path to save the output catalog in pixel coordinates
        outcat_wcs_name : str
            Path to save the output catalog in world (astronomical) coordinates
        """
        self.file_name = file_name
        self.mask_name = mask_name
        self.outcat_name = outcat_name
        self.outcat_wcs_name = outcat_wcs_name

        # Load the FITS header for coordinate information
        data_header = fits.getheader(file_name)
        # Load the data and preprocess it
        origin_data = fits.getdata(file_name)
        # Remove any extra dimensions (e.g., if data is 4D but we need 3D)
        origin_data = np.squeeze(origin_data)
        # Replace NaN values with -999 to avoid computational issues
        origin_data[np.isnan(origin_data)] = -999
        # Create WCS object for coordinate transformations between pixel and sky coordinates
        data_wcs = wcs.WCS(data_header)

        # Store essential data as object attributes
        self.data_header = data_header
        self.data_wcs = data_wcs
        self.origin_data = origin_data
        origin_data_shape = origin_data.shape
        self.origin_data_shape = origin_data_shape

        # Calculate the data ranges in l,b,v (galactic longitude, latitude, velocity) coordinates
        data_ranges_lbv = Clump_Class_Funs.Get_Data_Ranges_WCS(self.origin_data, self.data_wcs)
        self.data_ranges_lbv = data_ranges_lbv
        # Calculate velocity resolution (delta_v) - the step size in velocity dimension
        delta_v = (data_ranges_lbv[2][1] - data_ranges_lbv[2][0]) / origin_data_shape[0]
        # Round to 3 decimal places for readability
        self.delta_v = np.around(delta_v, 3)
        

    def Cal_Infor_From_Mask_Or_Algorithm(self, mask_or_algorithm='FacetClumps', parameters=None):
        """
        Calculate clump information either from a predefined mask or using a detection algorithm.
        
        Parameters:
        -----------
        mask_or_algorithm : str
            Either 'FacetClumps' to use the FacetClumps algorithm or 'mask' to use a predefined mask
        parameters : list
            Parameters for the chosen algorithm, typically [RMS, Threshold] for FacetClumps
            where RMS is the noise level and Threshold is detection significance level
        """
        self.parameters = parameters
        if mask_or_algorithm == 'FacetClumps':
            # Use the FacetClumps algorithm to detect clumps and generate tables
            # This creates a mask file and catalogs in both pixel and WCS coordinates
            did_tables_FacetClumps = DF_FacetClumps(self.file_name, self.parameters, self.mask_name, self.outcat_name,
                                                    self.outcat_wcs_name)
            self.did_tables = did_tables_FacetClumps

        elif mask_or_algorithm == 'mask':
            # Use a predefined mask to identify regions and generate tables
            did_tables_mask = Clump_Class_Funs.Detect_From_Regions(self)
            # # self.did_tables = did_tables_mask 

        else:
            # Raise error if invalid option is provided
            raise TypeError("Please choose an algorithm or give a mask.")

    
    def Get_Clumps_Infor(self, sr_origin=False, fit_flag=True, ErosedK=2):
        """
        Extract detailed information about detected clumps from mask data.
        
        Parameters:
        -----------
        sr_origin : bool
            If True, generate signal regions using the FacetClumps algorithm;
            if False, generate signal regions directly from the mask
        fit_flag : bool
            If True, perform Gaussian fitting to refine clump properties
        ErosedK : int
            Erosion kernel size for morphological operations, controls detail level
        """
        # Check if mask file exists
        if not os.path.exists(self.mask_name):
            print('The mask file does not exist.')
        else:
            # Load necessary data
            origin_data = self.origin_data
            regions_data = fits.getdata(self.mask_name)  # Load the mask identifying different regions
            # Load catalog tables with clump information
            outcat_table = Table.read(self.outcat_name)       # In pixel coordinates
            outcat_wcs_table = Table.read(self.outcat_wcs_name)  # In world coordinates

            # Extract clump centers, peaks and properties from tables
            # Note: Adjusting indices by -1 to convert from 1-based to 0-based indexing
            # The 3 dimensions are typically velocity, latitude, longitude in astronomy
            centers = np.array([outcat_table['Cen3'] - 1, outcat_table['Cen2'] - 1, outcat_table['Cen1'] - 1]).T
            peaks = np.array([outcat_table['Peak3'] - 1, outcat_table['Peak2'] - 1, outcat_table['Peak1'] - 1]).T
            centers_wcs = np.array([outcat_wcs_table['Cen1'], outcat_wcs_table['Cen2'], outcat_wcs_table['Cen3']]).T
            edges = outcat_table['Edge']  # Edge flags for regions near data boundaries
            angles = outcat_table['Angle']  # Orientation angles of clumps

            # Get region properties from the mask using scikit-image
            regions_list = measure.regionprops(regions_data)
            
            # Create a list with original and eroded versions of the regions for multi-scale analysis
            input_data = [origin_data, regions_data]
            for i in range(2):
                # Apply morphological erosion to the regions mask using a 3D ball structuring element
                # This helps isolate the core parts of each clump
                regions_data_erosed = regions_data * morphology.erosion(regions_data > 0, morphology.ball(ErosedK + i))
                input_data += [regions_data_erosed]
            
            if sr_origin:
                # Generate signal regions using the FacetClumps algorithm
                RMS, Threshold = self.parameters[0], self.parameters[1]
                # Initial signal region detection - creates first pass of regions
                srs_list_0, srs_array_0 = FacetClumps.FacetClumps_3D_Funs.Get_Regions_FacetClumps(
                    origin_data, RMS, Threshold, np.array([0]))
                # Refined signal region detection using the initial regions as input
                srs_list, srs_array = FacetClumps.FacetClumps_3D_Funs.Get_Regions_FacetClumps(
                    origin_data, RMS, Threshold, srs_array_0)
                # Build mapping between regions and clumps
                rc_dict = Clump_Class_Funs.Build_RC_Dict_Simplified(peaks, srs_array, srs_list)
            else:
                # Generate signal regions directly from the mask
                # Label connected regions with unique IDs (connectivity=3 for 3D data)
                srs_array = measure.label(regions_data > 0, connectivity=3)
                # Calculate properties of each labeled region
                srs_list = measure.regionprops(srs_array, intensity_image=origin_data)
                # Map regions to clumps
                rc_dict = Clump_Class_Funs.Build_RC_Dict_Simplified(peaks, srs_array, srs_list)
            
            # Create a dictionary of signal region coordinates for easier access
            sr_coords_dict = {}
            for index in range(len(srs_list)):
                sr_coords = srs_list[index].coords  # Get pixel coordinates for each region
                sr_coords_dict[index] = sr_coords

            if fit_flag:
                # Perform Gaussian fitting to refine clump properties
                # This often produces more accurate centers and shape parameters
                centers, angles, clump_coords_dict, connected_ids_dict_lists = \
                    Clump_Class_Funs.Gaussian_Fit_Infor(input_data, regions_list, centers, edges, angles,fit_flag)
                
                # Convert refined centers from pixel to world coordinates
                if self.data_wcs.world_n_dim == 3:
                    # For 3D WCS (typical for spectral cubes)
                    cen1, cen2, cen3 = self.data_wcs.all_pix2world(centers[:, 2], centers[:, 1], centers[:, 0], 0)
                    # Scale velocity (cen3) by 1000 to convert from km/s to m/s
                    centers_wcs = np.column_stack([np.around(cen1, 3), np.around(cen2, 3), np.around(cen3 / 1000, 3)])
                elif self.data_wcs.world_n_dim == 4:
                    # For 4D WCS (with an additional dimension, often time)
                    cen1, cen2, cen3, temp_c = self.data_wcs.all_pix2world(centers[:, 2], centers[:, 1], centers[:, 0], 0, 0)
                    centers_wcs = np.column_stack([np.around(cen1, 3), np.around(cen2, 3), np.around(cen3 / 1000, 3)])
                # Ensure floating-point values are displayed without scientific notation
                np.set_printoptions(suppress=True)
            else:
                # If not fitting, still get the clump coordinates and connection information
                # but don't update the centers and angles
                centers_T, angles_T, clump_coords_dict, connected_ids_dict_lists = \
                    Clump_Class_Funs.Gaussian_Fit_Infor(input_data, regions_list, centers, edges, angles,fit_flag)
            
            # Store all calculated properties as object attributes for later use
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