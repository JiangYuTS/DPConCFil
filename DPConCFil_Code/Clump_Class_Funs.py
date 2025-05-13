import time
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from skimage import measure, morphology
from scipy import optimize
import copy
from tqdm import tqdm

import FacetClumps


def Cal_Table_From_Regions(clumpsObj):
    """
    Calculate physical properties for each identified clump region.
    
    This function extracts various properties from detected clump regions, including:
    - Peak value and location
    - Center of mass
    - Clump size/dimensions
    - Total flux (sum)
    - Volume
    - Edge status (whether clump touches data boundary)
    - Orientation angle
    
    Parameters:
    -----------
    clumpsObj : object
        An object containing the original data and mask information
        
    Returns:
    --------
    detect_infor_dict : dict
        Dictionary containing all calculated clump properties
    """
    origin_data = clumpsObj.origin_data
    regions_data = fits.getdata(clumpsObj.mask_name)
    # regions_data = np.array(regions_data, dtype='int')
    regions_list = measure.regionprops(regions_data)
    peak_value = []
    peak_location = []
    clump_com = []
    clump_size = []
    clump_sum = []
    clump_volume = []
    clump_edge = []
    clump_angle = []
    detect_infor_dict = {}
    
    # Loop through each detected region
    for index in range(len(regions_list)):
        clump_coords = regions_list[index].coords
        #     clump_coords = (clump_coords[:,0],clump_coords[:,1],clump_coords[:,2])
        clump_coords_x = clump_coords[:, 0]
        clump_coords_y = clump_coords[:, 1]
        clump_coords_z = clump_coords[:, 2]
        
        # Get the bounding box of the clump
        clump_x_min, clump_x_max = clump_coords_x.min(), clump_coords_x.max()
        clump_y_min, clump_y_max = clump_coords_y.min(), clump_coords_y.max()
        clump_z_min, clump_z_max = clump_coords_z.min(), clump_coords_z.max()
        
        # Create a local data cube for the clump
        clump_item = np.zeros(
            (clump_x_max - clump_x_min + 1, clump_y_max - clump_y_min + 1, clump_z_max - clump_z_min + 1))
        clump_item[(clump_coords_x - clump_x_min, clump_coords_y - clump_y_min, clump_coords_z - clump_z_min)] = \
            origin_data[clump_coords_x, clump_coords_y, clump_coords_z]
        
        # Extract intensity values (masses) at the clump coordinates
        od_mass = origin_data[(clump_coords_x, clump_coords_y, clump_coords_z)]
        #         od_mass = od_mass - od_mass.min()
        
        # Duplicate mass values for weighted calculations
        mass_array = np.c_[od_mass, od_mass, od_mass]
        
        # Calculate intensity-weighted center of mass
        com = np.around((mass_array * clump_coords).sum(0) \
                        / od_mass.sum(), 3).tolist()
        
        # Calculate clump size (standard deviation along each axis)
        size = np.sqrt(np.abs((mass_array * (np.array(clump_coords) ** 2)).sum(0) / od_mass.sum() - \
                       ((mass_array * np.array(clump_coords)).sum(0) / od_mass.sum()) ** 2))
        
        clump_com.append(com)
        clump_size.append(size.tolist())
        
        # Calculate orientation using principal component analysis
        com_item = [com[0] - clump_x_min, com[1] - clump_y_min, com[2] - clump_z_min]
        D, V, size_ratio, angle = FacetClumps.FacetClumps_3D_Funs.Get_DV(clump_item, com_item)
        clump_angle.append(angle)
        
        # Find peak position and value
        peak_coord = np.where(clump_item == clump_item.max())
        peak_coord = [(peak_coord[0] + clump_x_min)[0], (peak_coord[1] + clump_y_min)[0],
                      (peak_coord[2] + clump_z_min)[0]]
        peak_value.append(origin_data[peak_coord[0], peak_coord[1], peak_coord[2]])
        peak_location.append(peak_coord)
        
        # Calculate total flux and volume
        clump_sum.append(od_mass.sum())
        clump_volume.append(len(clump_coords_x))
        
        # Check if clump touches the edge of the data cube
        data_size = origin_data.shape
        if clump_x_min == 0 or clump_y_min == 0 or clump_z_min == 0 or \
                clump_x_max + 1 == data_size[0] or clump_y_max + 1 == data_size[1] or clump_z_max + 1 == data_size[2]:
            clump_edge.append(1)  # Clump touches data edge
        else:
            clump_edge.append(0)  # Clump is fully inside data
    
    # Store all calculated properties in the output dictionary
    detect_infor_dict['peak_value'] = np.around(peak_value, 3).tolist()
    detect_infor_dict['peak_location'] = peak_location
    detect_infor_dict['clump_center'] = clump_com
    detect_infor_dict['clump_size'] = np.around(clump_size, 3).tolist()
    detect_infor_dict['clump_sum'] = np.around(clump_sum, 3).tolist()
    detect_infor_dict['clump_volume'] = clump_volume
    detect_infor_dict['clump_angle'] = clump_angle
    detect_infor_dict['clump_edge'] = clump_edge
    detect_infor_dict['regions_data'] = regions_data
    return detect_infor_dict


def Detect_From_Regions(clumpsObj):
    """
    Main function to extract and save clump information from region data.
    
    This function processes 3D data to:
    1. Calculate clump properties
    2. Create catalog tables in pixel and WCS coordinates
    3. Save results to output files
    4. Record execution time and parameters
    
    Parameters:
    -----------
    clumpsObj : object
        An object containing input and output file information and the data
        
    Returns:
    --------
    did_tables : dict
        Dictionary containing output tables and region mask
    """
    start_1 = time.time()
    start_2 = time.ctime()
    did_table, td_outcat, td_outcat_wcs = [], [], []
    file_name, outcat_name, outcat_wcs_name = clumpsObj.file_name, clumpsObj.outcat_name, clumpsObj.outcat_wcs_name
    origin_data = clumpsObj.origin_data
    ndim = origin_data.ndim
    
    # Check data dimensionality - only works with 3D data
    if ndim == 3:
        did_table = Cal_Table_From_Regions(clumpsObj)
    else:
        raise Exception('Please check the dimensionality of the data!')
    
    # Create output tables if clumps were found
    if len(did_table['peak_value']) != 0:
        data_header = fits.getheader(file_name)
        # Convert from pixel to world coordinates using FITS header info
        td_outcat, td_outcat_wcs, convert_to_WCS = FacetClumps.Detect_Files_Funs.Table_Interface(did_table, data_header, ndim)
        td_outcat.write(outcat_name, overwrite=True)
        td_outcat_wcs.write(outcat_wcs_name, overwrite=True)
        print('Number:', len(did_table['peak_value']))
    else:
        print('No clumps!')
        convert_to_WCS = False
    
    # Record execution time and parameters
    end_1 = time.time()
    end_2 = time.ctime()
    delta_time = np.around(end_1 - start_1, 2)
    par_time_record = np.hstack([[start_2, end_2, delta_time, convert_to_WCS]])
    par_time_record = Table(par_time_record, names=['Start', 'End', 'DTime', 'CToWCS'])
    par_time_record.write(outcat_name[:-4] + '_FacetClumps_record.csv', overwrite=True)
    print('Time:', delta_time)
    
    # Prepare return dictionary
    did_tables = {}
    did_tables['outcat_table'] = td_outcat
    did_tables['outcat_wcs_table'] = td_outcat_wcs
    did_tables['mask'] = did_table['regions_data']
    return did_tables


def Build_RC_Dict_Simplified(com, regions_array, regions_list):
    """
    Build a dictionary mapping region IDs to centers of mass.
    
    This function creates a mapping from region ID to indices of centers of mass
    that fall within that region.
    
    Parameters:
    -----------
    com : list
        List of center of mass coordinates
    regions_array : ndarray
        Array of region IDs
    regions_list : list
        List of region properties
        
    Returns:
    --------
    rc_dict : dict
        Dictionary mapping region IDs to indices of centers contained in them
    """
    k1 = 0
    rc_dict = {}
    # Initialize dictionary with empty lists for each region
    for i in range(1, len(regions_list)+1):
        rc_dict[i] = []
    
    # Round coordinates to integers for indexing
    center = np.array(np.around(com, 0), dtype='uint16')
    
    # Assign each center to its containing region
    for cent in center:
        if regions_array[cent[0], cent[1], cent[2]] != 0:
            rc_dict[regions_array[cent[0], cent[1], cent[2]]].append(k1)
        else:
            print('Lose com:', cent)
        k1 += 1
    return rc_dict


def Single_Gaussian_Fit(*parameters_init):
    """
    Create a 2D Gaussian function with given parameters.
    
    Parameters:
    -----------
    parameters_init : list
        [amplitude, center_x, center_y, width_x, width_y, theta]
        where theta is the rotation angle in radians
    
    Returns:
    --------
    Single_Gauss : function
        A function that calculates 2D Gaussian values for given coordinates
    """
    Single_Gauss = lambda x, y: \
        parameters_init[0] * np.exp(
            -((x - parameters_init[1]) ** 2 * (np.cos(parameters_init[5]) ** 2 / (2 * parameters_init[3] ** 2) + \
                                               np.sin(parameters_init[5]) ** 2 / (2 * parameters_init[4] ** 2)) + \
              (x - parameters_init[1]) * (y - parameters_init[2]) * (
                          np.sin(2 * parameters_init[5]) / (2 * parameters_init[4] ** 2) - \
                          np.sin(2 * parameters_init[5]) / (2 * parameters_init[3] ** 2)) + \
              (y - parameters_init[2]) ** 2 * (np.sin(parameters_init[5]) ** 2 / (2 * parameters_init[3] ** 2) + \
                                               np.cos(parameters_init[5]) ** 2 / (2 * parameters_init[4] ** 2))))
    return Single_Gauss


def Parameters_Init(data):
    """
    Estimate initial parameters for Gaussian fitting.
    
    This function calculates initial estimates for:
    - Center position (intensity-weighted mean position)
    - Width in x and y (intensity-weighted mean distance from center)
    - Peak height (maximum value in data)
    - Rotation angle (default to 45 degrees)
    
    Parameters:
    -----------
    data : ndarray
        2D array of data to be fitted
        
    Returns:
    --------
    parameters_init : list
        Initial parameter estimates [height, center_x, center_y, width_x, width_y, angle]
    """
    # Default rotation angle at 45 degrees (in radians)
    theta1 = 45 * np.pi / 180
    
    # Calculate total flux for weighted calculations
    total = np.nansum(data)
    
    # Get coordinate grids
    X, Y = np.indices(data.shape)
    
    # Calculate intensity-weighted center position
    center_x1 = np.nansum(X * data) / total
    center_y1 = np.nansum(Y * data) / total
    
    # Extract 1D profiles through the center
    row = data[int(center_x1), :]
    col = data[:, int(center_y1)]
    
    # Estimate width as intensity-weighted distance from center
    width_x1 = np.nansum(np.sqrt(abs((np.arange(col.size) - center_y1) ** 2 * col)) / np.nansum(col))
    width_y1 = np.nansum(np.sqrt(abs((np.arange(row.size) - center_x1) ** 2 * row)) / np.nansum(row))
    
    # Use maximum value as height
    height1 = np.nanmax(data)
    
    parameters_init = [height1, center_x1, center_y1, width_x1, width_y1, theta1]
    return parameters_init


def Gaussian_Fit(data):
    """
    Fit a 2D Gaussian function to data.
    
    This function attempts to fit a 2D Gaussian to the provided data using 
    least squares optimization. If the fit fails, it retries with different
    optimization methods.
    
    Parameters:
    -----------
    data : ndarray
        2D array of data to fit
        
    Returns:
    --------
    fit_infor : object
        Optimization result object with fit information
    fit_flag : int
        Flag indicating whether fit was successful (1) or failed (0)
    """
    # Get initial parameter estimates
    parameters_init = Parameters_Init(data)
    
    # Set bounds for the parameters
    bounds_low = [parameters_init[0] / 2, parameters_init[1] - parameters_init[3], \
                  parameters_init[2] - parameters_init[4], parameters_init[3] / 2, parameters_init[4] / 2, -90]
    bounds_up = [parameters_init[0] * 2, parameters_init[1] + parameters_init[3], \
                 parameters_init[2] + parameters_init[4], parameters_init[3] * 2, parameters_init[4] * 2, 90]
    
    # Define error function to minimize (difference between model and data)
    error_fun = lambda p: np.ravel(Single_Gaussian_Fit(*p)(*np.indices(data.shape)) - data)
    
    # Attempt to fit using trust region reflective algorithm
    fit_infor = optimize.least_squares(error_fun, parameters_init, f_scale=0.01, method='trf')
    
    # Check if fit converged (nfev < 1000)
    k = 0
    fit_flag = 1
    while fit_infor.nfev >= 1000:
        # Note: this section has an error - random module is not imported
        # Should be: method=(['trf', 'dogbox', 'lm'][np.random.randint(0, 3)])
        fit_infor = optimize.least_squares(error_fun, parameters_init, f_scale=0.01, \
                                           method='{}'.format(['trf', 'dogbox', 'lm'][random.randint(0, 3)]))
        print('Number of function evaluations done:', fit_infor.nfev)
        k += 1
        if k > 10:
            fit_flag = 0
            print('The fit has failed!')
            break
    return fit_infor, fit_flag


def Clump_Items_Con(input_data, index, regions_list, output_dicts):
    """
    Analyze a single clump and find connected clumps.
    
    This function extracts a clump, creates a standardized box around it,
    and finds neighboring clumps that are connected to it at different
    structuring element sizes.
    
    Parameters:
    -----------
    input_data : list
        List containing the original data and region arrays
    index : int
        Index of the clump to analyze
    regions_list : list
        List of region properties
    output_dicts : list
        List of dictionaries to store output information
        
    Returns:
    --------
    clump_item : ndarray
        3D array containing the clump data
    start_coords : list
        Starting coordinates of the clump box in the original data
    output_dicts : list
        Updated dictionaries with clump coordinates and connectivity information
    """
    origin_data, regions_data = input_data[0], input_data[1]
    clump_coords_dict, connected_ids_dict_lists = output_dicts
    
    # Get coordinates of the current clump
    clump_coords = regions_list[index].coords
    clump_coords_dict[index] = clump_coords
    core_x = clump_coords[:, 0]
    core_y = clump_coords[:, 1]
    core_z = clump_coords[:, 2]
    
    # Find bounding box of the clump
    x_min = core_x.min()
    x_max = core_x.max()
    y_min = core_y.min()
    y_max = core_y.max()
    z_min = core_z.min()
    z_max = core_z.max()
    
    # Create a standardized box around the clump with padding
    length = np.max([x_max - x_min, y_max - y_min, z_max - z_min]) + 5
    wish_len = 10
    if length < wish_len:
        length = wish_len + 5
    
    # Create empty box and position clump in the center
    clump_item = np.zeros([length, length, length])
    start_x = np.int64((length - (x_max - x_min)) / 2)
    start_y = np.int64((length - (y_max - y_min)) / 2)
    start_z = np.int64((length - (z_max - z_min)) / 2)
    
    # Copy clump data to the standardized box
    clump_item[core_x - x_min + start_x, core_y - y_min + start_y, core_z - z_min + start_z] = origin_data[core_x, core_y, core_z]
    
    # Calculate starting coordinates in the original data
    start_coords = [x_min - start_x, y_min - start_y, z_min - start_z]
    
    # Adjust for boundary conditions
    start_x_for_de = max(x_min - start_x, 0)
    start_y_for_de = max(y_min - start_y, 0)
    start_z_for_de = max(z_min - start_z, 0)
    
    # Find connected clumps at different structuring element sizes
    for i in range(3):
        # Get region data with erosion level i
        regions_data_erosed = input_data[1+i]
        
        # Extract region data around the clump
        clump_region_box = regions_data_erosed[start_x_for_de:start_x_for_de + length,
                                               start_y_for_de:start_y_for_de + length,
                                               start_z_for_de:start_z_for_de + length]
        
        # Create a binary mask of the clump
        clump_item_region_box = np.zeros_like(clump_region_box)
        clump_item_region_box[core_x - start_x_for_de, core_y - start_y_for_de, core_z - start_z_for_de] = 1
        
        # Dilate the clump mask to find neighboring regions
        clump_item_region_box_dilated = morphology.dilation(clump_item_region_box, morphology.ball(1))
        
        # Find overlapping regions (connected clumps)
        clump_region_multipled = (clump_item_region_box_dilated - clump_item_region_box) * clump_region_box
        connected_centers_id = list(set((clump_region_multipled[clump_region_multipled.astype(bool)] - 1).astype(int)))
        
        # Store the list of connected clump IDs
        connected_ids_dict_lists[i][index] = connected_centers_id
        
    return clump_item, start_coords, output_dicts


def Gaussian_Fit_Infor(input_data, regions_list, centers, edges, angles,fit_flag):
    """
    Perform Gaussian fitting on all clumps to refine their parameters.
    
    This function processes each clump to:
    1. Create a standardized box around the clump
    2. Find connected neighboring clumps
    3. Perform 2D Gaussian fits on projected data to refine center and angle parameters
    
    Parameters:
    -----------
    input_data : list
        List containing original data and region arrays
    regions_list : list
        List of region properties
    centers : list
        Initial estimates of clump centers
    edges : list
        Flags indicating whether clumps touch data edges
    angles : list
        Initial estimates of clump orientation angles
        
    Returns:
    --------
    centers_fited : list
        Refined clump center positions
    angles_fited : list
        Refined clump orientation angles
    clump_coords_dict : dict
        Dictionary of clump coordinates
    connected_ids_dict_lists : list
        Lists of connected clump IDs at different scales
    """
    start_1 = time.time()
    
    # Initialize output dictionaries
    clump_coords_dict = {}
    connected_ids_dict_lists = [{},{},{}]
    output_dicts = [clump_coords_dict, connected_ids_dict_lists]
    
    # Create copies to store refined values
    centers_fited = centers.copy()
    angles_fited = angles.copy()
    origin_data_shape = input_data[0].shape
    
    # Process each clump with progress bar
    for index in tqdm(range(len(centers))):
        # Get clump data and connectivity information
        clump_item, start_coords, output_dicts = Clump_Items_Con(input_data, index, regions_list, output_dicts)
        
        # Only fit non-edge clumps (edge clumps may have incomplete data)
        if edges[index] == 0 and fit_flag:
            # Project clump to 2D by summing along z-axis
            data = clump_item.sum(0)
            fit_infor, fit_flag_1 = Gaussian_Fit(data)
            
            # Project clump to 2D by summing along y-axis
            data = clump_item.sum(1)
            fit_infor_1, fit_flag_2 = Gaussian_Fit(data)
            
            # If both fits succeeded, update center and angle
            if fit_flag_1 and fit_flag_2:
                parameters = fit_infor.x
                parameters_1 = fit_infor_1.x
                
                # Calculate refined center in original data coordinates
                centers_fited_index_x = np.around(parameters_1[1] + start_coords[0], 3)
                centers_fited_index_y = np.around(parameters[1] + start_coords[1], 3)
                centers_fited_index_z = np.around(parameters[2] + start_coords[2], 3)
                
                # Check if center is inside data boundaries
                if centers_fited_index_x > 0 and centers_fited_index_x < origin_data_shape[0] and \
                        centers_fited_index_y > 0 and centers_fited_index_y < origin_data_shape[1] and \
                        centers_fited_index_z > 0 and centers_fited_index_z < origin_data_shape[2]:
                    
                    # Update center coordinates
                    centers_fited[index] = [centers_fited_index_x, centers_fited_index_y, centers_fited_index_z]
                    
                    # Calculate orientation angle
                    theta = (parameters[5] * 180 / np.pi) % 180
                    if parameters[3] > parameters[4]:
                        theta -= 90
                    elif parameters[3] < parameters[4] and theta > 90:
                        theta -= 180
                    
                    # Update angle
                    angles_fited[index] = np.around(theta, 2)
    
    end_1 = time.time()
    delta_time = np.around(end_1 - start_1, 2)
    print('Fitting Clumps Time:', delta_time)
    
    return centers_fited, angles_fited, clump_coords_dict, connected_ids_dict_lists


def Get_Data_Ranges_WCS(origin_data, data_wcs):
    """
    Calculate the physical (world coordinate) ranges of the data cube.
    
    This function converts the pixel boundaries of the data cube into
    physical coordinates (typically Galactic longitude, latitude, and velocity).
    
    Parameters:
    -----------
    origin_data : ndarray
        Original data cube
    data_wcs : WCS object
        World Coordinate System object from the FITS header
        
    Returns:
    --------
    data_ranges_lbv : list
        List of physical coordinate ranges [longitude, latitude, velocity]
    """
    origin_data_shape = origin_data.shape
    
    # Handle 4D WCS (common in radio astronomy data cubes)
    if data_wcs.naxis == 4:
        # Convert corner pixels to world coordinates
        data_ranges_start = data_wcs.all_pix2world(0, 0, 0, 0, 0)
        data_ranges_end = data_wcs.all_pix2world(origin_data_shape[2] - 1, origin_data_shape[1] - 1, \
                                                 origin_data_shape[0] - 1, 0, 0)
    # Handle 3D WCS
    elif data_wcs.naxis == 3:
        data_ranges_start = data_wcs.all_pix2world(0, 0, 0, 0)
        data_ranges_end = data_wcs.all_pix2world(origin_data_shape[2] - 1, origin_data_shape[1] - 1, \
                                                 origin_data_shape[0] - 1, 0)
    
    # Format coordinate ranges as [longitude, latitude, velocity]
    # Note: velocity is converted from m/s to km/s
    data_ranges_lbv = [[data_ranges_start[0].tolist(), data_ranges_end[0].tolist()], \
                       [data_ranges_start[1].tolist(), data_ranges_end[1].tolist()], \
                       [data_ranges_start[2] / 1000, data_ranges_end[2] / 1000]]
    
    return data_ranges_lbv