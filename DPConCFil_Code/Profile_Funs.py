import numpy as np
import copy
from radfil import radfil_class


def Cal_Mean_Profile(filamentObj, EProfileTime=2, EProfileLen=6, ExtendRange=20):
    """
    Calculate the mean profile of a filament object by analyzing its cross-sections.
    
    This function processes filament cross-sectional profiles to generate a representative
    mean profile. It first adjusts coordinates relative to starting point, then filters
    profiles based on their length to ensure consistency. Finally, it computes the mean
    profile across all valid cross-sections.
    
    Parameters:
        filamentObj: The filament object containing profile data and coordinates
        EProfileTime: Factor to control profile selection based on standard deviation (default=2)
        EProfileLen: Minimum profile length to consider in analysis (default=6)
        ExtendRange: Extension range in pixels for profile analysis (default=20)
        
    Returns:
        None: Results are stored in the filamentObj attributes
    """
    # Create a deep copy of cuts dictionary to avoid modifying the original
    dictionary_cuts = copy.deepcopy(filamentObj.dictionary_cuts)
    start_coords = filamentObj.start_coords
    
    # Adjust coordinates relative to the starting point
    for key in ['plot_peaks', 'plot_cuts']:
        dictionary_cuts[key] = np.array(dictionary_cuts[key]) - start_coords[1:][::-1]
    
    # Adjust each point in the points list
    for i in range(len(dictionary_cuts['points'])):
        dictionary_cuts['points'][i] -= start_coords[1:][::-1]

    # Store parameters in the filament object
    filamentObj.EProfileTime = EProfileTime
    filamentObj.ExtendRange = ExtendRange
    
    # Initialize arrays for statistical analysis
    mean_arr = []
    std_arr = []
    xall_peak, yall_peak, xall_peak_eff = np.array([]), np.array([]), np.array([])
    delta_dists = []
    
    # Calculate the distance range for each profile cross-section
    for i in range(0, len(dictionary_cuts['distance'])):
        # Consider only non-zero profile points
        dists_i = dictionary_cuts['distance'][i][np.where(dictionary_cuts['profile'][i] != 0)]
        delta_dist = dists_i[-1] - dists_i[0]  # Total width of the profile
        delta_dists.append(delta_dist)
    
    # Calculate mean and standard deviation of profile widths
    mean_value = np.mean(delta_dists)
    std_value = np.std(delta_dists)
    
    # Calculate upper and lower bounds for profile width selection
    # Using a statistical approach to determine valid profile range
    dist_up = np.around(mean_value + EProfileTime * (mean_value + std_value) / std_value, 2)
    dist_down = np.around(np.max([mean_value - EProfileTime * (mean_value + std_value) / std_value, EProfileLen]), 2)
    
    # Store the calculated profile length limits
    filamentObj.MaxProfileLen = dist_up
    filamentObj.MinProfileLen = dist_down

    # Collect profile data from valid cross-sections
    profile_number = 0
    while profile_number < 4:  # Try to get at least 4 valid profiles
        for i in range(0, len(dictionary_cuts['distance'])):
            dists_i = dictionary_cuts['distance'][i][np.where(dictionary_cuts['profile'][i] != 0)]
            delta_dist = dists_i[-1] - dists_i[0]
            
            # Select profiles within the calculated width range
            if delta_dist < dist_up and delta_dist > dist_down:
                profile_number += 1
                # Concatenate valid profile data
                xall_peak = np.concatenate([xall_peak, dictionary_cuts['distance'][i]])
                yall_peak = np.concatenate([yall_peak, dictionary_cuts['profile'][i]])
        
        # Extract non-zero profile points
        xall_peak_eff = xall_peak[np.where(yall_peak != 0)]
        yall_peak_eff = yall_peak[np.where(yall_peak != 0)]
        
        # Reduce the lower bound if not enough profiles are found
        dist_down -= 1
        if dist_down < 0:
            break
    
    # Store the final EProfileLen value used
    filamentObj.EProfileLen = dist_down + 1

    # Calculate the maximum range needed for the profile analysis
    # Ensure the range covers all data points plus the extension
    max_range = np.int32(
        np.max([np.abs(np.min(xall_peak_eff) - ExtendRange), np.int32(np.max(xall_peak_eff)) + ExtendRange]))
    
    # Create uniform bins across the profile range
    bins = np.linspace(-max_range, max_range, 2 * max_range + 1)
    axis_coords = bins[:-1]  # Bin centers for profile coordinates
    
    # Calculate mean and standard deviation for each bin
    for axis_coords_i in axis_coords:
        # Select profile points within the current bin
        yall_bin_i = yall_peak[((xall_peak >= (axis_coords_i - .5 * np.diff(bins)[0])) & \
                                (xall_peak < (axis_coords_i + .5 * np.diff(bins)[0])))]
        coords_bin_i = np.where(yall_bin_i != 0)[0]
        
        # Calculate statistics if there are non-zero points in the bin
        if len(coords_bin_i) != 0:
            mean_arr.append(np.mean(yall_bin_i[coords_bin_i]))
            std_arr.append(np.std(yall_bin_i))
        else:
            # Set to zero if no valid points
            mean_arr.append(0)
            std_arr.append(0)
    
    # Replace NaN values with zeros
    mean_profile = np.nan_to_num(mean_arr, 0)
    std_arr = np.nan_to_num(std_arr, 0)
    
    # Store all calculated data in the filament object
    filamentObj.xall_peak = xall_peak
    filamentObj.yall_peak = yall_peak
    filamentObj.xall_peak_eff = xall_peak_eff
    filamentObj.yall_peak_eff = yall_peak_eff
    filamentObj.max_range = max_range
    filamentObj.axis_coords = axis_coords
    filamentObj.mean_profile = mean_profile
    filamentObj.std_arr = std_arr
    
    # Separate left and right sides of the profile for symmetry analysis
    mean_profile_left = mean_profile[1:max_range + 1]
    mean_profile_left_r = mean_profile_left[::-1]  # Reversed left profile
    mean_profile_right = mean_profile[max_range:]
    mean_profile_right_r = mean_profile_right[::-1]  # Reversed right profile
    
    # Create coordinate arrays for left and right sides
    axis_coords_left = np.linspace(-max_range + 1, 0, max_range)
    axis_coords_right = np.linspace(0, max_range - 1, max_range)
    
    # Store the left and right profile components
    filamentObj.mean_profile_left = mean_profile_left
    filamentObj.mean_profile_left_r = mean_profile_left_r
    filamentObj.mean_profile_right = mean_profile_right
    filamentObj.mean_profile_right_r = mean_profile_right_r
    filamentObj.axis_coords_left = axis_coords_left
    filamentObj.axis_coords_right = axis_coords_right


def Cal_Profile_IOU(filamentObj, MeanProfile=True):
    """
    Calculate the Intersection over Union (IOU) of the filament profile to assess symmetry.
    
    This function compares the left and right sides of the filament profile to determine
    how symmetric the profile is. It can use either the mean profile or calculate IOUs
    for each individual profile and then average them.
    
    Parameters:
        filamentObj: The filament object containing profile data
        MeanProfile: Whether to use the mean profile (True) or individual profiles (False)
        
    Returns:
        None: Results are stored in the filamentObj.profile_IOU attribute
    """
    if MeanProfile:
        # Use the precomputed mean profile for IOU calculation
        max_range = filamentObj.max_range
        mean_profile = filamentObj.mean_profile
        mean_profile_left_r = filamentObj.mean_profile_left_r  # Reversed left side
        mean_profile_right = filamentObj.mean_profile_right  # Right side
        
        # Calculate intersection and union at each point
        intersection_profile = []
        union_profile = []
        for i in range(len(mean_profile_left_r)):
            if mean_profile_left_r[i] > mean_profile_right[i]:
                # Intersection is the minimum value at each point
                intersection_profile.append(mean_profile_right[i])
                # Union is the maximum value at each point
                union_profile.append(mean_profile_left_r[i])
            else:
                intersection_profile.append(mean_profile_left_r[i])
                union_profile.append(mean_profile_right[i])
        
        # Calculate IOU as sum of intersection divided by sum of union
        profile_IOU = np.around(np.array(intersection_profile).sum() / np.array(union_profile).sum(), 2)
    else:
        # Calculate IOU for each individual profile, then average
        profile_IOU_is = []
        dictionary_cuts = filamentObj.dictionary_cuts
        
        # Process each profile separately
        for i in range(0, len(dictionary_cuts['distance'])):
            # Get non-zero profile points
            dists_i = dictionary_cuts['distance'][i][np.where(dictionary_cuts['profile'][i] != 0)]
            x_peak_i = dictionary_cuts['distance'][i]
            y_peak_i = dictionary_cuts['profile'][i]
            
            # Calculate range for this profile
            max_range_i = np.int64(np.max([np.abs(np.min(dists_i) - filamentObj.ExtendRange), \
                                         np.int64(np.max(dists_i)) + filamentObj.ExtendRange]))
            
            # Create bins for this profile
            bins = np.linspace(-max_range_i, max_range_i, 2 * max_range_i + 1)
            axis_coords = bins[:-1]
            mean_profile_i = np.zeros(2 * max_range_i)
            
            # Bin the profile data
            j = 0
            for axis_coords_i in axis_coords:
                y_bin_i = y_peak_i[((x_peak_i >= (axis_coords_i - .5 * np.diff(bins)[0])) & \
                                    (x_peak_i < (axis_coords_i + .5 * np.diff(bins)[0])))]
                coords_bin_i = np.where(y_bin_i != 0)[0]
                if len(coords_bin_i) != 0:
                    mean_profile_i[j] = np.mean(y_bin_i[coords_bin_i])
                j += 1
            
            # Separate left and right sides
            mean_profile_left_i = mean_profile_i[1:max_range_i + 1]
            mean_profile_left_r_i = mean_profile_left_i[::-1]  # Reversed left side
            mean_profile_right_i = mean_profile_i[max_range_i:]
            mean_profile_right_r_i = mean_profile_right_i[::-1]  # Reversed right side

            # Calculate intersection and union for this profile
            intersection_profile = []
            union_profile = []
            for k in range(len(mean_profile_left_r_i)):
                if mean_profile_left_r_i[k] > mean_profile_right_i[k]:
                    intersection_profile.append(mean_profile_right_i[k])
                    union_profile.append(mean_profile_left_r_i[k])
                else:
                    intersection_profile.append(mean_profile_left_r_i[k])
                    union_profile.append(mean_profile_right_i[k])
            
            # Calculate IOU for this individual profile
            profile_IOU_i = np.around(np.array(intersection_profile).sum() / np.array(union_profile).sum(), 2)
            profile_IOU_is.append(profile_IOU_i)
        
        # Average IOUs across all profiles
        profile_IOU = np.around(np.mean(profile_IOU_is), 2)
    
    # Store the final IOU value
    filamentObj.profile_IOU = profile_IOU


def Construct_radObj(filamentObj):
    """
    Construct a RadFil object for the filament to enable profile fitting.
    
    This function creates a RadFil object from the radfil package and initializes it
    with the filament image and profile data. RadFil is used for fitting analytical
    profiles to the observed filament data.
    
    Parameters:
        filamentObj: The filament object containing filament data
        
    Returns:
        None: The RadFil object is stored in filamentObj.radObj
    """
    # Sum the filament image along the first dimension to get a 2D projection
    fil_image = filamentObj.filament_item.sum(0)
    
    # Create a new RadFil object with the filament image
    radObj = radfil_class.radfil(fil_image)

    # Configure RadFil parameters
    radObj.shift = True     # Enables shifting the peak to center
    radObj.binning = False  # Disables binning of the profile
    radObj.fold = False     # Disables folding the profile
    
    # Provide profile data from the filament object
    radObj.xall = filamentObj.xall_peak
    radObj.yall = filamentObj.yall_peak

    # Store the RadFil object in the filament object
    filamentObj.radObj = radObj


def Fit_Profile(filamentObj, FitFunc='Gaussian', FitDist=None, FitMeanProfile=False, BGDist=None, BGDegree=0,
                BeamWidth=None):
    """
    Fit an analytical function to the filament profile.
    
    This function uses the RadFil object to fit either a Gaussian or Plummer function
    to the filament profile. It allows fitting either to the mean profile or to all data points.
    It also calculates the Full Width at Half Maximum (FWHM) of the fitted profile.
    
    Parameters:
        filamentObj: The filament object containing profile data
        FitFunc: The function to fit ('Gaussian' or 'Plummer', default='Gaussian')
        FitDist: Maximum distance to consider in fitting (default=None)
        FitMeanProfile: Whether to fit the mean profile (True) or all data points (False, default)
        BGDist: Distance at which to start fitting background (default=None)
        BGDegree: Polynomial degree for background fitting (default=0)
        BeamWidth: Width of the beam for convolution (default=None)
        
    Returns:
        None: Results are stored in filamentObj attributes
    """
    # Store fitting parameters in the RadFil object
    filamentObj.radObj.FitFunc = FitFunc
    filamentObj.radObj.FitDist = FitDist
    filamentObj.radObj.FitMeanProfile = FitMeanProfile
    
    # Duplicate parameters to ensure they're set correctly
    filamentObj.radObj.FitFunc = FitFunc
    filamentObj.radObj.FitDist = FitDist
    filamentObj.radObj.FitMeanProfile = FitMeanProfile

    # Set data for fitting based on whether we're using the mean profile or all points
    if FitMeanProfile:
        filamentObj.radObj.masterx = filamentObj.axis_coords
        filamentObj.radObj.mastery = filamentObj.mean_profile
    else:
        filamentObj.radObj.masterx = filamentObj.xall_peak_eff
        filamentObj.radObj.mastery = filamentObj.yall_peak_eff
    
    # Set fitting distance if not provided
    if filamentObj.radObj.FitDist is None:
        FitDist = np.int32(np.abs(filamentObj.xall_peak_eff).max() + 0.5)
    else:
        FitDist = filamentObj.radObj.FitDist
    
    # Perform the profile fitting
    filamentObj.radObj.fit_profile(fitfunc=FitFunc, fitdist=FitDist, bgdist=BGDist, bgdegree=BGDegree,
                                   beamwidth=BeamWidth)

    # Process results based on the fitting function used
    if FitFunc == 'Plummer':
        # Calculate fitted profile values at each point
        profile_fited = filamentObj.radObj.profilefit(filamentObj.axis_coords)
        
        # Calculate Full Width at Half Maximum (FWHM) and its error
        FWHM = np.abs(np.around(filamentObj.radObj.profilefit.parameters[2] * 2 * np.sqrt(2 * np.log(2)), 2))
        FWHM_error = np.around(filamentObj.radObj.std_error[2] * 2 * np.sqrt(2 * np.log(2)), 2)
        
        # Store Plummer-specific results
        filamentObj.profile_fited_P = profile_fited
        filamentObj.FWHM_P = FWHM
        filamentObj.FWHM_error_P = FWHM_error
    elif FitFunc == 'Gaussian':
        # Calculate fitted profile values at each point
        profile_fited = filamentObj.radObj.profilefit(filamentObj.axis_coords)
        
        # Calculate FWHM for Gaussian (2.355 * sigma) and its error
        FWHM = np.around(filamentObj.radObj.profilefit.parameters[2] * 2 * np.sqrt(2 * np.log(2)), 2)
        FWHM_error = np.around(filamentObj.radObj.std_error[1] * 2 * np.sqrt(2 * np.log(2)), 2)
        
        # Store Gaussian-specific results
        filamentObj.profile_fited_G = profile_fited
        filamentObj.FWHM_G = FWHM
        filamentObj.FWHM_error_G = FWHM_error