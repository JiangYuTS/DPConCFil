import time
import numpy as np

from radfil import radfil_class


def Cal_Mean_Profile(filamentObj, EProfileTime=2, EProfileLen=6, ExtendRange=20):
    dictionary_cuts = filamentObj.dictionary_cuts
    filamentObj.EProfileTime = EProfileTime
    filamentObj.ExtendRange = ExtendRange
    mean_arr = []
    std_arr = []
    xall_peak, yall_peak, xall_peak_eff = np.array([]), np.array([]), np.array([])
    delta_dists = []
    for i in range(0, len(dictionary_cuts['distance'])):
        dists_i = dictionary_cuts['distance'][i][np.where(dictionary_cuts['profile'][i] != 0)]
        delta_dist = dists_i[-1] - dists_i[0]
        delta_dists.append(delta_dist)
    mean_value = np.mean(delta_dists)
    std_value = np.std(delta_dists)
    dist_up = np.around(mean_value + EProfileTime * (mean_value + std_value) / std_value, 2)
    dist_down = np.around(np.max([mean_value - EProfileTime * (mean_value + std_value) / std_value, EProfileLen]), 2)
    filamentObj.MaxProfileLen = dist_up
    filamentObj.MinProfileLen = dist_down

    profile_number = 0
    while profile_number < 4:
        for i in range(0, len(dictionary_cuts['distance'])):
            dists_i = dictionary_cuts['distance'][i][np.where(dictionary_cuts['profile'][i] != 0)]
            delta_dist = dists_i[-1] - dists_i[0]
            if delta_dist < dist_up and delta_dist > dist_down:
                profile_number += 1
                xall_peak = np.concatenate([xall_peak, dictionary_cuts['distance'][i]])
                yall_peak = np.concatenate([yall_peak, dictionary_cuts['profile'][i]])
        xall_peak_eff = xall_peak[np.where(yall_peak != 0)]
        yall_peak_eff = yall_peak[np.where(yall_peak != 0)]
        dist_down -= 1
        if dist_down < 0:
            break
    filamentObj.EProfileLen = dist_down + 1

    max_range = np.int32(
        np.max([np.abs(np.min(xall_peak_eff) - ExtendRange), np.int32(np.max(xall_peak_eff)) + ExtendRange]))
    bins = np.linspace(-max_range, max_range, 2 * max_range + 1)
    axis_coords = bins[:-1]  # +.5*np.diff(bins)
    for axis_coords_i in axis_coords:
        yall_bin_i = yall_peak[((xall_peak >= (axis_coords_i - .5 * np.diff(bins)[0])) & \
                                (xall_peak < (axis_coords_i + .5 * np.diff(bins)[0])))]
        coords_bin_i = np.where(yall_bin_i != 0)[0]
        if len(coords_bin_i) != 0:
            mean_arr.append(np.mean(yall_bin_i[coords_bin_i]))
            std_arr.append(np.std(yall_bin_i))
        else:
            mean_arr.append(0)
            std_arr.append(0)
    mean_profile = np.nan_to_num(mean_arr, 0)
    std_arr = np.nan_to_num(std_arr, 0)
    filamentObj.xall_peak = xall_peak
    filamentObj.yall_peak = yall_peak
    filamentObj.xall_peak_eff = xall_peak_eff
    filamentObj.yall_peak_eff = yall_peak_eff
    filamentObj.max_range = max_range
    filamentObj.axis_coords = axis_coords
    filamentObj.mean_profile = mean_profile
    filamentObj.std_arr = std_arr
    mean_profile_left = mean_profile[1:max_range + 1]
    mean_profile_left_r = mean_profile_left[::-1]
    mean_profile_right = mean_profile[max_range:]
    mean_profile_right_r = mean_profile_right[::-1]
    axis_coords_left = np.linspace(-max_range + 1, 0, max_range)
    axis_coords_right = np.linspace(0, max_range - 1, max_range)
    filamentObj.mean_profile_left = mean_profile_left
    filamentObj.mean_profile_left_r = mean_profile_left_r
    filamentObj.mean_profile_right = mean_profile_right
    filamentObj.mean_profile_right_r = mean_profile_right_r
    filamentObj.axis_coords_left = axis_coords_left
    filamentObj.axis_coords_right = axis_coords_right


def Cal_Profile_IOU(filamentObj, MeanProfile=True):
    if MeanProfile:
        max_range = filamentObj.max_range
        mean_profile = filamentObj.mean_profile
        mean_profile_left_r = filamentObj.mean_profile_left_r
        mean_profile_right = filamentObj.mean_profile_right
        intersection_profile = []
        union_profile = []
        for i in range(len(mean_profile_left_r)):
            if mean_profile_left_r[i] > mean_profile_right[i]:
                intersection_profile.append(mean_profile_right[i])
                union_profile.append(mean_profile_left_r[i])
            else:
                intersection_profile.append(mean_profile_left_r[i])
                union_profile.append(mean_profile_right[i])
        profile_IOU = np.around(np.array(intersection_profile).sum() / np.array(union_profile).sum(), 2)
    else:
        profile_IOU_is = []
        dictionary_cuts = filamentObj.dictionary_cuts
        for i in range(0, len(dictionary_cuts['distance'])):
            dists_i = dictionary_cuts['distance'][i][np.where(dictionary_cuts['profile'][i] != 0)]
            x_peak_i = dictionary_cuts['distance'][i]
            y_peak_i = dictionary_cuts['profile'][i]
            max_range_i = np.int(np.max([np.abs(np.min(dists_i) - filamentObj.ExtendRange), \
                                         np.int(np.max(dists_i)) + filamentObj.ExtendRange]))
            bins = np.linspace(-max_range_i, max_range_i, 2 * max_range_i + 1)
            axis_coords = bins[:-1]
            mean_profile_i = np.zeros(2 * max_range_i)
            j = 0
            for axis_coords_i in axis_coords:
                y_bin_i = y_peak_i[((x_peak_i >= (axis_coords_i - .5 * np.diff(bins)[0])) & \
                                    (x_peak_i < (axis_coords_i + .5 * np.diff(bins)[0])))]
                coords_bin_i = np.where(y_bin_i != 0)[0]
                if len(coords_bin_i) != 0:
                    mean_profile_i[j] = np.mean(y_bin_i[coords_bin_i])
                j += 1
            mean_profile_left_i = mean_profile_i[1:max_range_i + 1]
            mean_profile_left_r_i = mean_profile_left_i[::-1]
            mean_profile_right_i = mean_profile_i[max_range_i:]
            mean_profile_right_r_i = mean_profile_right_i[::-1]

            intersection_profile = []
            union_profile = []
            for k in range(len(mean_profile_left_r_i)):
                if mean_profile_left_r_i[k] > mean_profile_right_i[k]:
                    intersection_profile.append(mean_profile_right_i[k])
                    union_profile.append(mean_profile_left_r_i[k])
                else:
                    intersection_profile.append(mean_profile_left_r_i[k])
                    union_profile.append(mean_profile_right_i[k])
            profile_IOU_i = np.around(np.array(intersection_profile).sum() / np.array(union_profile).sum(), 2)
            profile_IOU_is.append(profile_IOU_i)
        profile_IOU = np.around(np.mean(profile_IOU_is), 2)
    filamentObj.profile_IOU = profile_IOU


def Construct_radObj(filamentObj):
    fil_image = filamentObj.filament_data.sum(0)
    radObj = radfil_class.radfil(fil_image)

    radObj.shift = True
    radObj.binning = False
    radObj.fold = False
    radObj.xall = filamentObj.xall_peak
    radObj.yall = filamentObj.yall_peak

    filamentObj.radObj = radObj


def Fit_Profile(filamentObj, FitFunc='Gaussian', FitDist=None, FitMeanProfile=False, BGDist=None, BGDegree=0,
                BeamWidth=None):
    filamentObj.radObj.FitFunc = FitFunc
    filamentObj.radObj.FitDist = FitDist
    filamentObj.radObj.FitMeanProfile = FitMeanProfile
    filamentObj.radObj.FitFunc = FitFunc
    filamentObj.radObj.FitDist = FitDist
    filamentObj.radObj.FitMeanProfile = FitMeanProfile

    if FitMeanProfile:
        filamentObj.radObj.masterx = filamentObj.axis_coords
        filamentObj.radObj.mastery = filamentObj.mean_profile
    else:
        filamentObj.radObj.masterx = filamentObj.xall_peak_eff
        filamentObj.radObj.mastery = filamentObj.yall_peak_eff
    if filamentObj.radObj.FitDist is None:
        FitDist = np.int32(np.abs(filamentObj.xall_peak_eff).max() + 0.5)
    else:
        FitDist = filamentObj.radObj.FitDist
    filamentObj.radObj.fit_profile(fitfunc=FitFunc, fitdist=FitDist, bgdist=BGDist, bgdegree=BGDegree,
                                   beamwidth=BeamWidth)

    if FitFunc == 'Plummer':
        profile_fited = filamentObj.radObj.profilefit(filamentObj.axis_coords)
        FWHM = np.abs(np.around(filamentObj.radObj.profilefit.parameters[2] * 2 * np.sqrt(2 * np.log(2)), 2))
        FWHM_error = np.around(filamentObj.radObj.std_error[2] * 2 * np.sqrt(2 * np.log(2)), 2)
        filamentObj.profile_fited_P = profile_fited
        filamentObj.FWHM_P = FWHM
        filamentObj.FWHM_error_P = FWHM_error
    elif FitFunc == 'Gaussian':
        profile_fited = filamentObj.radObj.profilefit(filamentObj.axis_coords)
        FWHM = np.around(filamentObj.radObj.profilefit.parameters[2] * 2 * np.sqrt(2 * np.log(2)), 2)
        FWHM_error = np.around(filamentObj.radObj.std_error[1] * 2 * np.sqrt(2 * np.log(2)), 2)
        filamentObj.profile_fited_G = profile_fited
        filamentObj.FWHM_G = FWHM
        filamentObj.FWHM_error_G = FWHM_error

