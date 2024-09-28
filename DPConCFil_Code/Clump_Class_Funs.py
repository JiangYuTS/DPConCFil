import time
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from skimage import measure, morphology
from scipy import optimize, linalg

from tqdm import tqdm

import FacetClumps


def Cal_Table_From_Regions(clumpsObj):
    origin_data = clumpsObj.origin_data
    regions_data = fits.getdata(clumpsObj.mask_name)
    regions_data = np.array(regions_data, dtype='int')
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
    for index in range(len(regions_list)):
        clump_coords = regions_list[index].coords
        #     clump_coords = (clump_coords[:,0],clump_coords[:,1],clump_coords[:,2])
        clump_coords_x = clump_coords[:, 0]
        clump_coords_y = clump_coords[:, 1]
        clump_coords_z = clump_coords[:, 2]
        clump_x_min, clump_x_max = clump_coords_x.min(), clump_coords_x.max()
        clump_y_min, clump_y_max = clump_coords_y.min(), clump_coords_y.max()
        clump_z_min, clump_z_max = clump_coords_z.min(), clump_coords_z.max()
        clump_item = np.zeros(
            (clump_x_max - clump_x_min + 1, clump_y_max - clump_y_min + 1, clump_z_max - clump_z_min + 1))
        clump_item[(clump_coords_x - clump_x_min, clump_coords_y - clump_y_min, clump_coords_z - clump_z_min)] = \
            origin_data[clump_coords_x, clump_coords_y, clump_coords_z]
        od_mass = origin_data[(clump_coords_x, clump_coords_y, clump_coords_z)]
        #         od_mass = od_mass - od_mass.min()
        mass_array = np.c_[od_mass, od_mass, od_mass]
        com = np.around((mass_array * clump_coords).sum(0) \
                        / od_mass.sum(), 3).tolist()
        size = np.sqrt((mass_array * (np.array(clump_coords) ** 2)).sum(0) / od_mass.sum() - \
                       ((mass_array * np.array(clump_coords)).sum(0) / od_mass.sum()) ** 2)
        clump_com.append(com)
        clump_size.append(size.tolist())
        com_item = [com[0] - clump_x_min, com[1] - clump_y_min, com[2] - clump_z_min]
        D, V, size_ratio, angle = FacetClumps.FacetClumps_3D_Funs.Get_DV(clump_item, com_item)
        clump_angle.append(angle)
        peak_coord = np.where(clump_item == clump_item.max())
        peak_coord = [(peak_coord[0] + clump_x_min)[0], (peak_coord[1] + clump_y_min)[0],
                      (peak_coord[2] + clump_z_min)[0]]
        peak_value.append(origin_data[peak_coord[0], peak_coord[1], peak_coord[2]])
        peak_location.append(peak_coord)
        clump_sum.append(od_mass.sum())
        clump_volume.append(len(clump_coords_x))
        data_size = origin_data.shape
        if clump_x_min == 0 or clump_y_min == 0 or clump_z_min == 0 or \
                clump_x_max + 1 == data_size[0] or clump_y_max + 1 == data_size[1] or clump_z_max + 1 == data_size[2]:
            clump_edge.append(1)
        else:
            clump_edge.append(0)
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
    start_1 = time.time()
    start_2 = time.ctime()
    did_table, td_outcat, td_outcat_wcs = [], [], []
    file_name, mask_name, outcat_name, outcat_wcs_name = clumpsObj.file_name, clumpsObj.mask_name, clumpsObj.outcat_name, clumpsObj.outcat_wcs_name
    origin_data = clumpsObj.origin_data
    ndim = origin_data.ndim
    if ndim == 3:
        did_table = Cal_Table_From_Regions(clumpsObj)
    else:
        raise Exception('Please check the dimensionality of the data!')
    if len(did_table['peak_value']) != 0:
        data_header = fits.getheader(file_name)
        td_outcat, td_outcat_wcs, convert_to_WCS = FacetClumps.Detect_Files_Funs.Table_Interface(did_table, data_header,
                                                                                                 ndim)
        td_outcat.write(outcat_name, overwrite=True)
        td_outcat_wcs.write(outcat_wcs_name, overwrite=True)
        print('Number:', len(did_table['peak_value']))
    else:
        print('No clumps!')
        convert_to_WCS = False
    end_1 = time.time()
    end_2 = time.ctime()
    delta_time = np.around(end_1 - start_1, 2)
    par_time_record = np.hstack([[start_2, end_2, delta_time, convert_to_WCS]])
    par_time_record = Table(par_time_record, names=['Start', 'End', 'DTime', 'CToWCS'])
    par_time_record.write(outcat_name[:-4] + '_FacetClumps_record.csv', overwrite=True)
    print('Time:', delta_time)
    did_tables = {}
    did_tables['outcat_table'] = td_outcat
    did_tables['outcat_wcs_table'] = td_outcat_wcs
    did_tables['mask'] = did_table['regions_data']
    return did_tables


def Build_RC_Dict(com, regions_array, regions_first):
    k1 = 0
    k2 = 0
    i_record = []
    temp_rc_dict = {}
    rc_dict = {}
    new_regions = []
    temp_regions_array = np.zeros_like(regions_array)
    for i in range(1, np.int64(regions_array.max() + 1)):
        temp_rc_dict[i] = []
    center = np.array(np.around(com, 0), dtype='uint16')
    for cent in center:
        if regions_array[cent[0], cent[1], cent[2]] != 0:
            temp_rc_dict[regions_array[cent[0], cent[1], cent[2]]].append(k1)
            i_record.append(regions_array[cent[0], cent[1], cent[2]])
        k1 += 1
    for i in range(1, np.int64(regions_array.max()) + 1):
        if i in i_record:
            coordinates = regions_first[i - 1].coords
            temp_regions_array[(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])] = k2 + 1
            new_regions.append(regions_first[i - 1])
            rc_dict[k2] = temp_rc_dict[i]
            k2 += 1
    return new_regions, temp_regions_array, rc_dict


def Single_Gaussian_Fit(*parameters_init):
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
    theta1 = 45 * np.pi / 180
    total = np.nansum(data)
    X, Y = np.indices(data.shape)
    center_x1 = np.nansum(X * data) / total
    center_y1 = np.nansum(Y * data) / total
    row = data[int(center_x1), :]
    col = data[:, int(center_y1)]
    width_x1 = np.nansum(np.sqrt(abs((np.arange(col.size) - center_y1) ** 2 * col)) / np.nansum(col))
    width_y1 = np.nansum(np.sqrt(abs((np.arange(row.size) - center_x1) ** 2 * row)) / np.nansum(row))
    height1 = np.nanmax(data)
    parameters_init = [height1, center_x1, center_y1, width_x1, width_y1, theta1]
    return parameters_init


def Gaussian_Fit(data):
    parameters_init = Parameters_Init(data)
    bounds_low = [parameters_init[0] / 2, parameters_init[1] - parameters_init[3], \
                  parameters_init[2] - parameters_init[4], parameters_init[3] / 2, parameters_init[4] / 2, -90]
    bounds_up = [parameters_init[0] * 2, parameters_init[1] + parameters_init[3], \
                 parameters_init[2] + parameters_init[4], parameters_init[3] * 2, parameters_init[4] * 2, 90]
    error_fun = lambda p: np.ravel(Single_Gaussian_Fit(*p)(*np.indices(data.shape)) - data)
    fit_infor = optimize.least_squares(error_fun, parameters_init, f_scale=0.01,
                                       method='trf')  # ,bounds=[bounds_low,bounds_up]
    #     fit_infor = optimize.least_squares(error_fun, parameters_init, f_scale=0.01,method = 'trf')#trf,dogbox,lm,
    k = 0
    fit_flag = 1
    while fit_infor.nfev >= 1000:
        fit_infor = optimize.least_squares(error_fun, parameters_init, f_scale=0.01, \
                                           method='{}'.format(
                                               ['trf', 'dogbox', 'lm'][random.randint(0, 3)]))  # trf,dogbox,lm
        print('Number of function evaluations done:', fit_infor.nfev)
        k += 1
        if k > 10:
            fit_flag = 0
            print('The fit has failed!')
            break
    return fit_infor, fit_flag


def Alternative_Np_Where(arr, arr_sequence, goal_value):
    goal_logic = arr == goal_value
    goal_logic_indexs = arr_sequence[np.array(goal_logic.flat)].astype('int')
    coords = np.unravel_index(goal_logic_indexs, arr.shape)
    return coords


def Clump_Items(real_data, regions_data, centers, index, regions_list, clump_coords_dict, connected_ids_dict):
    #     coords = np.where(regions_data == index+1)
    #     clump_coords = Alternative_Np_Where(regions_data,arr_sequence,index+1)
    clump_coords = regions_list[index].coords
    clump_coords_dict[index] = clump_coords
    core_x = clump_coords[:, 0]
    core_y = clump_coords[:, 1]
    core_z = clump_coords[:, 2]
    x_min = core_x.min()
    x_max = core_x.max()
    y_min = core_y.min()
    y_max = core_y.max()
    z_min = core_z.min()
    z_max = core_z.max()
    length = np.max([x_max - x_min, y_max - y_min, z_max - z_min]) + 5
    wish_len = 10
    if length < wish_len:
        length = wish_len + 5
    clump_item = np.zeros([length, length, length])
    start_x = np.int64((length - (x_max - x_min)) / 2)
    start_y = np.int64((length - (y_max - y_min)) / 2)
    start_z = np.int64((length - (z_max - z_min)) / 2)
    clump_item[core_x - x_min + start_x, core_y - y_min + start_y, core_z - z_min + start_z] = real_data[
        core_x, core_y, core_z]
    center_index = [centers[index][0] - x_min + start_x, centers[index][1] - y_min + start_y,
                    centers[index][2] - z_min + start_z]
    start_coords = [x_min - start_x, y_min - start_y, z_min - start_z]
    start_x_for_dilation = max(x_min - start_x, 0)
    start_y_for_dilation = max(y_min - start_y, 0)
    start_z_for_dilation = max(z_min - start_z, 0)
    clump_region_box = regions_data[start_x_for_dilation:start_x_for_dilation + length, \
                       start_y_for_dilation:start_y_for_dilation + length,
                       start_z_for_dilation:start_z_for_dilation + length]
    clump_item_region_box = np.zeros_like(clump_region_box)
    clump_item_region_box[
        core_x - start_x_for_dilation, core_y - start_y_for_dilation, core_z - start_z_for_dilation] = 1
    # selem = morphology.ball(1)
    clump_item_region_box_dilated = morphology.dilation(clump_item_region_box, morphology.ball(1))
    clump_region_multipled = (clump_item_region_box_dilated - clump_item_region_box) * clump_region_box
    connected_centers_id = list(set((clump_region_multipled[clump_region_multipled.astype(bool)] - 1).astype(int)))
    connected_ids_dict[index] = connected_centers_id
    return clump_item, center_index, start_coords


# def Gaussian_Fit_Infor(origin_data,regions_data,centers,edges,angles):
#     start_1 = time.time()
#     angles_fited = []
#     clump_coords_dict = {}
#     connected_ids_dict = {}
#     centers_fited = centers.copy()
#     angles_fited = angles.copy()
# #     total_pixels = len(origin_data.ravel())
# #     arr_sequence = np.linspace(0,total_pixels-1,total_pixels)
#     regions_data = np.array(regions_data,dtype='int')
#     regions_list = measure.regionprops(regions_data)
#     origin_data_shape = origin_data.shape
#     for index in tqdm(range(len(centers))):
#         clump_item,center_index,start_coords = \
#             Clump_Items(origin_data,regions_data,centers,index,regions_list,clump_coords_dict,connected_ids_dict)
#         if edges[index] == 0:
#             data = clump_item.sum(0)
#             fit_infor,fit_flag = Gaussian_Fit(data)
#             if fit_flag:
#                 parameters = fit_infor.x
#                 centers_fited_index_y = np.around(parameters[1]+start_coords[1],3)
#                 centers_fited_index_z = np.around(parameters[2]+start_coords[2],3)
#                 if centers_fited_index_y > 0 and centers_fited_index_y<origin_data_shape[1] and \
#                    centers_fited_index_z > 0 and centers_fited_index_z<origin_data_shape[2]:
#                     centers_fited[index] = [centers_fited[index][0],centers_fited_index_y,centers_fited_index_z]
#                     theta = (parameters[5]*180/np.pi)%180
#                     if parameters[3]>parameters[4]:
#                         theta -= 90
#                     elif parameters[3]<parameters[4] and theta>90:
#                         theta -= 180
#                     angles_fited[index] = np.around(theta,2)
#     end_1 = time.time()
#     delta_time = np.around(end_1-start_1,2)
#     print('Fitting Clumps Time:', delta_time)
#     return centers_fited,angles_fited,clump_coords_dict,connected_ids_dict

def Gaussian_Fit_Infor(origin_data, regions_data, centers, edges, angles):
    start_1 = time.time()
    angles_fited = []
    clump_coords_dict = {}
    connected_ids_dict = {}
    centers_fited = centers.copy()
    angles_fited = angles.copy()
    #     total_pixels = len(origin_data.ravel())
    #     arr_sequence = np.linspace(0,total_pixels-1,total_pixels)
    regions_data = np.array(regions_data, dtype='int')
    regions_list = measure.regionprops(regions_data)
    origin_data_shape = origin_data.shape
    for index in tqdm(range(len(centers))):
        clump_item, center_index, start_coords = \
            Clump_Items(origin_data, regions_data, centers, index, regions_list, clump_coords_dict, connected_ids_dict)
        if edges[index] == 0:
            data = clump_item.sum(0)
            fit_infor, fit_flag = Gaussian_Fit(data)
            data = clump_item.sum(1)
            fit_infor_1, fit_flag_1 = Gaussian_Fit(data)
            if fit_flag and fit_flag_1:
                parameters = fit_infor.x
                parameters_1 = fit_infor_1.x
                centers_fited_index_x = np.around(parameters_1[1] + start_coords[0], 3)
                centers_fited_index_y = np.around(parameters[1] + start_coords[1], 3)
                centers_fited_index_z = np.around(parameters[2] + start_coords[2], 3)
                if centers_fited_index_x > 0 and centers_fited_index_x < origin_data_shape[0] and \
                        centers_fited_index_y > 0 and centers_fited_index_y < origin_data_shape[1] and \
                        centers_fited_index_z > 0 and centers_fited_index_z < origin_data_shape[2]:
                    centers_fited[index] = [centers_fited_index_x, centers_fited_index_y, centers_fited_index_z]
                    theta = (parameters[5] * 180 / np.pi) % 180
                    if parameters[3] > parameters[4]:
                        theta -= 90
                    elif parameters[3] < parameters[4] and theta > 90:
                        theta -= 180
                    angles_fited[index] = np.around(theta, 2)
    end_1 = time.time()
    delta_time = np.around(end_1 - start_1, 2)
    print('Fitting Clumps Time:', delta_time)
    return centers_fited, angles_fited, clump_coords_dict, connected_ids_dict


def Get_Data_Ranges_WCS(origin_data, data_wcs):
    origin_data_shape = origin_data.shape
    if data_wcs.naxis == 4:
        data_ranges_start = data_wcs.all_pix2world(0, 0, 0, 0, 0)
        data_ranges_end = data_wcs.all_pix2world(origin_data_shape[2] - 1, origin_data_shape[1] - 1, \
                                                 origin_data_shape[0] - 1, 0, 0)
    elif data_wcs.naxis == 3:
        data_ranges_start = data_wcs.all_pix2world(0, 0, 0, 0)
        data_ranges_end = data_wcs.all_pix2world(origin_data_shape[2] - 1, origin_data_shape[1] - 1, \
                                                 origin_data_shape[0], 0)
    data_ranges_lbv = [[data_ranges_start[0].tolist(), data_ranges_end[0].tolist()], \
                       [data_ranges_start[1].tolist(), data_ranges_end[1].tolist()], \
                       [data_ranges_start[2] / 1000, data_ranges_end[2] / 1000]]
    return data_ranges_lbv


