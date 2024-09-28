import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy import units as u
import copy

from . import Filament_Class_Funs_Analysis as FCFA


def Plot_Origin_Data(clumpsObj, figsize=(8, 6), fontsize=12, spacing=12 * u.arcmin, save_path=None):
    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_subplot(111, projection=clumpsObj.data_wcs.celestial)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.color'] = 'green'
    plt.rcParams['ytick.color'] = 'green'
    plt.xlabel("Galactic Longitude", fontsize=fontsize)
    plt.ylabel("Galactic Latitude", fontsize=fontsize)

    lon = ax0.coords[0]
    lat = ax0.coords[1]
    lon.set_major_formatter("d.d")
    lat.set_major_formatter("d.d")
    lon.set_ticks(spacing=spacing)

    gci = plt.imshow(clumpsObj.origin_data.sum(axis=0) * clumpsObj.delta_v, cmap='gray')
    cbar = plt.colorbar(gci, pad=0)
    cbar.set_label('K km s$^{-1}$', fontsize=fontsize, color='black')
    cbar.ax.tick_params(axis='y', colors='black')
    if save_path != None:
        plt.savefig(save_path, format='pdf', dpi=1000)
    plt.show()


def Plot_Clumps_Infor(clumpsObj, figsize=(8, 6), line_scale=3, save_path=None):
    centers = clumpsObj.centers
    angles = clumpsObj.angles
    edges = clumpsObj.edges
    clumps_data = np.zeros_like(clumpsObj.origin_data)
    for i in range(len(clumpsObj.clump_coords_dict)):
        clump_coords = (clumpsObj.clump_coords_dict[i][:, 0], clumpsObj.clump_coords_dict[i][:, 1], \
                        clumpsObj.clump_coords_dict[i][:, 2])
        clumps_data[clump_coords] = clumpsObj.origin_data[clump_coords]
    fig, (ax0) = plt.subplots(1, 1, figsize=figsize)
    for index in range(len(centers)):
        center_x = centers[index][1]
        center_y = centers[index][2]
        cen_x1 = center_x + line_scale * np.sin(np.deg2rad(angles[index]))
        cen_y1 = center_y + line_scale * np.cos(np.deg2rad(angles[index]))
        cen_x2 = center_x - line_scale * np.sin(np.deg2rad(angles[index]))
        cen_y2 = center_y - line_scale * np.cos(np.deg2rad(angles[index]))
        if edges[index] == 0:
            lines = plt.plot([cen_y1, center_y, cen_y2], [cen_x1, center_x, cen_x2])
            plt.setp(lines[0], linewidth=2, color='red', marker='.', markersize=3)
        ax0.plot(center_y, center_x, 'r*', markersize=6)
    #         ax0.text(center_y,center_x,"{}".format(index),color='r',fontsize=10)
    ax0.imshow(clumps_data.sum(0),
               origin='lower',
               cmap='gray',
               interpolation='none')
    ax0.contourf(clumps_data.sum(0),
                 levels=[0., .1],
                 colors='w')
    fig.tight_layout()
    plt.xticks([]), plt.yticks([])
    if save_path != None:
        plt.savefig(save_path, format='pdf', dpi=1000)
    plt.show()


def Plot_Filament_Item(filamentObj, figsize=(8, 6), fontsize=12, spacing=12 * u.arcmin, save_path=None):
    # filament_com = filamentObj.filament_com
    filament_com_wcs = filamentObj.filament_com_wcs
    filament_ratio = filamentObj.filament_ratio
    filament_angle = filamentObj.filament_angle
    filament_item = filamentObj.filament_item
    start_coords = filamentObj.start_coords
    data_wcs = filamentObj.data_wcs_item
    dictionary_cuts_item = copy.deepcopy(filamentObj.dictionary_cuts)
    for key in ['plot_peaks', 'plot_cuts']:
        dictionary_cuts_item[key] = np.array(dictionary_cuts_item[key]) - start_coords[1:][::-1]
    for i in range(len(dictionary_cuts_item['points'])):
        dictionary_cuts_item['points'][i] -= start_coords[1:][::-1]

    filament_item_shape = filament_item.shape

    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_subplot(111, projection=data_wcs.celestial)
    points_array = dictionary_cuts_item['points'][0]
    for i in range(1, len(dictionary_cuts_item['points'])):
        points_array = np.r_[points_array, dictionary_cuts_item['points'][i]]
    peaks_array = np.asarray(dictionary_cuts_item['plot_peaks'])
    dist_matrix = FCFA.Dists_Array(points_array, peaks_array)
    pp_distance = np.diag(dist_matrix)
    mask_width_mean = np.mean(dictionary_cuts_item['mask_width'])
    for cut_line_id in range(len(dictionary_cuts_item['plot_cuts'])):
        if pp_distance[cut_line_id] < mask_width_mean / 2:
            start = dictionary_cuts_item['plot_cuts'][cut_line_id][0]
            end = dictionary_cuts_item['plot_cuts'][cut_line_id][1]
            ax0.plot([start[0], end[0]], [start[1], end[1]], 'r-', markersize=8., linewidth=1., alpha=1)

    for points in dictionary_cuts_item['points']:
        ax0.plot(points[:, 0], points[:, 1], 'r', label='fit', lw=3, alpha=1.0, markersize=8.)
    ax0.plot(np.asarray(dictionary_cuts_item['plot_peaks'])[:, 0].astype(int),
             np.asarray(dictionary_cuts_item['plot_peaks'])[:, 1].astype(int),
             'b.', markersize=8., alpha=0.75, markeredgecolor='white', markeredgewidth=0.5)

    vmin = np.min(filament_item.sum(0)[np.where(filament_item.sum(0) != 0)])
    vmax = np.nanpercentile(filament_item.sum(0)[np.where(filament_item.sum(0) != 0)], 98.)
    ax0.imshow(filament_item.sum(0),
               origin='lower',
               cmap='gray',
               interpolation='none',
               norm=colors.Normalize(vmin=vmin, vmax=vmax))
    ax0.contourf(filament_item.sum(0),
                 levels=[0., .01],
                 colors='w')

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.color'] = 'red'
    plt.rcParams['ytick.color'] = 'red'
    plt.xlabel("Galactic Longitude", fontsize=fontsize)
    plt.ylabel("Galactic Latitude", fontsize=fontsize)
    lon = ax0.coords[0]
    lat = ax0.coords[1]
    lon.set_major_formatter("d.d")
    lat.set_major_formatter("d.d")
    if spacing != None:
        lon.set_ticks(spacing=spacing)

    xmin_coord = filament_item.shape[2] / 30
    ymax_coord = filament_item.shape[1] / 1.08
    ax0.text(xmin_coord, ymax_coord, r'Center: [{}$\degree$,{}$\degree$,{}km/s]'. \
             format(filament_com_wcs[0], filament_com_wcs[1], filament_com_wcs[2]), color='black', fontsize=fontsize)
    ax0.text(xmin_coord, ymax_coord - filament_item_shape[1] / 15, r'$\theta={}\degree$'. \
             format(np.around(filament_angle, 2)), color='black', fontsize=fontsize)
    ax0.text(xmin_coord, ymax_coord - filament_item_shape[1] / 8, r'LWRatio={}'. \
             format(np.around(filament_ratio, 2)), color='black', fontsize=fontsize)
    #     ax0.set_title('Filament Item Image',fontsize=14,color='b')
    #     fig.tight_layout()
    #     plt.xticks([]),plt.yticks([])
    if save_path != None:
        sava_name = save_path + '_l{}_b{}_v{}.png'. \
            format(filament_com_wcs[0], filament_com_wcs[1], filament_com_wcs[2])
        plt.savefig(sava_name)
    plt.show()


def Plot_Filament(filamentObj, figsize=(8, 6), fontsize=12, spacing=12 * u.arcmin, save_path=None):
    data_wcs = filamentObj.clumpsObj.data_wcs
    origin_data = filamentObj.clumpsObj.origin_data
    # regions_data = filamentObj.clumpsObj.regions_data
    filament_regions_data = filamentObj.filament_regions_data
    filaments_data = np.zeros_like(origin_data)
    filaments_data[filament_regions_data > 0] = origin_data[filament_regions_data > 0]
    dictionary_cuts = filamentObj.dictionary_cuts

    fig = plt.figure(figsize=(8, 6))
    ax0 = fig.add_subplot(111, projection=data_wcs.celestial)
    #     for cut_line_id in range(len(dictionary_cuts['plot_cuts'])):
    #         start = dictionary_cuts['plot_cuts'][cut_line_id][0]
    #         end = dictionary_cuts['plot_cuts'][cut_line_id][1]
    #         ax0.plot([start[0], end[0]], [start[1], end[1]], 'r-', markersize = 8.,linewidth = 1.,alpha=1)
    for points in dictionary_cuts['points']:
        ax0.plot(points[:, 0], points[:, 1], 'r', label='fit', lw=3, alpha=1.0, markersize=8.)
    #     ax0.plot(np.asarray(dictionary_cuts['plot_peaks'])[:, 0].astype(int),
    #              np.asarray(dictionary_cuts['plot_peaks'])[:, 1].astype(int),
    #              'b.', markersize = 8.,alpha=0.75, markeredgecolor='white',markeredgewidth=0.5)
    vmin = np.min(filaments_data.sum(0)[np.where(filaments_data.sum(0) != 0)])
    vmax = np.nanpercentile(filaments_data.sum(0)[np.where(filaments_data.sum(0) != 0)], 98.)
    ax0.imshow(filaments_data.sum(0),
               origin='lower',
               cmap='gray',
               interpolation='none',
               norm=colors.Normalize(vmin=vmin, vmax=vmax))
    ax0.contourf(filaments_data.sum(0),
                 levels=[0., .01],
                 colors='w')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.color'] = 'red'
    plt.rcParams['ytick.color'] = 'red'
    plt.xlabel("Galactic Longitude", fontsize=fontsize)
    plt.ylabel("Galactic Latitude", fontsize=fontsize)
    lon = ax0.coords[0]
    lat = ax0.coords[1]
    lon.set_major_formatter("d.d")
    lat.set_major_formatter("d.d")
    lon.set_ticks(spacing=spacing)

    if save_path != None:
        #         filament_data_shape = filaments_data.shape
        #         ax0.text(filament_data_shape[2]/20,filament_data_shape[1]-filament_data_shape[1]/15,'{}'.\
        #          format(save_path.split('/')[-1].split('_')[0]),color='black',fontsize=fontsize)
        sava_name = save_path
        plt.savefig(sava_name)
    plt.show()


def Get_Data_Item_Ranges_WCS(filamentObj, data_ranges_lbv):
    filament_data = filamentObj.filament_data
    filament_item = filamentObj.filament_item
    start_coords = filamentObj.start_coords
    origin_data_shape = filament_data.shape
    filament_item_shape = filament_item.shape
    delta_l = (data_ranges_lbv[0][1] - data_ranges_lbv[0][0]) / origin_data_shape[2]
    delta_b = (data_ranges_lbv[1][1] - data_ranges_lbv[1][0]) / origin_data_shape[1]
    delta_v = (data_ranges_lbv[2][1] - data_ranges_lbv[2][0]) / origin_data_shape[0]
    data_item_ranges_l = [data_ranges_lbv[0][0] + delta_l * start_coords[2],
                          data_ranges_lbv[0][0] + delta_l * (start_coords[2] + filament_item_shape[2])]
    data_item_ranges_b = [data_ranges_lbv[1][0] + delta_b * start_coords[1],
                          data_ranges_lbv[1][0] + delta_b * (start_coords[1] + filament_item_shape[1])]
    data_item_ranges_v = [data_ranges_lbv[2][0] + delta_v * start_coords[0],
                          data_ranges_lbv[2][0] + delta_v * (start_coords[0] + filament_item_shape[0])]
    data_item_ranges_lbv = [data_item_ranges_l, data_item_ranges_b, data_item_ranges_v]
    return data_item_ranges_lbv


def Get_WCS_Ticks(data_ranges, delta_sign, delta_interval, decimal_digits):
    data_ranges_start = data_ranges[0]
    data_ranges_end = data_ranges[1]
    ticks = []
    ticks_num = 0
    tick_start = data_ranges_start
    while True:
        tick_item = np.around(np.around(tick_start + delta_sign * 5 / 10 ** (decimal_digits + 1), decimal_digits + 2),
                              decimal_digits)
        tick_start += delta_sign * delta_interval
        if delta_sign * tick_item > delta_sign * data_ranges_end:
            break
        if decimal_digits == 0:
            tick_item = int(tick_item)
        ticks.append(tick_item)
    return ticks


def Get_Pix_Ticks(wcs_ticks, data_wcs):
    wcs_ticks_l = wcs_ticks[0]
    wcs_ticks_b = wcs_ticks[1]
    wcs_ticks_v = wcs_ticks[2]
    pix_ticks_l = []
    for wcs_ticks_i in wcs_ticks_l:
        if data_wcs.naxis == 4:
            pix_ticks_i = data_wcs.all_world2pix(wcs_ticks_i, 0, 0, 0, 0)
        elif data_wcs.naxis == 3:
            pix_ticks_i = data_wcs.all_world2pix(wcs_ticks_i, 0, 0, 0)
        pix_ticks_l.append(round(pix_ticks_i[0].tolist()))

    pix_ticks_b = []
    for wcs_ticks_i in wcs_ticks_b:
        if data_wcs.naxis == 4:
            pix_ticks_i = data_wcs.all_world2pix(0, wcs_ticks_i, 0, 0, 0)
        elif data_wcs.naxis == 3:
            pix_ticks_i = data_wcs.all_world2pix(0, wcs_ticks_i, 0, 0)
        pix_ticks_b.append(round(pix_ticks_i[1].tolist()))

    pix_ticks_v = []
    for wcs_ticks_i in wcs_ticks_v:
        if data_wcs.naxis == 4:
            pix_ticks_i = data_wcs.all_world2pix(0, 0, wcs_ticks_i * 1000, 0, 0)
        elif data_wcs.naxis == 3:
            pix_ticks_i = data_wcs.all_world2pix(0, 0, wcs_ticks_i * 1000, 0)
        pix_ticks_v.append(round(pix_ticks_i[2].tolist()))
    pix_ticks_lbv = [pix_ticks_l, pix_ticks_b, pix_ticks_v]
    return pix_ticks_lbv


def Plot_PV_Integrate(filamentObj, figsize=(10, 8), fontsize=12, spacing=[0.2, 0.2, None], save_path=None):
    data_wcs = filamentObj.clumpsObj.data_wcs
    origin_data = filamentObj.clumpsObj.origin_data
    data_ranges_lbv = filamentObj.clumpsObj.data_ranges_lbv
    data_item_ranges_lbv = Get_Data_Item_Ranges_WCS(filamentObj, data_ranges_lbv)

    if spacing[2] == None:
        delta_interval_v = np.around((data_item_ranges_lbv[2][1] - data_item_ranges_lbv[2][0]) / 5 + 0.5, 0)
    else:
        delta_interval_v = spacing[2]
    wcs_ticks_l = Get_WCS_Ticks(data_item_ranges_lbv[0], delta_sign=-1, delta_interval=spacing[0], decimal_digits=1)
    wcs_ticks_b = Get_WCS_Ticks(data_item_ranges_lbv[1], delta_sign=1, delta_interval=spacing[1], decimal_digits=1)
    wcs_ticks_v = Get_WCS_Ticks(data_item_ranges_lbv[2], delta_sign=1, delta_interval=delta_interval_v,
                                decimal_digits=0)
    wcs_ticks_lbv = [wcs_ticks_l, wcs_ticks_b, wcs_ticks_v]
    pix_ticks_lbv = Get_Pix_Ticks(wcs_ticks_lbv, data_wcs)

    filament_item = filamentObj.filament_item
    start_coords = filamentObj.start_coords
    filament_com_wcs = filamentObj.filament_com_wcs
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

    ax0.imshow(filament_item.sum(1), origin='lower', cmap='gray', interpolation='none')
    ax0.contourf(filament_item.sum(1), levels=[0., .01], colors='w')

    ax1.imshow(filament_item.sum(2), origin='lower', cmap='gray', interpolation='none')
    ax1.contourf(filament_item.sum(2), levels=[0., .01], colors='w')

    xticks = np.array(pix_ticks_lbv[0]) - start_coords[2]
    xtick_labels = wcs_ticks_lbv[0]
    xtick_labels_format = [f"{element}°" for element in xtick_labels]
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xtick_labels_format)

    xticks = np.array(pix_ticks_lbv[1]) - start_coords[1]
    xtick_labels = wcs_ticks_lbv[1]
    xtick_labels_format = [f"{element}°" for element in xtick_labels]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels_format)

    yticks = np.array(pix_ticks_lbv[2]) - start_coords[0]
    ytick_labels = wcs_ticks_lbv[2]
    ax0.set_yticks(yticks)
    ax0.set_yticklabels(ytick_labels)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ytick_labels)
    ax0.set_xlabel("Galactic Longitude", fontsize=fontsize)
    ax0.set_ylabel("V (km s$^{-1}$)", fontsize=fontsize)
    ax1.set_xlabel("Galactic Latitude", fontsize=fontsize)
    ax1.set_ylabel("V (km s$^{-1}$)", fontsize=fontsize)
    # ax0.invert_yaxis()
    # ax1.invert_yaxis()
    if save_path != None:
        plt.savefig(save_path)
    plt.show()


def Plot_Filament_Profile(filamentObj, figsize=(8, 6), fontsize=16, xlims=(-30, 30), save_path=None):
    fig, (ax0) = plt.subplots(1, 1, figsize=figsize)
    dictionary_cuts = filamentObj.dictionary_cuts

    for i in range(0, len(dictionary_cuts['distance'])):
        dists_i = dictionary_cuts['distance'][i][np.where(dictionary_cuts['profile'][i] != 0)]
        delta_dist = dists_i[-1] - dists_i[0]
        if delta_dist > filamentObj.MinProfileLen:
            ax0.plot(dictionary_cuts['distance'][i], dictionary_cuts['profile'][i], c='gray', alpha=0.3)

    ax0.plot(filamentObj.axis_coords_left, filamentObj.mean_profile_left, c='r', marker='.', alpha=1,
             label='Mean Profile')
    ax0.plot(filamentObj.axis_coords_right, filamentObj.mean_profile_right, c='r', marker='.', alpha=1)
    ax0.plot(filamentObj.axis_coords_right, filamentObj.mean_profile_left_r, c='b', marker='.', alpha=1,
             label='Mean Profile of Left Part')
    # ax0.plot(filamentObj.axis_coords, filamentObj.mean_profile,c='r',marker='.',alpha=1,label='Mean Profile')
    ax0.axvline(0, color='b', linestyle='dashed', alpha=0.5, label='Axis of Symmetry')
    # ax0.text(-25,70,'(a)',color='black',fontsize=fontsize+4)
    #     ax0.text(12,50,'SIOU={}'.format(filamentObj.profile_IOU),color='black',fontsize=fontsize)

    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.xlim(xlims[0], xlims[1])
    plt.legend(fontsize=fontsize - 4)
    plt.xlabel("Radial Distance (Pix)", fontsize=fontsize)
    plt.ylabel(r"Integrated Intensity (K)", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)

    if save_path != None:
        plt.savefig(save_path, format='pdf', dpi=1000)
    plt.show()


def Plot_Clumps_Velocity(filamentObj, figsize=(8, 6), fontsize=12, spacing=12 * u.arcmin, save_path=None):
    data_wcs = filamentObj.clumpsObj.data_wcs
    origin_data = filamentObj.clumpsObj.origin_data
    regions_data = filamentObj.clumpsObj.regions_data
    filament_regions_data = filamentObj.filament_regions_data
    filaments_data = np.zeros_like(origin_data)
    filaments_data[regions_data > 0] = origin_data[regions_data > 0]
    dictionary_cuts = filamentObj.dictionary_cuts

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_subplot(111)  # ,projection=data_wcs.celestial

    filament_regions_data = np.array(filament_regions_data, dtype='int')
    filament_regions_list = measure.regionprops(filament_regions_data)
    #     for i in range(len(filament_regions_list)):
    #         region_coords = filament_regions_list[i].coords
    #         region_coords_lb = []
    #         for i in range(len(region_coords[:,1])):
    #             region_coords_lb.append(tuple([region_coords[:,2][i],region_coords[:,1][i]]))
    #         region_coords_lb = np.array(list(set(region_coords_lb)))
    #         contour_data = np.zeros_like(filament_regions_data.sum(0))
    #         contour_data[region_coords_lb[:,1],region_coords_lb[:,0]] = 1
    #         contours = measure.find_contours(contour_data,0.5)
    #         contours_T = contours[0]
    #         for i in range(1,len(contours)):
    #             if len(contours[i])>len(contours_T):
    #                 contours_T = contours[i]
    #         ax0.plot(contours_T[:,1],contours_T[:,0],linewidth=4)

    centers = filamentObj.clumpsObj.centers
    angles = filamentObj.clumpsObj.angles
    edges = filamentObj.clumpsObj.edges
    line_scale = 3
    for index in range(len(centers)):
        center_x = centers[index][1]
        center_y = centers[index][2]
        cen_x1 = center_x + line_scale * np.sin(np.deg2rad(angles[index]))
        cen_y1 = center_y + line_scale * np.cos(np.deg2rad(angles[index]))
        cen_x2 = center_x - line_scale * np.sin(np.deg2rad(angles[index]))
        cen_y2 = center_y - line_scale * np.cos(np.deg2rad(angles[index]))
        if edges[index] == 0:
            lines = plt.plot([cen_y1, cen_y2], [cen_x1, cen_x2])
            plt.setp(lines[0], linewidth=2, color='red', marker='.', markersize=2)
    #             ax0.plot([cen_y1, cen_y2], [cen_x1, cen_x2], 'r-', markersize = 8.,linewidth = 1.,alpha=1)
    velocity_colors = filamentObj.clumpsObj.centers_wcs[:, 2]
    scatter = ax0.scatter(centers[:, 2], centers[:, 1], c=velocity_colors, marker='*', s=50, alpha=2, cmap='winter')

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("top", size="5%", pad=0)
    cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')
    cbar.set_label('km s$^{-1}$', fontsize=fontsize, color='darkcyan', loc='left')
    cbar.ax.tick_params(axis='x', colors='darkcyan', labelsize=fontsize)

    filaments_data = filaments_data * 0.166
    vmin = np.min(filaments_data.sum(0)[np.where(filaments_data.sum(0) != 0)])
    vmax = np.nanpercentile(filaments_data.sum(0)[np.where(filaments_data.sum(0) != 0)], 98.)
    img = ax0.imshow(filaments_data.sum(0),
                     origin='lower',
                     cmap='gray',
                     interpolation='none',
                     norm=colors.Normalize(vmin=vmin, vmax=vmax))
    ax0.contourf(filaments_data.sum(0),
                 levels=[0., .01],
                 colors='w')

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0)
    cbar = plt.colorbar(img, cax=cax, orientation='vertical')
    cbar.set_label('K km s$^{-1}$', fontsize=fontsize, color='black')
    cbar.ax.tick_params(axis='y', colors='black', labelsize=fontsize)

    fig.tight_layout()
    ax0.set_xticks([]), ax0.set_yticks([])
    if save_path != None:
        filament_data_shape = filaments_data.shape
        #         ax0.text(filament_data_shape[2]/20,filament_data_shape[1]-filament_data_shape[1]/15,'{}'.\
        #          format(save_path.split('/')[-1].split('_')[0]),color='black',fontsize=fontsize)
        sava_name = save_path
        plt.savefig(sava_name, format='pdf', bbox_inches='tight', dpi=1200)
    plt.show()