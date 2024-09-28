import time
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table


def Table_Interface_Pix(filamentObj):
    CenL = list(np.array(filamentObj.filament_com_all)[:, 0])
    CenB = list(np.array(filamentObj.filament_com_all)[:, 1])
    CenV = list(np.array(filamentObj.filament_com_all)[:, 2])

    Length = filamentObj.filament_length_all
    Ratio = filamentObj.filament_ratio_all
    Angle = filamentObj.filament_angle_all
    Clumps = filamentObj.clumps_number_all
    Area = filamentObj.filament_lb_area_all
    # VSpan = filamentObj.filament_v_span_all
    # VGrad = filamentObj.filament_v_grad_all

    index_id = list(np.arange(1, len(CenL) + 1, 1))
    d_outcat = np.hstack([[index_id, CenL, CenB, CenV, Length, Area, Ratio, Angle, Clumps]]).T
    columns = ['ID', 'CenL', 'CenB', 'CenV', 'Length', 'Area', 'LWRatio', 'Angle', 'Clumps']
    units = [None, 'pix', 'pix', 'pix', 'pix', 'pix', None, 'deg', None]
    dtype = ['int', 'float32', 'float32', 'float32', 'int', 'int', 'float16', 'float16', 'int']
    Filament_Table_Pix = Table(d_outcat, names=columns, dtype=dtype, units=units)
    for i in range(len(dtype)):
        if dtype[i] == 'float32':
            Filament_Table_Pix[columns[i]].info.format = '.3f'
        if dtype[i] == 'float16':
            Filament_Table_Pix[columns[i]].info.format = '.2f'
    return Filament_Table_Pix


def Table_Interface_WCS(filamentObj):
    CenL = list(np.array(filamentObj.filament_com_wcs_all)[:, 0])
    CenB = list(np.array(filamentObj.filament_com_wcs_all)[:, 1])
    CenV = list(np.array(filamentObj.filament_com_wcs_all)[:, 2])

    Length = filamentObj.filament_length_all
    Length_Arcmin = np.array(Length) * 0.5
    Ratio = filamentObj.filament_ratio_all
    Angle = filamentObj.filament_angle_all
    Clumps = filamentObj.clumps_number_all
    Area = filamentObj.filament_lb_area_all
    Area_Arcmin = np.array(Area) * 0.25

    index_id = list(np.arange(1, len(CenL) + 1, 1))
    d_outcat = np.hstack([[index_id, CenL, CenB, CenV, Length_Arcmin, Area_Arcmin, Ratio, Angle, Clumps]]).T
    columns = ['ID', 'CenL', 'CenB', 'CenV', 'Length', 'Area', 'LWRatio', 'Angle', 'Clumps']
    units = [None, 'deg', 'deg', 'km/s', 'arcmin', 'arcmin^2', None, 'deg', None]
    dtype = ['int', 'float32', 'float32', 'float32', 'float16', 'float16', 'float16', 'float16', 'int']
    Filament_Table_WCS = Table(d_outcat, names=columns, dtype=dtype, units=units)
    for i in range(len(dtype)):
        if dtype[i] == 'float32':
            Filament_Table_WCS[columns[i]].info.format = '.3f'
        if dtype[i] == 'float16':
            Filament_Table_WCS[columns[i]].info.format = '.2f'
    return Filament_Table_WCS



