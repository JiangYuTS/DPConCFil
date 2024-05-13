import time
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table

def Table_Interface_Pix(filamentObj):
    CenL = list(np.array(filamentObj.filament_com_all)[:, 0] )
    CenB = list(np.array(filamentObj.filament_com_all)[:, 1] )
    CenV = list(np.array(filamentObj.filament_com_all)[:, 2] )
    
    Length = filamentObj.filament_length_all
    Ratio = filamentObj.filament_ratio_all
    Angle = np.around(filamentObj.filament_angle_all,0)
    Clumps = filamentObj.clumps_number_all
    Area = filamentObj.filament_lb_area_all
    VSpan = filamentObj.filament_v_span_all
    VGrad = filamentObj.filament_v_grad_all
    
    index_id = list(np.arange(1, len(CenL) + 1, 1))
    d_outcat = np.hstack([[index_id, CenL, CenB, CenV,Length,Area,Ratio,Angle,Clumps]]).T
    columns=['ID', 'CenL', 'CenB', 'CenV', 'Length','Area', 'Ratio', 'Angle','Clumps']
    units = [None,'pix','pix','pix','pix','pix',None,'deg',None]
    dtype = ['int','float32','float32','float32','int','int','float32','int','int']
    Filament_Table_Pix = Table(d_outcat,names = columns,dtype=dtype,units=units)
    for i in range(len(dtype)):
        if dtype[i] == 'float32':
            Filament_Table_Pix[columns[i]].info.format = '.3f'
    return Filament_Table_Pix

def Table_Interface_WCS(filamentObj):
    CenL = list(np.array(filamentObj.filament_com_wcs_all)[:, 0] )
    CenB = list(np.array(filamentObj.filament_com_wcs_all)[:, 1] )
    CenV = list(np.array(filamentObj.filament_com_wcs_all)[:, 2] )
    
    Length = filamentObj.filament_length_all
    Length_Arcmin = np.array(Length) * 0.5
    Ratio = filamentObj.filament_ratio_all
    Angle = filamentObj.filament_angle_all
    Clumps = filamentObj.clumps_number_all
    Area = filamentObj.filament_lb_area_all
    Area_Arcmin = np.array(Area) * 0.25
    
    index_id = list(np.arange(1, len(CenL) + 1, 1))
    d_outcat = np.hstack([[index_id, CenL, CenB, CenV,Length_Arcmin,Area_Arcmin,Ratio,Angle,Clumps]]).T
    columns=['ID', 'CenL', 'CenB', 'CenV', 'Length','Area', 'Ratio', 'Angle','Clumps']
    units = [None,'deg','deg','km/s','arcmin','arcmin^2',None,'deg',None]
    dtype = ['int','float32','float32','float32','float32','float32','float32','int','int']
    Filament_Table_WCS = Table(d_outcat,names = columns,dtype=dtype,units=units)
    for i in range(len(dtype)):
        if dtype[i] == 'float32':
            Filament_Table_WCS[columns[i]].info.format = '.3f'
    return Filament_Table_WCS

def Filament_Detect(filamentObj):
    start_1 = time.time()
    start_2 = time.ctime()
    save_files = filamentObj.save_files
    mask_name = save_files[0]
    filament_table_pix_name = save_files[1]
    filament_table_wcs_name = save_files[2]
    filament_infor_name = save_files[3]
    
    filament_infor_all = filamentObj.Filament_Infor_All()
    np.savez(filament_infor_name,filament_infor_all = filament_infor_all)
    
    coms_vbl = filamentObj.filament_com_all
    if len(coms_vbl)!=0:
        filament_regions_data = filamentObj.filament_regions_data
        Filament_Table_Pix = Table_Interface_Pix(filamentObj)
        Filament_Table_WCS = Table_Interface_WCS(filamentObj)
        fits.writeto(mask_name, filament_regions_data, overwrite=True)
        Filament_Table_Pix.write(filament_table_pix_name,overwrite=True)  
        Filament_Table_WCS.write(filament_table_wcs_name,overwrite=True)  
        end_1 = time.time()
        end_2 = time.ctime()
        delta_time = np.around(end_1-start_1,2)
        time_record = np.hstack([[start_2, end_2, delta_time]])
        time_record = Table(time_record, names=['Start', 'End', 'DTime'])
        time_record.write(mask_name[:-5] + 'time_record.csv',overwrite=True)    
        filamentObj.delta_time = delta_time
        print('Number:', len(coms_vbl))
        print('Time:', delta_time)
        return filament_infor_all,Filament_Table_Pix,Filament_Table_WCS
    else:
        print('Number:', len(coms_vbl))
        return None,None,None

