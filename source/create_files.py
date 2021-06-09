# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:36:06 2020

@author: fabri
"""

# import general librarys
import cv2                # image processing tool
import csv                # handling of csv files
import numpy as np        # mathematical oparations
import json               # for writing and reading to file in json format

##############################################################################

def create_csv(directory,dir_name,name_list,exp_data):
    """
    creates a csv file based on the results of the defect detection
    """

    header=['No','sample','magnification','material','supply_factor','geometry','scan_strategy','spotsize[um]',
            'hatch[mm]','layer_thickness[mm]','laser_power[W]','markspeed[mm/s]','volume_energy_density[J/mm3]','area_accuracy','perimeter_accuracy',
            'porosity[-]','total_defect_count','porosity_LOF[-]','porosity_gas_pore[-]','porosity_crack[-]','porosity_ump[-]',
            'x_min[um]','x_max[um]','x0_10[um]','x0_50[um]','x0_90[um]','x2_10[um]','x2_50[um]','x2_90[um]']

    with open(directory+ '\\' + dir_name + '_list_of_defects.csv', 'a', newline='') as f_output:
        csv_output = csv.DictWriter(f_output, fieldnames=header)
        f_output.seek(0, 2)
        
        if f_output.tell() == 0:
            csv_output.writeheader()
    
        csv_output.writerow(exp_data)



##############################################################################

def create_json(dir_name,sample,magnification,parameter_set,porosity,
                shape_accuracy,pores,defect_list,pixel_size,directory,distribution_parameter):
    """
    creates a json file based on the results of the defect detection with the
    assigntment of the process paramenters from .json file if available

    """

    exp_data={
        'No': dir_name,
        'sample': sample,
        'magnification': magnification,
        'material': parameter_set[0],
        'supply_factor': parameter_set[1],
        'geometry': parameter_set[2],
        'scan_strategy': parameter_set[3],
        'spotsize[um]': parameter_set[4],
        'hatch[mm]': parameter_set[5],
        'layer_thickness[mm]': parameter_set[6],
        'laser_power[W]': parameter_set[7],
        'markspeed[mm/s]': parameter_set[8],
        'volume_energy_density[J/mm3]': parameter_set[9],
        'porosity[-]': porosity,
        'area_accuracy': shape_accuracy[0],
        'perimeter_accuracy': shape_accuracy[1],
        'total_defect_count': sum(len(row)/2 for row in defect_list),
        'porosity_LOF[-]': pores[0],
        'porosity_gas_pore[-]': pores[1],  
        'porosity_crack[-]': pores[2],
        'porosity_ump[-]': pores[3],
        'x_min[um]': distribution_parameter[0][0],
        'x_max[um]': distribution_parameter[0][1],
        'x0_10[um]': distribution_parameter[0][2],
        'x0_50[um]': distribution_parameter[0][3],
        'x0_90[um]': distribution_parameter[0][4],
        'x2_10[um]': distribution_parameter[1][0],
        'x2_50[um]': distribution_parameter[1][1],
        'x2_90[um]': distribution_parameter[1][2]
    } 
    
    defects={

        'area_ump[um2]':     list(np.multiply(defect_list[0],pixel_size**2)),
        'feret_ump[um]':    list(np.multiply(defect_list[1],pixel_size)),
        'area_gas[um2]':     list(np.multiply(defect_list[2],pixel_size**2)),
        'feret_gas[um]':    list(np.multiply(defect_list[3],pixel_size)),
        'area_crack[um2]':   list(np.multiply(defect_list[4],pixel_size**2)),
        'feret_crack[um]':  list(np.multiply(defect_list[5],pixel_size)),
        'area_LOF[um2]':     list(np.multiply(defect_list[6],pixel_size**2)),
        'feret_LOF[um]':    list(np.multiply(defect_list[7],pixel_size)),
        
        }

    with open(directory+ '\\JSON\\' +dir_name + '_' + sample + '.json','w') as f: 
        json.dump([exp_data,defects], f, indent = 4)
        
    return(exp_data)