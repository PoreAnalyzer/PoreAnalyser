# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:09:30 2020

@author: fabri
"""
# import general librarys
import cv2                         # image processing tool
import numpy as np                 # mathematical oparations
import matplotlib.pyplot as plt    # plotting stuff
from matplotlib.ticker import AutoMinorLocator   # automated axis scaling
import json


import walk_dirs

##############################################################################

def compute_porosity(defect_list,a_in,a_scnt,area):
    """
    calculates relativ porosity of the image by comparing the areas of each 
    defect type
    """  
    # summarize defects porosities
    pores=[sum(defect_list[6])/area, # LOF
           sum(defect_list[2])/area, # Gas
           sum(defect_list[4])/area, # Crack
           # the particles which are inside the Unmelted particle Pores are
           # white areas, so its the difference between both 
           (sum(defect_list[0])-sum(a_in))/area] # Ump
    
    # by summing each pore types porosity the whole porosity is calculated
    # those contours which are to small for evaluation are also considered
    porosity = (sum(pores)+(sum(a_scnt)/area))

    return(porosity,pores)


##############################################################################

def size_distribution(contours,inner,small_cnt,pixel_size,name,directory,dir_name):
    """
    computes and plots the Q0 and Q2 size distribution of the surface equivalent diameter into 
    a pdf file. Also gives x_min, x_max and x_50
    """
    
    # empty variable
    d_equi=[]

    # search for those contours, which are counted as defect
    for i in range(len(contours)):
        # if actual contour was not found in the inner unmelted particle list
        if i not in [x for row in inner for x in row]:
            # if actual contour wasnt in the the too small list
            if i not in small_cnt:
                # compute equivalent diameter for each relevant contour
                d_equi.append(np.sqrt((4*cv2.contourArea(contours[i]))/np.pi)*pixel_size)

    # sort list by value
    d_equi = np.sort(d_equi)
    
    # get maximum and minimum value
    d_max,d_min = max(d_equi),min(d_equi)
    
    classes,class_size = generate_classes(d_equi)
    
    number_frequency,area_frequency = sum_frequency(classes,d_equi)
    Q_0,q_0 = compute_cdf(number_frequency,classes,class_size)
    Q_2,q_2 = compute_cdf(area_frequency,classes,class_size)

    d0_10,d0_50,d0_90 = interpolate_distribution(classes,Q_0)
    d2_10,d2_50,d2_90 = interpolate_distribution(classes,Q_2)

    # summarize distribution parameters
    distribution_parameter_0 = np.around([d_min,d_max,d0_10,d0_50,d0_90],2)
    distribution_parameter_2 = np.around([d2_10,d2_50,d2_90],2)

    plot_distributions(classes, Q_0, q_0, name, directory,0,dir_name)
    plot_distributions(classes, Q_2, q_2, name, directory,2,dir_name)

    return([distribution_parameter_0,distribution_parameter_2])

##############################################################################

def assign_parameters(file_list,image):
    """
    assign process parameters from .npy file in the 1_Sample folder to the current 
    image
    """
   
    for file in file_list:
        if '.json' in file:
            process_parameters=json.load(open(file,))
                
            material = process_parameters['material']

            # supply factor as int
            supply_factor= process_parameters['supply_factor']
            
            # sample geometry as tupel  
            geometry = process_parameters['geometry']
            
            # scan strategy as sting
            scan_strategy = process_parameters['scan_strategy']

            # spot size in µm
            spot_size = process_parameters['spotsize']
            
            # Hatch in mm
            h_s = process_parameters['hatch']
            
            # Layer thickness in mm
            l_z = process_parameters['layer_thickness']
 
                            
            for i,sample in enumerate(process_parameters['samples']):
                if image == sample:

                    # laser power in W 
                    P_l = process_parameters['laser_power'][i]
                    
                    # Markspeed in mm/s
                    v_s = process_parameters['markspeed'][i]
                    
                    # calculate volume energy density in J/mm³
                    E_v = process_parameters['volume_energy_density'][i]
                    
                    # summarize the values into one variable
                    parameter_set = [material,supply_factor,geometry,scan_strategy,spot_size,h_s,l_z,P_l,v_s,E_v]
                    
                    break
                
                else:
                    parameter_set=[np.nan]*11
                    continue
            break   
         
        else:
            continue
    
    
    return(parameter_set)

##############################################################################

def compute_cdf(frequency,classes,class_size):
    """
    computes the cumulative density function of the given defects
    """

        
    Q_i = np.cumsum(frequency)/sum(frequency)
    
    q_i =[]
    for q in range(len(classes)):
        try:
            q_i.append((frequency[q]/sum(frequency))/class_size[q])
        except IndexError:
            Q_i = np.sort(np.append(Q_i,0))
            q_i.append(0)
        
    return(Q_i,q_i)

##############################################################################

def generate_classes(d_equi):
    """ 
    generate classes between d_min and d_max with unequal classsizes
    """
    # define empty vars
    classes =[]
    class_size=[]
    # initial value for classes
    i=min(d_equi)
    
    # generate classes with different class size
    while i < max(d_equi):
        classes.append(i)  
        if 0 <= i < 5:
            i += 0.5
        elif 5 <= i < 10:
            i += 1    
        elif 10 <= i < 20:
            i += 2       
        elif 20 <= i < 50:
            i += 5
        elif 50 <= i < 100:
            i += 10
        elif 100 <= i < 200:
            i += 25
        elif 200 <= i < 400:
            i += 50
        elif 400 <= i < max(d_equi):
            i += 100
            
    classes.append(max(d_equi)) 
    
        
    for i in range(len(classes)):
        try:
            class_size.append(classes[i+1] - classes[i])
            
        except IndexError:
            continue

    
    
    return(classes,class_size)

##############################################################################

def sum_frequency(classes,d_equi):
    """ 
    sum the defect count per class and the equivalent area inside the class
    
    """
    
    number_frequency,area_frequency =[],[]
    
    for i in range(len(classes)):
        try:
            sum_number = sum(classes[i] <= n <= classes[i+1] for n in d_equi)
            number_frequency.append(sum_number)
            sum_area = 0
            for n in d_equi:
                if classes[i] <= n <= classes[i+1]:
                    sum_area += np.pi*(n/2)**2
            area_frequency.append(sum_area)
            
        except IndexError:
            continue   

    return(number_frequency,area_frequency)

##############################################################################

def interpolate_distribution(classes,Q_i):
    """
    interpolate between the classes to get x_10,x_50 and x_90 from
    cumulative sum function
    """

    d_10 = np.interp(0.1,Q_i,classes)
    d_50 = np.interp(0.5,Q_i,classes)
    d_90 = np.interp(0.9,Q_i,classes)
    
    
    return(d_10,d_50,d_90)

##############################################################################

def plot_distributions(classes,Q_i,q_i,name,directory, distribution_type,dir_name):
    """
    plots the given distribution to a pdf file in the Results folder
    """

    fig, ax1 = plt.subplots()
    color1 = "tab:blue"
    ax1.plot(classes,q_i, ds='steps-post')
    ax1.set_title('Q'+str(distribution_type)+ ' distribution of surface equivalent diameter')
    ax1.set_ylabel("normalized density function", color = color1)
    ax1.tick_params(axis='y', labelcolor = color1)
    ax1.set_xlabel("equivalent pore diameter in µm")
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_xlim([0,max(classes)])
    
    ax1.set_ylim(0)
    # plotting the cumulative density function onto a second y-axis
    color2 = "tab:red"
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel("cumulative density function", color = color2)  
    ax2.plot(classes, Q_i, color = color2)
    ax2.tick_params(axis='y', labelcolor = color2)
    ax2.set_ylim(0,1)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
   
    
    plt.tight_layout()
    
    # save the figures of each sample into the 4_Results folder
    plt.savefig(directory + '\\size_distribution\\Q'+str(distribution_type)+
                '\\Q' +str(distribution_type)+'_'+dir_name + '_' + name + ".pdf")
    
    # close each figure, due to memory workload
    plt.close(fig)
    
    return()