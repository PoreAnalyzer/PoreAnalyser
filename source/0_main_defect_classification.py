# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:50:05 2020

@author: fabri

this code detects and analyses defects from polished metal sample surfaces 
produced in a SLM-printer 
"""
# import general librarys
## ensure that they are installed
import csv                        # handling of csv files
import cv2                        # image processing tool
import imutils                    # image manipulation
import json                       # for writing and reading to file in json format
import matplotlib.pyplot as plt   # plotting stuff
import numpy as np                # mathematical oparations
import os                         # file and directory management
import timeit                     # counting time
from tqdm import tqdm             # process bar in console

# import specific python scripts (they have to be in the same directory as __main__)
import walk_dirs                  # finds directorys and files
import process_image              # image processing
import detect_obj                 # Object detection
import generate_data              # generate specific informatoion from image objects
import create_files               # creating files from data

start = timeit.default_timer()    # start timer for runtime detection



##############################################################################
#____Initialization BEGIN____#
"1. Definition of your sample data"

# file format of the images to be evaluated
### do not use .jpg -> could generate an error
suffix = '.tif'

# choose:
### 'array'     for an image with more than one sample
### 'samples'   for a bunch of images with only one sample
image_type = 'samples'

# choose if image_type == 'array'.
### it has to be a list (specified by square brackets). 
### Each entry represents the number of samples on the images in the 
### 0_Array folder. -> first entry = first folder,...
### Notice: the number of samples in the images is asked not the total number 
### of samples in the folder
### alphabetical order in explorer
number_of_samples = [4]
                     
# magnification
### it has to be a list (specified by square brackets). 
### Each entry represents one folder in data directory
### -> first entry = first folder,...
### alphabetical order in explorer
magnification =  [10]
                     

# sample size in mm as tuple for each experiment directory 
### for circular samples give a tuple with (diameter,diameter)
### for rectangle samples give a tuple with (height,width)
sample_size = [(4,4)]
# if image_type == 'sample' the images were cropped to eliminate the edge,when 
# they are in bad shape
### define how much of the sample is cropped. If value is to low some parts of edge
### can remain in image. 
crop_factor = 0.05        # default: 0.1
    
"2. Definition of pores"
# the script searches for four differnt pore types
### unmelted particles      -- singular particles inside a pore
### gas pores               -- almost circular
### cracks                  -- long defects     
### Lack of fusion (LOF)    -- all the rest

# defines the tolerable deviation of a gas pore, which is almost circular
circularity = 0.95      # default: 0.95

# the ratio from height to width of each pore. if larger its defines as crack
crack_ratio = 5


"3. Definition of the output"   

# prints all the defects of one sample in a corresponding folder
print_defects = False

# saves all of the processed sample images (binarized, rotated, cropped)
# in 2_Frames folder in low quality to check if correctly cropped
print_proc = False

# assign process parameter to each sample. You need to create a .json file with
# process paramters first. !-> Use parameter_set.py! 
# Move the file into the associated  1_Sample folder
assign_data = True

#____Initialization END____#

##############################################################################

#____Load Directory BEGIN____#

# loads all directorys inside data folder
dir_list,subdir_list,file_list = walk_dirs.find_files()

#____Load Directory END____#   

##############################################################################

#____Process Arrays BEGIN____#

# if image_type == 'array', the image has to be cropped in single samples
### those single sample images are saved to the 1_Samples folder
if image_type == 'array':
    print('\nProcessing arrays. Please wait...')
    # crop each sample from the array and save it to a new file
    process_image.array_to_samples(subdir_list,file_list,suffix,number_of_samples)

    # reload the filenames in the 1_Samples folder
    dir_list,subdir_list,file_list = walk_dirs.find_files()

    print('\nProcessing arrays succeeded.\nFor analysing your samples' ,
          'change image_type to "samples"')
    
   
#____Process Arrays END____#

##############################################################################

#____Process Samples BEGIN____#

# detect defects in every image of each folder in data\1_Array
if image_type == 'samples':
    
    # pixel size in im µm
    ### for Axio magnifications: 
    pixel_size = []
    for m in range(len(magnification)):
        if magnification[m] == 2.5:
            pixel_size.append(2.00000)
        elif magnification[m] == 5:
            pixel_size.append(1.11111)
        elif magnification[m] == 10:
            pixel_size.append(0.55556)
        elif magnification[m] == 20:
            pixel_size.append(0.27778)
        elif magnification[m] == 50:
            pixel_size.append(0.10989)

    # iterate for every experiment folder in 1_Experiments
    for d in range(len(dir_list)):

        # check if there is at least one file to analyse, if not skip the directory
        if walk_dirs.file_exist(file_list[d][1],suffix) == False:
            continue
            
        # crop the edges from each sample and save it to an array
        name_list,im_crop,area,shape_accuracy=process_image.crop_edge_from_sample(d,dir_list,file_list,
                                                                   suffix,crop_factor,
                                                                   sample_size[d],pixel_size[d],
                                                                   print_proc)
        
        # invert color in image
        im_inv = process_image.invert_color(im_crop)
        
        # loop over all images
        for img in tqdm(range(len(name_list)),desc='analysing ' + walk_dirs.find_name(dir_list[d],'')):
            # print(name_list[img])     # for debugging
            
            # get contours and its hierachy of each object inside the image
            contours,hierarchy = cv2.findContours(im_inv[img], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            # define number of unmelted particles (ump) and position of its contour
            outer,inner=detect_obj.count_ump(contours,hierarchy,im_inv[img],
                                                 subdir_list[d][3],print_defects,
                                                 name_list[img])
            
            # get countours which are defined as gas pore
            gas_pores = detect_obj.count_gaspores(contours,circularity,im_inv[img],
                                                  outer,inner,subdir_list[d][3],
                                                  print_defects,name_list[img])
              
            # identify contours that consists of less than five data points 
            ### due to fitEllipse() need at least 5 contour points
            small_cnt,a_scnt = detect_obj.define_small_cnt(contours,outer,inner,
                                                         gas_pores)
          
            # classify defects by geometric relations
            LOF,cracks,defect_list,a_in = detect_obj.classify_defects(contours,small_cnt,outer,inner,
                                                     gas_pores,crack_ratio,
                                                     print_defects,im_inv[img],
                                                     subdir_list[d][3],name_list[img])
                                                        
        

            # calculate porosity
            porosity,pores = generate_data.compute_porosity(defect_list,a_in,a_scnt,area[img])

             
            # generates plots of the number size distribution of the defects
            distribution_parameter=generate_data.size_distribution(contours,inner,small_cnt,pixel_size[d],
                                            name_list[img],subdir_list[d][4],walk_dirs.find_name(dir_list[d],''))
            parameter_set = [np.nan]*11
            if assign_data == True:
                # check if .json file exists
                if walk_dirs.file_exist(file_list[d][1],'.json') == True:
                    # assign process data to image 
                    parameter_set = generate_data.assign_parameters(file_list[d][1],name_list[img])
                else:
                    print('\nno .json document found which contains process parameters')
            # write process data into json file
            exp_data=create_files.create_json(walk_dirs.find_name(dir_list[d],''),
                                              name_list[img],magnification[d],
                                              parameter_set,porosity,shape_accuracy[img],
                                              pores,defect_list,pixel_size[d],subdir_list[d][4],distribution_parameter)
            
            # save image data to a csv into the 4_Results folder
            create_files.create_csv(subdir_list[d][4],walk_dirs.find_name(dir_list[d],''),
                    name_list[img],exp_data)
        
#____Process Samples END____#

##############################################################################
    
stop = timeit.default_timer()       # stop timer for runtime detection
print('\nruntime: ' + str(np.around((stop - start),2))+ ' s')




