# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:39:21 2020

@author: fabri
"""

# import general librarys
import cv2                         # image processing tool
import matplotlib.pyplot as plt    # plotting stuff
import numpy as np                 # math
import imutils                     # image manipulation
from tqdm import tqdm              # process bar in console


import walk_dirs


##############################################################################

def array_to_samples(subdir_list,file_list,suffix,number_of_samples):
    """
    crops an array image to single sample images and saves them to to 1_Samles folder
    """
                
    # iterate for every experiment folder in 1_Experiments
    for d in tqdm(range(len(file_list))):
        
        # iterate for every file in 0_Array
        for f in range(len(file_list[d][0])):
            # get the full path of the file
            file = file_list[d][0][f]
            
            if suffix in file:
                # get the name of the current file as string
                name = walk_dirs.find_name(file,suffix)
    
                # get the path for the new images
                ### it is the second folder in the directory
                path = subdir_list[d][1]
                
                # read the image as grayscale
                image = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
                
                # convert the grayscale to colorspace due to draw text and boxes
                im_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
                # Otsu's thresholding after Gaussian filtering
                blur = cv2.GaussianBlur(image,(5,5),0)
                thresh,im_bin = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                
                # get the contours of each object in the image
                contours,_= cv2.findContours(im_bin,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                
                # create a rectangular bounding box around each contour and get its area
                area,box = [],[]
                for c in contours:
                    x,y,w,h = cv2.boundingRect(c)
                    area.append(h*w)
                    box.append([x,y,w,h])
                    
                # sort the area list that the greatest is on top
                area.sort(reverse=True)    
    
                i=0         # initial value of sample counter
                
                # check the size of each bounding box
                ### assumption: the samples have the biggest bounding boxes
                ### each bounding box in the range of the number of samples is cropped
                ### and saved to a new file. 
                for b in box:
                    x,y,w,h = b
                    if h*w > area[number_of_samples[d]]:                
                        cv2.rectangle(im_color,(x,y),(x+w,y+h),(0,255,0),10)
                        cv2.putText(im_color, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,255,0), 30)
                        im_crop = im_bin[round(y):round(y+h),round(x):round(x+w)]
                        cv2.imwrite(path+'/'+ name + '_' + str(i)+ suffix,im_crop)
                        i=i+1
                # for each array a .jpg is created, where the position of the samples
                # and its index is shown       
                cv2.imwrite(path+'\\'+ name +'.jpg',im_color)
    return()

##############################################################################

def crop_edge_from_sample(d,dir_list,file_list,suffix,crop_factor,sample_size,pixel_size,print_proc):
    """
    crop the edge of each sample image if its in bad shape. If the shape is quite 
    good the pores are copied to an empty array to eliminate the countour of the 
    sample itself.
    """
    name=[];image=[];area=[];shape_accuracy=[]
   
    # iterate for every file in 0_Array
    for f in tqdm(range(len(file_list[d][1])),desc='cropping ' + walk_dirs.find_name(dir_list[d],'')):
        # get the full path of the file
        file = file_list[d][1][f]
       
        # check if the file is a file with chosen suffix
        if suffix in file:
            # get the name of the current file as string
            name.append(walk_dirs.find_name(file,suffix))
            
            path = dir_list[d]+'\\2_Frames\\'+ name[f] 
            # rotate the image for cutting correctly
            im_rot = rotate_img(file,print_proc,path)
            p,q = im_rot.shape
            
            # visualize the rotated image in 2_Frames folder
            if print_proc == True:
                plt.figure()
                plt.imshow(im_rot)
                plt.axis("off")
                plt.savefig(path +'_rotated.png')
                
            
            # finds contours in imagefig
            contours,hierarchy= cv2.findContours(im_rot,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
            
            # define shape of sample
            shape,box,area_s,max_cnt = define_shape(im_rot,contours)
            
            # get the perimeter and area of the largest contour
            perimeter_cnt = cv2.arcLength(contours[max_cnt],True)*(pixel_size*1e-3)
            area_cnt = cv2.contourArea(contours[max_cnt])*(pixel_size*1e-3)**2
            
            # crop the image depending on the sample shape
            if shape == 'rectangle':
                # calculate the perimeter and area of the theoretical sample shape
                perimeter_sample = 2*sample_size[0]+2*sample_size[1]
                area_sample = np.prod(sample_size)
                
                # compares the theoretical with the actual area and perimeter of the sample
                p_ratio = perimeter_cnt/perimeter_sample
                a_ratio = area_cnt/area_sample
                
                # definition of cropping conditions
                ### if this conditions are true, the sample is in very bad shape
                ### so it is cropped very rough in terms of the bounding box
                if (p_ratio > 3.0 and a_ratio < 1) or a_ratio < 0.75:
                    # box information
                    x,y,w,h = box
                    
                    # crop the image from each side by crop factor * (1/2)
                    im_crop = im_rot[round(y+crop_factor*h/2):round(y+(1-crop_factor/2)*h),
                                     round(x+crop_factor*w/2):round(x+(1-crop_factor/2)*w)]
                              
                    # list image 
                    image.append(im_crop)
                    
                    # compute and list area
                    area.append(im_crop.shape[0]*im_crop.shape[1])
                    
                 # if the sample is in good shape all objects inside the sample
                # contour were extracted
                else:
                    im_crop = extract_cnt(hierarchy,contours,max_cnt,im_rot)
                    area.append(cv2.contourArea(contours[max_cnt]))
                    image.append(im_crop)
                
            elif shape == 'circle':
                # calculate the perimeter and area of the theoretical sample shape
                perimeter_sample = np.pi*sample_size[0]
                area_sample = np.pi*(sample_size[0]/2)**2
                
                # compares the theoretical with the actual area and perimeter of the sample
                p_ratio = perimeter_cnt/perimeter_sample
                a_ratio = area_cnt/area_sample
                
                                
                # definition of cropping conditions
                ### if this conditions are true, the sample is in very bad shape
                ### so it is cropped very rough in terms of the fitted circle
                if (p_ratio > 3.0 and a_ratio < 1) or a_ratio < 0.75:
                    # box information
                    x,y,h,w = box
    
                    # get the size of the circle
                    radius = int(round(np.sqrt(area_s/np.pi)*(1-crop_factor)))
                    
                    # creating a empty mask for cropping a circle from image
                    mask = np.full((p,q),0,dtype= np.uint8)
                    
                    # filling the circle inside the mask
                    cv2.circle(mask,(round(x+w/2),round(y+h/2)),radius,(255,255,255),-1,8,0)
                    
                    # get only inside pixles
                    fg = cv2.bitwise_or(im_rot,im_rot,mask=mask)
                    
                    mask= cv2.bitwise_not(mask)
                    background = np.full((p,q),255,dtype= np.uint8)
                    bk = cv2.bitwise_or(background,background,mask=mask)
                    im_crop = cv2.bitwise_or(fg,bk)
                    
                    # compute and list area
                    area.append(np.pi*radius**2)
                    image.append(im_crop)
                    
                # if the sample is in good shape all objects inside the sample
                # contour were extracted
                else:
                    im_crop = extract_cnt(hierarchy,contours,max_cnt,im_rot)
                    area.append(cv2.contourArea(contours[max_cnt]))
                    image.append(im_crop)
                
            # visulize the cropped image in workspace
            if print_proc == True:
                plt.figure()
                plt.imshow(im_crop)
                plt.axis("off")
                plt.savefig(path +'_cropped.png')
            
            shape_accuracy.append([a_ratio,p_ratio])
    
    return(name,image,area,shape_accuracy)

##############################################################################

def rotate_img(file,print_proc,path):
    """
    rotates the image. Returns a image in which the object edges are parallel to image edges
    """
    image = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(image,(5,5),0)
    thresh,im_bin = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    if print_proc == True:
        plt.figure()
        plt.imshow(im_bin)
        plt.axis("off")
        plt.savefig(path +'_binarized.png')

    contours,_= cv2.findContours(im_bin,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    area,box = [],[]
    for c in contours:
        rect = cv2.minAreaRect(c)
        _,[w,h],_ = rect
        area.append(h*w)
        box.append(rect)
  
    # finds angle of the biggest bounding box
    _,_,angle = box[area.index(max(area))]

    # rotate the image
    im_rot = imutils.rotate_bound(im_bin, angle=-angle)
    
    return(im_rot)

##############################################################################

def define_shape(im_rot,contours):
    """    
    finds the shape of a sample by comparing a fitted circle and rectangle on the shape
    """  
    
    # the contour with the biggest bounding box is the sample
    area,box = [],[]
    for cnt in contours:
        x,y,h,w = cv2.boundingRect(cnt)
        area.append(h*w)
        box.append([x,y,h,w])

    # get the contour and it's box of the biggest object in image
    max_cnt = area.index(max(area))

    box = box[area.index(max(area))]

    # calculating a circle and rectangle for comparison
    _,radius = cv2.minEnclosingCircle(contours[max_cnt])
    _,[w,h],_ = cv2.minAreaRect(contours[max_cnt])
    
   
    # compute area of circle and rectangle
    area_c = np.pi*radius**2
    area_r = w*h
    
    if area_c < area_r:
        shape ='circle'
        area=area_c
    else:
        shape = 'rectangle'
        area=area_r
        
        
    return(shape,box,area,max_cnt)

##############################################################################

def invert_color(im_crop):
    """
    invert the color, due to defect detection
    -> findContours() finds white objects on black background
    """ 
    
    im_inv=[]
    for img in im_crop:
        im_inv.append(255 - img)
        
    return(im_inv)

##############################################################################

def extract_cnt(hierarchy,contours,max_cnt,im_rot):
    """
    extracts all contours which are inside a sample and copies it to a new array
    """
    outer=[];inner=[]

    # get those contours, which are outside and inside the sample contour
    for i in range(len(hierarchy[0])):
        if  hierarchy[0,i,3] == -1:
            outer.append(i) 
        elif hierarchy[0,i,3] == max_cnt:
            inner.append(i)
            
    # generate an emty mask for filling with contours
    im_crop = np.ones(im_rot.shape,dtype='uint8')*255

    # draw the contour objects in the mask
    for i in range(len(contours)):
        if not i in outer:
            cv2.fillPoly(im_crop, pts =[contours[i]], color=(0,0,0))
        if not i in inner:
            cv2.fillPoly(im_crop, pts =[contours[i]], color=(255,255,255))

    
    return(im_crop)

##############################################################################



