# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:11:09 2020

@author: fabri
"""

# import general librarys
import cv2              # image processing tool
import os               # file and directory management
import numpy as np

# import specific python scripts (they have to be in the same directory as __main__)
import process_image              # image processing


##############################################################################

def define_small_cnt(contours,outer,inner,gas_pores):
    """
    identify contours that consists of less than five data points
    due to fitEllipse() need at least 5 contour points
    """

    small_cnt=[]
    a_scnt=[]
    for i  in range(len(contours)):
        if i not in np.append([x for x in outer],[x for row in inner for x in row]):
            if i not in gas_pores:
                if  contours[i].shape < (5,1,2) or cv2.contourArea(contours[i]) == 0.0:
                    small_cnt.append(i)
                    a_scnt.append(cv2.contourArea(contours[i]))
                    
            
    return(small_cnt,a_scnt)

##############################################################################            

def count_ump(contours,hierarchy,im_inv,directory,print_defects,name):
    """
    identify objects within an other contour. 
    -> has to be a unmelted particle (ump)
    """

    inner=[]; outer=[];
    for i in range(len(hierarchy[0])):
        if  hierarchy[0,i,2] != -1:
            inside = np.where(hierarchy[0,:,3] == i)
            inner.append(inside[0])                      # inner contour
            outer.append(i)                              # outer contour

    # print ump into image
    if print_defects == True:
        for i in outer:
            x,y,w,h = cv2.boundingRect(contours[i])
            defect=im_inv[round(y):round(y+h) , round(x):round(x+w)]
            cv2.imwrite(directory+'\\3_ump\\'+ name +'_'+ str(i)+'.tif', defect)

    # number and index of parent contours
    outer=np.array(outer,dtype=object)
    
    # number and index of child contours
    inner=np.array(inner,dtype=object)
    
    
    return(outer,inner)

##############################################################################

def count_gaspores(contours,circularity,im_inv,outer,inner,directory,print_defects,name):
    """
    count gas pores by definition. It compares the contours perimeter to the 
    theoretical perimeter of a perfect circle with same area. Add's a numerator if ratio is 
    in range of the circularity condition
    """
    gas_pores=[]
    for i  in range(len(contours)):
        if i not in np.append([x for x in outer],[x for row in inner for x in row]):
            
            area = cv2.contourArea(contours[i])
            
            if area == 0.0:
                continue
            
            perimeter_circle = 2*np.pi*np.sqrt(area/np.pi)
            perimeter_object = cv2.arcLength(contours[i],True)
            
            
            circularity_obj = perimeter_circle/perimeter_object
            
            if circularity_obj > circularity:
                # add the position of gas pore contour
                gas_pores.append(i)
                
                # print gaspore into image
                if print_defects == True:
                    x,y,w,h = cv2.boundingRect(contours[i])
                    defect=im_inv[round(y):round(y+h) , round(x):round(x+w)]
                    cv2.imwrite(directory+'\\1_gas_pore\\'+ name +'_'+ str(i)+'.tif', 
                                defect)
            
    
    return(gas_pores)

##############################################################################

def classify_defects(contours,small_cnt,outer,inner,gas_pores,crack_ratio,print_defects,
                     im_inv,directory,name):
    """
    classify the pore types by fitting the contours to an ellipse and compare
    geometric relations.
    """
    # dummy variables
    ellipse,LOF,crack= [],[],[]
    a_ump,f_ump=[],[]
    a_gas,f_gas=[],[]
    a_crack,f_crack=[],[]
    a_LOF,f_LOF=[],[]
    
    a_in=[]
    
    # for all listed contours in the image
    for i in range(len(contours)):
        # if actual contour was not found in the unmelted particle list
        if i not in np.append([x for x in outer],[x for row in inner for x in row]):
            # if actual contour was not found in the gas pores list
            if i not in gas_pores:
            # if actual contour has more than 5 contour points
                if i not in small_cnt:
    
                    e = cv2.fitEllipse(contours[i])
                    ellipse.append(e)    
                    
                    # get height and width of the ellipse
                    width = ellipse[i][1][0]
                    height = ellipse[i][1][1]       # is always the lager length
                    if width == 0.0:
                        width = 0.0001

                    # if defect is very long
                    elif height/width > crack_ratio:
                        crack.append(i)
                        a_crack.append(cv2.contourArea(contours[i]))
                        _,size,_ = cv2.minAreaRect(contours[i])
                        f_crack.append(max(size))
                        # save image of defect 
                        if print_defects == True:
                            x,y,w,h = cv2.boundingRect(contours[i])
                            defect=im_inv[round(y):round(y+h) , round(x):round(x+w)]
                            cv2.imwrite(directory+'\\2_crack\\'+ name +'_'+ str(i)+'.tif', defect)
                            

                    # in all other cases
                    else:
                        LOF.append(i) 
                        a_LOF.append(cv2.contourArea(contours[i]))
                        _,size,_ = cv2.minAreaRect(contours[i])
                        f_LOF.append(max(size))
                        # save image of defect 
                        if print_defects == True:
                            x,y,w,h = cv2.boundingRect(contours[i])
                            defect=im_inv[round(y):round(y+h) , round(x):round(x+w)]
                            cv2.imwrite(directory+'\\0_LOF\\'+ name +'_'+ str(i)+'.tif', defect)
                            
                
                # when contour has less than 5 points add an empty row to ellipse array
                else:
                    ellipse.append([])
            
            # if contour is a gaspore
            else:
                ellipse.append([])
                a_gas.append(cv2.contourArea(contours[i]))
                _,size,_ = cv2.minAreaRect(contours[i])
                f_gas.append(max(size))
                
        # when contour is an ump add an empty row to ellipse array        
        else:
             ellipse.append([])
             if i in outer:
                 a_ump.append(cv2.contourArea(contours[i]))
                 _,size,_ = cv2.minAreaRect(contours[i])
                 f_ump.append(max(size))
             if i in [x for row in inner for x in row]:
                 a_in.append(cv2.contourArea(contours[i]))
     
    defect_list=[a_ump,f_ump,a_gas,f_gas,a_crack,f_crack,a_LOF,f_LOF]           

    return(LOF,crack,defect_list,a_in)

##############################################################################