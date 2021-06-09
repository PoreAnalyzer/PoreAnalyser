# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:10:23 2020

@author: fabri

"""

# import general librarys
import os                         # file and directory management

##############################################################################

def find_files():
    " finds and lists the files in the corrensponting directorys"
    
    # navigate to parent directory 
    start_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
    
    # lists all the directorys in the '1_Experiments' folder 
    path = start_dir+ '\\data'
    dir_list = [f.path for f in os.scandir(path) if f.is_dir()]
    dir_list.sort()
    # lists all the subdirectorys, found in the directorys
    subdir_list=[]
    for directory in dir_list:
        subdir_list.append([f.path for f in os.scandir(directory) if f.is_dir()])
    for i in range(len(subdir_list)):
        subdir_list[i].sort()
    # lists all the files, found in the subdirectorys
    file_list =[[[] for sd in d ] for d in subdir_list]
    for j in range(len(subdir_list)):
        for i in range(len(subdir_list[j])):
            file_list[j][i]=([f.path for f in os.scandir(subdir_list[j][i])])
    
    return(dir_list,subdir_list,file_list)

##############################################################################

def find_name(string,suffix): 
    """
    find the names of the given string, if the string is a absolute path. It takes the
    last words which are between the last "/" and the ending of the file/folder
    """    
    start = string.rindex( '\\' ) + len( '\\' )
    end = string.rindex( suffix, start )
    
    name = string[start:end]

    return(name)  

##############################################################################

def file_exist(file_list,suffix):
    """
    checks if a file with the given suffix exists in the given file list
    """  

    if file_list == []:
        file=False
        
    for i in file_list:
        
        if suffix in i:
            file=True
            break
        else:
            file=False

            
    return(file)

##############################################################################
        
        
