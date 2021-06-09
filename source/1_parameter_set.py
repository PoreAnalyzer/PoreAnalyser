# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:10:29 2020

@author: fabri
"""

"""
this script saves the given print parameter into a npy file
"""

import numpy as np
import sys
import json

# name of experiment as string
name = 'V277'

# material as string
material = '316L_0-45_0,5Vol%TiO2_21nm_MK3000_1440min'

# supply factor as int
supply_factor= 3

# sample geometry as tupel in mm [width, height, depth] 
### for cylindric samples: width=height
### view: buildup direction
### means: depth is layerthickness x layer count
geometry = (4,4,5)

# scan strategy as sting
scan_strategy = '90°'

# spot size in µm
spot_size = 90

# Hatch in mm
h_s = 70e-3 

# Layer thickness in mm
l_z = 50e-3

# laser power in W as matrix with array_shape
P_l = np.array([300])

# Markspeed in mm/s
v_s = np.array([200])

# sample assignment to matrix position
samples = ['03']

# calculate volume energy density in J/mm³
E_v = P_l/(v_s*h_s*l_z)


parameter_set={
        'No': name,
        'samples': samples,
        'material': material,
        'supply_factor': supply_factor,
        'geometry': geometry,
        'scan_strategy': scan_strategy,
        'spotsize': spot_size,
        'hatch': h_s,
        'layer_thickness':l_z,
        'laser_power': list(P_l.astype(float)),
        'markspeed': list(v_s.astype(float)),
        'volume_energy_density': list(E_v)
        } 

with open(name + '.json','w') as f: 
        json.dump(parameter_set, f, indent = 4)


