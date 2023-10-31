# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:38:35 2023

@author: Stav
"""

#%% 1.2 Wichtige Befehle
import numpy as np

x = np.linspace(1,1001,1000)#New vector 1-1000
x = np.arange(1,1001,1)#new vector 1-1000

#x_new = x[10:50].copy()#Copy first 50 elements of vector x
#x = x[5]#Delete first 50 elements of vector x
x_new = x[10:50]

x_new[0] = 0#First element = 0
#Also changed the 10th element in the original vector

y = np.arange(0.2, 200.02, 0.2)#vector 0,2 - 200,00

skalar = np.dot(x,y)#skalarprodukt

element = x*y#elementweise multiplication

elementSum = np.sum(element)#sum of elements

randMatrix = np.random.rand(3,4)
einsen = np.ones((1,4))
np.append(randMatrix, einsen)