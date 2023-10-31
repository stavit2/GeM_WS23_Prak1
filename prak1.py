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
#Also changed the 10th element of the original vector

y = np.arange(0.2, 200.02, 0.2)#vector 0,2 - 200,00

skalar = np.dot(x,y)#skalarprodukt

element = x*y#elementweise multiplication

xSum = np.sum(x)#sum of elements
#%%
import numpy as np

randMatrix = np.random.randn(3,4)#random 3x4 matrix
einsen = np.ones((1,4))#1x4 1 matrix
new_randMatrix = np.vstack((randMatrix,einsen))#new matrix with extra row

transponiert = np.transpose(new_randMatrix)#transposed matrix
inverse = np.linalg.inv(new_randMatrix)#inverted matrix
determinant = np.linalg.det(new_randMatrix)#determinant

#%%
import numpy as np
import matplotlib.pyplot as plt

phi = np.linspace(0, 2*np.pi, 30)#angles for one periodlength
t = np.linspace(0, 1/50, 30)#time in 50hz
y = 5*np.sin(phi)#5*sin(phi)

plt.plot(t, y)#plotting y over t

plt.xlabel('Zeit in Sekunde')
plt.ylabel('Spannung in V')
plt.title('Spannungsverlauf')

yMax = np.argmax(y)#index where y hits maximum
tMax = t[yMax]#time of maximum

plt.plot(t,y,'-gD', markevery = [yMax])#Plot with marker at maximum

np.save('t',t)#saving t
np.save('y',y)#saving y

#%%
import numpy as np
import matplotlib.pyplot as plt

t = np.load('t.npy')#load time vector
y = np.load('y.npy')#load amplitude vector
phi = np.linspace(0, 2*np.pi, 30)#angles for one periodlength
rausch = np.random.randn(30,1)

plt.subplot(2,1,1)
plt.plot(t,y)
plt.xlabel('Zeit in Sekunde')
plt.ylabel('Spannung in V')
plt.title('Spannungsverlauf')
plt.subplot(2,1,2)
plt.plot(t,rausch)
plt.xlabel('Zeit in Sekunde')
plt.ylabel('Amplitude')
plt.title('Rauschsignal')
plt.tight_layout()

plt.savefig('Plot.pdf')
#%%

###Praktische Aufgaben

import mdt
import numpy as np

messung = mdt.dataRead(amplitude = 5, samplingRate = 48000, duration = 1/25, channels = [0,1], resolution = 14, outType = 'volt')#Measurement

np.save('messung',messung)#saving measurement

#%%

import numpy as np
import matplotlib.pyplot as plt

messung = np.load('messung.npy')#loading measurement
messungTransposed = np.transpose(messung)#transposing matrix to 1 column

t = np.linspace(0,1/50, 1920)#timespan 50Hz

plt.subplot(2,1,1)
plt.plot(t[500:1460],messungTransposed[500:1460])#plotting 1 period
plt.xlabel('Zeit in Sekunde')
plt.ylabel('Spannung in V')
plt.title('Ausgeschnittene Signal')
plt.subplot(2,1,2)
plt.plot(t,messungTransposed)#plotting whole signal
plt.xlabel('Zeit in Sekunde')
plt.ylabel('Spannung in V')
plt.title('Vollstaendige Signal')
plt.tight_layout()

#%%

###Matrizen und lineare Gleichungssysteme

import numpy as np
 
m = (0.45 - 3.1) / 35 #slope
b = 3.1 #Achsenschnittpunkt

koeffizienten = np.array([m,b])#Coefficient

theta = 20#input angle

spannung = np.dot(koeffizienten, np.array([theta, 1]))#Voltage calculation

print("Spannung in V:", {spannung})

#%%

###Selbsdefinierte Funktionen

import numpy as np

def angle2voltage(theta):
    m = (0.45 - 3.1) / 35 #slope
    b = 3.1 #Achsenschnittpunkt

    koeffizienten = np.array([m,b])#Coefficient
    
    spannung = np.dot(koeffizienten, np.array([theta, 1]))#Voltage calculation

    return spannung

print("Spannung in V für Winkel 23°:",{angle2voltage(23)})

#%%

###Grafiken

import numpy as np
import matplotlib.pyplot as plt

def angle(t):
    return 1/2*(np.tanh(t-5) + 1)*35#Calculating angle

t = np.linspace(0,10,11)#time vector t:[0;10]

angles = np.array(angle(t))#angle vector for t
voltage = np.vectorize(angle2voltage)(angles)#voltage vector

plt.plot(t, angles, label='Winkel in °')
plt.plot(t, voltage, label='Spannung in V')

plt.xlabel('Zeit in Sekunde')
plt.legend()
plt.title('Gaspedal')