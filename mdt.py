# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:36:14 2018

@author: Hauke Brunken
"""


import numpy as np
import math
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
sd_found = False
try:
    import sounddevice as sd
    sd_found = True
except:
    print('Das Modul "sounddevice" fehlt. Es lässt sich per "pip install sounddevice" im "Anaconda Prompt" installieren.')

def playSound(sound, fs):
    sd.play(sound/10, fs)

def spectrum(Messwerte, Abtastrate): 
	N=len(Messwerte)
	u_cmplx=fft(Messwerte)
	u_abs=np.abs(u_cmplx[0:N//2])/N
	u_abs[1:] *= 2
	f=np.linspace(0,Abtastrate//2,N//2)
	return (f,u_abs)

def dataRead(**kwargs):
    argListMust = {'amplitude', 'samplingRate', 'duration', 'channels', 'resolution', 'outType'}
    argList = {'amplitude', 'samplingRate', 'duration', 'channels', 'resolution', 'outType', 'continues'}
    for key in kwargs:
        if key not in argList:
            print('Folgende Argumente müssen übergeben werden: amplitude=[V], samplingRate=[Hz], duration=[s], channels=[[,]], resolution=[bits], outType=\'volt\' oder \'codes\')')
            return None
    for key in argListMust:
        if key not in kwargs:
            print('Folgende Argumente müssen übergeben werden: amplitude=[V], samplingRate=[Hz], duration=[s], channels=[[,]], resolution=[bits], outType=\'volt\' oder \'codes\')')
            return None
    amplitude = kwargs['amplitude']
    samplingRate = kwargs['samplingRate']
    duration = kwargs['duration']
    channels = kwargs['channels']
    resolution = kwargs['resolution']
    outType = kwargs['outType']
    if 'continues' not in kwargs.keys():
        continues = False
    else:
        continues = kwargs['continues']
        if type(continues) != bool:
            print('continues muss vom Typ bool sein')
            return None
    if not all(i < 4 for i in channels):
        print('Mögliche Kanäle sind 0 bis 4')
        return None
    if len(channels) > len(set(channels)):
        print('Kanäle dürfen nicht doppelt auftauchen')
        return None

    
    outtType = outType.capitalize()
    if outtType != 'Volt' and outtType != 'Codes':
        print(outtType)
        print('outType = \'Volt\' oder \'Codes\'')
        return None
    
    u_lsb = 2*amplitude/(2**resolution-1)
    bins = [-amplitude+u_lsb/2+u_lsb*i for i in range(2**resolution-1)]
    

    ai_voltage_rngs = [1,2,5,10]
    if amplitude not in ai_voltage_rngs:
        print('Unterstützt werden folgende Amplituden:')
        print(ai_voltage_rngs)
        return None
    
    for channel in channels:
        if resolution < 1 or resolution > 14:
            print(f'Die Auflösung muss zwischen 1 und 14 Bit liegen')
            return None
    
    if samplingRate > 48000:
        print(f'Mit dieser Kanalanzahl beträgt die höchste Abtastrate 48000 Hz:')
        return None
    
    
    if continues:
        print('Die Liveansicht ist nicht verfügbar.')
    else:
        t = np.arange(0,duration,1/samplingRate)
        data = np.zeros( (len(channels),t.size))
        data[0,:] = 2.5*np.sin(2*np.pi*50*t+np.random.rand()*np.pi*2)
        data[data>amplitude] = amplitude
        data[data<-amplitude] = -amplitude
        data = np.digitize(data,bins)
        if outtType == 'Volt':
            data = data*u_lsb-amplitude
        print(f"Die Messung wurde durchgeführt mit einer virtuellen Messkarte \n Messbereich: +/-{amplitude:1.2f} Volt\n samplingRate: {samplingRate:1.1f} Hz\n Messdauer: {duration:1.3f} s\n Auflösung: {resolution:d} bit\n Ausgabe in: {outtType:s}")

    return data