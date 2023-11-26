# -*- coding: utf-8 -*-
from PIL import Image as IM
import numpy as np
from scipy import ndimage
from func_lib_v2 import (lum, gkern, filtAMP, filtMAX_AMP, filtPrevitt, filtRoberts, filtSobel,
                         filtMarrHildret, filtCanny, filtDeriche, filtKirsch,
                         thresholdGlobal, thresholdLocMidsum, nonMaxSuppresion, doubleThreshold,
                         fractMassRadiusRelation, fractLocBoxCounting, fractLocGray)
from FuncUI import(errno, defineStartInfo, defineMarrHildret, defineCanny, defineDeriche,
                   imsaveFromArray, imshowFromArray, imsaveColorFromArray, imshowColorFromArray,
                   colorFract)

#from archive import (colorFract2M, colorFract3M, colorFract35M, colorFract2Percent, colorFract4Percent)
import time

tStart = time.monotonic()

defaultFlag = True
filtSOSFlag = "show"
thresholdSOSFlag = "show"
invFlag = True
manualThresholdFlag = False
fractFlag = False
fractSOSFlag = "show"
prevBlur = False

filtMap = ["AMP","MAX_AMP","Roberts","Previtt","Sobel","Marr-Hildret","Canny","Deriche", "Kirsch"]
thresholdMap = ["glThr", "locMidsumThr", "DoubleThr"]
fractMap = ["M-R_rel", "BoxCount", "GrBoxCount", "GrTriangle", ""] #GRAY: frtype: 0, 1

tGlobKoef =      np.array([30., 32., 28., 30., 30., -1., -1., -1., 30.])

tLocClassKoef = np.array([10., 10., 10., 10., 10., -1., -1., -1., 30.])
tLocBgKoef =    np.array([12., 12., 12., 12., 12., -1., -1., -1., 30.])
tLocObjKoef =   np.array([13., 13., 13., 15., 15., -1., -1., -1., 30.])

tLowRatio =  np.array([8., 10., 11., 14., 12., -1., -1., -1., 30.])
tHighRatio = np.array([30., 32., 35., 35., 25., -1., -1., -1., 30.])

if (defaultFlag):
    filename = "tank6"
    filtSOSFlag = "pass" # "show" or "save" or "pass"
    thresholdSOSFlag = "pass"
    fractFlag = True
    fractType = 1
    fractSOSFlag = "show"
    invFlag = True
    f_type = 5
    t_type = 2

else:#filename, filtType, save/show images, thresholdType, manualThreshold        
    filename, f_type, filtSOSFlag, thresholdSOSFlag, t_type = defineStartInfo()
              
#Opening file
with IM.open(filename + ".jpg") as image:
    width = image.size[0]
    height = image.size[1]
    M_orig = np.zeros((height,width))
    for i in range (height):
        for j in range (width):
            r, g, b = image.getpixel((j,i))
            M_orig[i,j] = lum(r,g,b) #"grayscale" 
            
if prevBlur:                
    kerlen = 7; std = 1                           
    kernel_gauss = gkern(kerlen, std)#матрица гаусса    
    M_orig = ndimage.convolve(M_orig, kernel_gauss)

#Filtering
if (f_type != 0):
    filename = filename + "_" + filtMap[f_type - 1]
    if (f_type == 1):
        M, angle = filtAMP(M_orig)
        
    elif (f_type == 2):
        M, angle = filtMAX_AMP(M_orig)
        
    elif (f_type == 3):
        M, angle = filtRoberts(M_orig)
        
    elif (f_type == 4):
        M, angle = filtPrevitt(M_orig)
        
    elif (f_type == 5):
        M, angle = filtSobel(M_orig)
        
    elif (f_type == 6):
        kerlenMH, stdMH, lvlMH = defineMarrHildret()#define filter and threshold parameters
        M = filtMarrHildret(M_orig, kerlenMH, stdMH, lvlMH)   
        
    elif (f_type == 7):
        kerlenC, stdC, lowRatioC, highRatioC = defineCanny()#define filter and threshold parameters
        MDoubleThr = filtCanny(M_orig, kerlenC, stdC, lowRatioC, highRatioC)
        M = np.copy(MDoubleThr)
    elif (f_type == 8):
        alphaD, omegaD, lowRatioD, highRatioD = defineDeriche()#define filter and threshold parameters
        MDoubleThr = filtDeriche(M_orig, alphaD, omegaD, lowRatioD, highRatioD)
        M = np.copy(MDoubleThr)
    elif (f_type == 9):
        M, alpha = filtKirsch(M_orig)
        
    if (filtSOSFlag == "save"):
        imsaveFromArray(M,filename,invFlag)
    elif (filtSOSFlag == "show"):
        imshowFromArray(M, invFlag)
        
else:#all gradient filters
    M1, angle1 = filtAMP(M_orig)
    M2, angle2 = filtMAX_AMP(M_orig)   
    M3, angle3 = filtRoberts(M_orig)
    M4, angle4 = filtPrevitt(M_orig)
    M5, angle5 = filtSobel(M_orig)
    
    MStack = np.stack((M1,M2,M3,M4,M5))#stack of gradient matrix
    angleStack = np.stack((angle1,angle2,angle3,angle4,angle5))#stack of angle matrix
    
    lenMStack = len(MStack)
    filenameStack = [filename] * (lenMStack)
    for i in range (lenMStack):
        filenameStack[i] = filenameStack[i] + "_" + filtMap[i]
        if(filtSOSFlag == "save"):
            imsaveFromArray(MStack[i], filenameStack[i] , invFlag)
        elif(filtSOSFlag == "show"):
            imshowFromArray(MStack[i], invFlag)
            
tFilt = time.monotonic()
print(f'Filt time: {(tFilt - tStart):.3f}')
#Threshold filtering
if ((not manualThresholdFlag) and fractType != 2 and fractType != 3):#default, fractType is not for grayscale
    if (t_type == 0):#global threshold
        if (f_type == 0):#all gradient filters
            for i in range(lenMStack):
                filenameStack[i] = filenameStack[i] + "_" + thresholdMap[i] + '_' + str(tGlobKoef[i])   
                
                mmax = np.max(MStack[i])
                globalLvl = mmax * tGlobKoef[i] / 100.               
                MiGlobal = thresholdGlobal(MStack[i], globalLvl)               
                if(thresholdSOSFlag == "save"):
                    imsaveFromArray(MiGlobal, filenameStack[i] , invFlag)
                elif(thresholdSOSFlag == "show"):
                    imshowFromArray(MiGlobal, invFlag)
        elif (f_type > 0 and f_type < 6 or f_type == 9):
            filename = filename + thresholdMap[t_type] + str(tGlobKoef[f_type - 1])           
            mmax = np.max(M)
            globalLvl = mmax * tGlobKoef[f_type - 1] / 100.            
            MGlobal = thresholdGlobal(M, globalLvl)            
            if(thresholdSOSFlag == "save"):
                imsaveFromArray(MGlobal, filename, invFlag)
            elif(thresholdSOSFlag == "show"):
                imshowFromArray(MGlobal, invFlag)                
    if (t_type == 1):#locMidsum threshold
        if (f_type == 0):
            for i in range(lenMStack):
                filenameStack[i] = (filenameStack[i] + "_" + thresholdMap[t_type] + '_' 
                                    + str(tLocClassKoef[i]) + '_' + str(tLocBgKoef[i]) + '_' 
                                    + str(tLocObjKoef[i]))              
                mmax = np.max(MStack[i])
                ClassLvl = mmax * tLocClassKoef[i] / 100.
                BgLvl = mmax * tLocBgKoef[i] / 100.
                ObjLvl = mmax * tLocObjKoef[i] / 100.             
                MiLocMidsum = thresholdLocMidsum(MStack[i], ClassLvl, BgLvl, ObjLvl)              
                if(thresholdSOSFlag == "save"):
                    imsaveFromArray(MiLocMidsum, filenameStack[i] , invFlag)
                elif(thresholdSOSFlag == "show"):
                    imshowFromArray(MiLocMidsum, invFlag)
        elif (f_type > 0 and f_type < 6 or f_type == 9):
            filename = (filename + "_" + thresholdMap[t_type] + '_' + str(tLocClassKoef[f_type - 1]) + '_' 
                        + str(tLocBgKoef[f_type - 1]) + '_' + str(tLocObjKoef[f_type - 1]))         
            mmax = np.max(M)
            ClassLvl = mmax * tLocClassKoef[f_type - 1] / 100.
            BgLvl = mmax * tLocBgKoef[f_type - 1] / 100.
            ObjLvl = mmax * tLocObjKoef[f_type - 1] / 100.
            MLocMidsum = thresholdLocMidsum(M, ClassLvl, BgLvl, ObjLvl)               
            if(thresholdSOSFlag == "save"):
                imsaveFromArray(MLocMidsum, filename , invFlag)
            elif(thresholdSOSFlag == "show"):
                imshowFromArray(MLocMidsum, invFlag)                
    if (t_type == 2):#double threshold
        if (f_type == 0):
            for i in range(lenMStack):    
                filenameStack[i] = (filenameStack[i] + "_" + thresholdMap[t_type] + '_' 
                                    + str(tLowRatio[i]) + '_' + str(tHighRatio[i]))              
                MiSup = nonMaxSuppresion(MStack[i], angleStack[i])
                MiDoubleThr = doubleThreshold(MiSup, tLowRatio[i], tHighRatio[i])               
                if(thresholdSOSFlag == "save"):
                    imsaveFromArray(MiDoubleThr, filenameStack[i] , invFlag)
                elif(thresholdSOSFlag == "show"):
                    imshowFromArray(MiDoubleThr, invFlag)                                    
        elif (f_type > 0 and f_type < 6 or f_type == 9):
                filename = (filename + '_' + thresholdMap[t_type] + '_' 
                            + str(tLowRatio[f_type - 1]) + '_' + str(tHighRatio[f_type - 1]))
                MSup = nonMaxSuppresion(M, angle)
                MDoubleThr = doubleThreshold(MSup, tLowRatio[f_type - 1], tHighRatio[f_type - 1])
                if(thresholdSOSFlag == "save"):
                    imsaveFromArray(MDoubleThr, filename , invFlag)
                elif(thresholdSOSFlag == "show"):
                    imshowFromArray(MDoubleThr, invFlag)
elif(fractType != 2 and fractType != 3): #fractType is not for grayscale
    if(f_type > 0 and f_type < 6 or f_type == 9):
        if (t_type == 0):#global threshold
            while True:#global lvl
                print ("Type global level in percents of max")
                tGlobKoef = input()
                try:
                    tGlobKoef = float(tGlobKoef)
                except ValueError:
                    errno(1)
                else:
                    if (0 <= tGlobKoef <= 100):
                        break
                    else:
                        print(tGlobKoef," - global level must be in [0,100]")
                        continue
            filename = filename + '_' + thresholdMap[t_type] + '_' + str(tGlobKoef)              
            mmax = np.max(M)
            globalLvl = mmax * tGlobKoef/ 100.        
            MGlobal = thresholdGlobal(M, globalLvl)  
            if(thresholdSOSFlag == "save"):
                imsaveFromArray(MGlobal, filename, invFlag)
            elif(thresholdSOSFlag == "show"):
                imshowFromArray(MiGlobal, invFlag)                    
        if(t_type == 1):#locMidsum threshold
            while True:#class lvl
                print ("Type class level in percents of max")
                tLocClassKoef = input()
                try:
                    tLocClassKoef = float(tLocClassKoef)
                except ValueError:
                    errno(1)
                else:
                    if (0 <= tLocClassKoef <= 100):
                        break
                    else:
                        print(tLocClassKoef," - class level must be in [0,100]")
                        continue
            while True:#bg lvl
                print ("Type background level in percents of max")
                tLocBgKoef = input()
                try:
                    tLocBgKoef = float(tLocBgKoef)
                except ValueError:
                    errno(1)
                else:
                    if  (0 <= tLocBgKoef <= 100):
                        break
                    else:
                        print(tLocBgKoef," - background level must be in [0,100]")
                        continue
            while True:#obj lvl
                print ("Type object level in percents of max")
                tLocObjKoef = input()
                try:
                    tLocBgKoef = input()
                except ValueError:
                    errno(1)
                else:
                    if (0 <= tLocObjKoef <= 100):
                        break
                    else:
                        print(tLocObjKoef," - object level must be in [0,100]")
                        continue
            filename = (filename + "_" + thresholdMap[t_type] + '_' + str(tLocClassKoef) + '_' 
                        + str(tLocBgKoef) + '_' + str(tLocObjKoef))           
            mmax = np.max(M)
            ClassLvl = mmax * tLocClassKoef / 100.
            BgLvl = mmax * tLocBgKoef / 100.
            ObjLvl = mmax * tLocObjKoef / 100.
            MLocMidsum = thresholdLocMidsum(M, ClassLvl, BgLvl, ObjLvl)                
            if(thresholdSOSFlag == "save"):
                imsaveFromArray(MiGlobal, filename , invFlag)
            elif(thresholdSOSFlag == "show"):
                imshowFromArray(MiGlobal, invFlag)
        if(t_type == 2):#double threshold
            while True:#lowRatio
                print("Type lowRatio for the double threshold")
                tLowRatio = input()
                try:
                    tLowRatio = float(tLowRatio)
                except ValueError:
                    errno(1)
                else:
                    if (0 <= tLowRatio <= 100):
                        break
                    else:
                        print(tLowRatio, " - must be in [0,100]")
                        continue
            while True:#highRatio
                print("Type highRatio for the double threshold")
                tHighRatio = input()
                try:
                    tHighRatio = float(tHighRatio)
                except ValueError:
                    errno(1)
                else:
                    if (0 <= tHighRatio <= 100):
                        break
                    else:
                        print(tHighRatio, " - must be in [0,100]")
                        continue
            filename = (filename + '_' + thresholdMap[t_type] + '_' 
                        + str(tLowRatio) + '_' + str(tHighRatio))
            MSup = nonMaxSuppresion(M, angle)
            MDoubleThr = doubleThreshold(MSup, tLowRatio, tHighRatio)
            if(thresholdSOSFlag == "save"):
                imsaveFromArray(MDoubleThr, filename , invFlag)
            elif(thresholdSOSFlag == "show"):
                imshowFromArray(MDoubleThr, invFlag)
    if(f_type == 0):
        if(t_type == 0):#global threshold
            for i in range (lenMStack):
                while True:
                    print("Type global level for ",filtMap[i], " filter")
                    tGlobKoef = input()
                    try:
                        tGlobKoef = float(tGlobKoef)
                    except ValueError:
                        errno(1)
                    else:
                        if (0 <= tGlobKoef <= 100):
                            break
                        else:
                            print("global level must be in [0,100]")
                            continue                        
                filenameStack[i] = (filenameStack[i] + '_' + thresholdMap[t_type] + '_'
                                    + str(tGlobKoef))
                mmax = np.max(MStack[i])
                globalLvl = mmax * tGlobKoef / 100.
                MiGlobal = thresholdGlobal(MStack[i], globalLvl)
                if(thresholdSOSFlag == "save"):
                    imsaveFromArray(MiGlobal, filenameStack[i], invFlag)
                elif(thresholdSOSFlag == "show"):
                    imshowFromArray(MiGlobal, invFlag)                  
        if(t_type == 1):#locMidsum threshold
            for i in range(lenMStack):    
                while True:#class lvl
                    print ("Type class level in percents of max for", filtMap[i], "filter")
                    tLocClassKoef = input()
                    try:
                        tLocClassKoef = float(tLocClassKoef)
                    except ValueError:
                        errno(1)
                    else:
                        if (0 <= tLocClassKoef <= 100):
                            break
                        else:
                            print(tLocClassKoef," - class level must be in [0,100]")
                            continue
                while True:#bg lvl
                    print ("Type background level in percents of max for", filtMap[i], "filter")
                    tLocBgKoef = input()
                    try:
                        tLocBgKoef = float(tLocBgKoef)
                    except ValueError:
                        errno(1)
                    else:
                        if  (0 <= tLocBgKoef <= 100):
                            break
                        else:
                            print(tLocBgKoef," - background level must be in [0,100]")
                            continue
                while True:#obj lvl
                    print ("Type object level in percents of max for", filtMap[i], "filter")
                    tLocObjKoef = input()
                    try:
                        tLocBgKoef = input()
                    except ValueError:
                        errno(1)
                    else:
                        if (0 <= tLocObjKoef <= 100):
                            break
                        else:
                            print(tLocObjKoef," - object level must be in [0,100]")
                            continue                        
                filenameStack[i] = (filenameStack[i] + "_" + thresholdMap[t_type] + '_' + str(tLocClassKoef) + '_' 
                            + str(tLocBgKoef) + '_' + str(tLocObjKoef))                
                mmax = np.max(MStack[i])
                ClassLvl = mmax * tLocClassKoef / 100.
                BgLvl = mmax * tLocBgKoef / 100.
                ObjLvl = mmax * tLocObjKoef / 100.
                MiLocMidsum = thresholdLocMidsum(MStack[i], ClassLvl, BgLvl, ObjLvl)                   
                if(thresholdSOSFlag == "save"):
                    imsaveFromArray(MiLocMidsum, filenameStack[i], invFlag)
                elif(thresholdSOSFlag == "show"):
                    imshowFromArray(MiLocMidsum, invFlag)
        if(t_type == 2):#double threshold
            for i in range(lenMStack):
                while True:#lowRatio
                    print("Type lowRatio for the double threshold")
                    tLowRatio = input()
                    try:
                        tLowRatio = float(tLowRatio)
                    except ValueError:
                        errno(1)
                    else:
                        if (0 <= tLowRatio <= 100):
                            break
                        else:
                            print(tLowRatio, " - must be in [0,100]")
                            continue
                while True:#highRatio
                    print("Type highRatio for the double threshold")
                    tHighRatio = input()
                    try:
                        tHighRatio = float(tHighRatio)
                    except ValueError:
                        errno(1)
                    else:
                        if (0 <= tHighRatio <= 100):
                            break
                        else:
                            print(tHighRatio, " - must be in [0,100]")
                            continue
                filenameStack[i] = (filenameStack[i] + '_' + thresholdMap[t_type] + '_' 
                            + str(tLowRatio) + '_' + str(tHighRatio))
                MiSup = nonMaxSuppresion(MStack[i], angleStack[i])
                MiDoubleThr = doubleThreshold(MiSup, tLowRatio, tHighRatio)
                if(thresholdSOSFlag == "save"):
                    imsaveFromArray(MiDoubleThr, filenameStack[i], invFlag)
                elif(thresholdSOSFlag == "show"):
                    imshowFromArray(MiDoubleThr, invFlag)                

tThresh = time.monotonic()
print(f'Threshold time: {(tThresh - tFilt):.3f}')                    
#Fractal

if (fractFlag and f_type != 0):
    filename += '_fract_'

#REWORKING
'''
if (fractFlag and f_type != 0):
    filename += '_fract_'+ fractMap[fractType]
    if (fractType == 0):
        if (f_type != 6):
            fractM = fractMassRadiusRelation(MDoubleThr)
    elif (fractType == 1):
        if (f_type != 6):
            fractM = fractLocBoxCounting(MDoubleThr)
    elif (fractType == 2):    
        if (f_type != 6 and f_type != 7 and f_type != 8):
            fractM = fractLocGray(M, 0)
    elif (fractType == 3):
        if (f_type != 6 and f_type != 7 and f_type != 8):
            fractM = fractLocGray(M, 1)
    colorImg = colorFract(fractM)
    if (fractSOSFlag == "save"):
        imsaveColorFromArray(colorImg, filename)
    elif (fractSOSFlag == "show"):
        imshowColorFromArray(colorImg)

tFract = time.monotonic()
print("Fract time:", tFract - tThresh)    
print("All time:", tFract - tStart)
'''