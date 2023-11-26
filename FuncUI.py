# -*- coding: utf-8 -*-
from PIL import Image as IM
import PIL.ImageOps
import numpy as np
#import sys
#np.set_printoptions(threshold = sys.maxsize)
'''
Defenition of the most of console UI
'''
filtMap = ["AMP","MAX_AMP","Roberts","Previtt","Sobel","Marr-Hildret","Canny","Deriche", "Kirsch"]
thresholdMap = ["globalThreshold", "locMidsumThreshold", "DoubleThreshold"]
fractMap = ["MassRadiusRel", "BoxCount", "GrBoxCount", "GrTriangle"] #GRAY: frtype: 0, 1

def defineStartInfo():
    '''
    Manual input of all nedded properties
    
    Returns
    -------
    filename : string
    f_type : int in [0;9] to choose one of 9 filters or all of them to compare
    filtSOSFlag : string: save or show (pass) filtered images
    thresholdSOSFlag : string: save or show (pass) images after threshold filtration
    t_type : int in [0;1] to choose откуда берутся параметры пороговой фильтрации

    '''
    while True:#filename
        print("Enter filename")
        filename = input()
        try:
            test = IM.open(filename + ".jpg")
        except FileNotFoundError:
            print("File not found!\n")
        else:
            test.close()
            break    
        
    while True:#filtType
        print("Choose filter type\n")
        print("0 all gradient filters")
        for i in range(len(filtMap)):
            print(i+1, filtMap[i])
        f_type = input()
        try:
            f_type = int(f_type)
        except ValueError:
            errno(0)
        else:
            if ((f_type < 0) or (f_type > 9)):
                print("Type int in range 0 to 9\n")
                continue
            else:
                break    
            
    print("Save filtered images or just show?\nType 'save' or 'show' or 'pass' without braces")
    while True:#save/show filt images
        wMode = input()
        if (wMode == 'save' or wMode == 'show' or wMode == 'pass'):
            filtSOSFlag = wMode
            break
        else:
            print("Type 'save' or 'show' without braces")
            
    print("Save threshold images or just show?\nType 'save' or 'show' or 'pass' without braces")
    while True:#save/show threshold images
        wMode = input()
        if (wMode == 'save' or wMode == 'show' or wMode == 'pass'):
            thresholdSOSFlag = wMode
            break
        else:
            print("Type 'save' or 'show' or 'pass' without braces")
            
    if (f_type != 6 and f_type != 7 and f_type != 8):       
        while True:#thresholdType
            print("Choose threshold type:\n")
            for i in range(len(thresholdMap)):
                print(i, thresholdMap[i])
            t_type = input()
            try:
                t_type = int(t_type)
            except ValueError:
                errno(0)
            else:
                if ((t_type < 0) or (t_type > 2)):
                    print("Type int in range 0 to 2")
                    continue
                else:
                    break
                
        while True:#manualThreshold
            print("Manual threshold or not?\nType 1 or 0")
            manualThresholdFlag = input()
            try:
                manualThresholdFlag = int(manualThresholdFlag)
            except ValueError:
                errno(0)
            else:
                if (manualThresholdFlag == 0 or manualThresholdFlag == 1):
                    manualThresholdFlag = bool(manualThresholdFlag)
                    break
                else:
                    print("Type 1 for manual threshold, 0 - default levels")
                    continue
    return filename, f_type, filtSOSFlag, thresholdSOSFlag, t_type

def defineMarrHildret():

    while True: #define kerlen
            kerlenMH = input("Type kerlen for Marr-Hildret filter in [3,21]\n")
            try:
                kerlenMH = int(kerlenMH)
            except ValueError:
                errno(0)
            else:
                if (kerlenMH < 3):
                    print(kerlenMH," - kerlen must be more then 3")
                elif(kerlenMH > 21):
                    print(kerlenMH, " - kerlen is too big (must be under the 21)")
                else:
                    break                
    while True: #define deviation
        print("Type deviation for Marr-Hildret filter\n")
        deviationMH = input("Firstly try deviation<=(kerlen-1)/6\n")
        try:
            deviationMH = float(deviationMH)
        except ValueError:
            errno(1)
        else:
            if (deviationMH < 0.5):
                print(deviationMH, " - deviation is too low; must be over 0.5")
                continue
            elif(deviationMH > kerlenMH/3):
                print(deviationMH, " - deviation is too high; must be lower then kerlen/3")
                continue
            else:
                break            
    while True: #define zero-level
        print("Type level for finding zero points; level must be in (0.01, 1)")
        lvlMH = input()
        try:
            lvlMH = float(lvlMH)
        except ValueError:
            errno(1)
        else:
            if (0.01 < lvlMH < 1):
                break
            else:
                print(lvlMH, " - level must be in (0.01, 1)")
                continue   
    return kerlenMH, deviationMH, lvlMH

def defineCanny():

    while True: #define kerlen
        print("Type kerlen for Canny filter")
        kerlenC = input()
        try:
            kerlenC = int(kerlenC)
        except ValueError:
            errno(1)
        else:
            if (kerlenC < 3):
                print(kerlenC," - kerlen must be more then 3")
                continue
            elif(kerlenC > 21):
                print(kerlenC, " - kerlen is too big (must be under the 21)")
                continue
            else:
                break            
    while True: #define deviation
        print("Type deviation for Canny filter\nFirstly try deviation<=(kerlen-1)/6")
        deviationC = input()
        try:
            deviationC = float(deviationC)
        except ValueError:
            errno(1)
        else:
            if (deviationC < 0.5):
                print(deviationC, " - deviation is too low; must be over 0.5")
                continue
            elif(deviationC > kerlenC/3):
                print(deviationC, " - deviation is too high; must be lower then kerlen/3")
                continue
            else:
                break            
    while True: #define lowRatio
        print("Type level for Canny lowRatio threshold")
        lowRatioC = input()
        try:
            lowRatioC = float(lowRatioC)
        except ValueError:
            errno(1)
        else:
            if (1 < lowRatioC < 50):
                break
            else:
                print(lowRatioC, " - level must be in (1, 50)")
                continue           
    while True: #define highRatio
        print("Type level for Canny highRatio threshold")
        highRatioC = input()
        try:
            highRatioC = float(highRatioC)
        except ValueError:
            errno(1)
        else:
            if (10 < highRatioC < 100):
                break
            else:
                print(highRatioC, " - level must be in (10, 100)")
                continue
            
    return kerlenC, deviationC, lowRatioC, highRatioC

def defineDeriche():

    while True: #define alpha
        print("Type alpha for Deriche filter")
        alphaD = input()
        try:
            alphaD = float(alphaD)
        except ValueError:
            errno(0)
        else:
            if (alphaD < 0.1):
                print(alphaD," - kerlen must be more then 0.1")
                continue
            elif(alphaD > 10):
                print(alphaD, " - kerlen is too big (must be under the 10)")
                continue
            else:
                break           
    while True: #define deviation
        print("Type omega for Deriche filter\nFirstly try omega<=1")
        omegaD = input()
        try:
            omegaD = float(omegaD)
        except ValueError:
            errno(1)
        else:
            if (omegaD < 0.01):
                print(omegaD, " - deviation is too low; must be over 0.01")
                continue
            elif(omegaD > np.pi/2.):
                print(omegaD, " - deviation is too high; must be lower then Pi/2")
                continue
            else:
                break         
    while True: #define lowRatio
        print("Type level for Deriche lowRatio threshold")
        lowRatioD = input()
        try:
            lowRatioD = float(lowRatioD)
        except ValueError:
            errno(1)
        else:
            if (1 < lowRatioD < 50):
                break
            else:
                print(lowRatioD, " - level must be in (1, 50)")
                continue
    while True: #define highRatio
        print("Type level for Deriche highRatio threshold")
        highRatioD = input()
        try:
            highRatioD = float(highRatioD)
        except ValueError:
            errno(1)
        else:
            if (10 < highRatioD < 100):
                break
            else:
                print(highRatioD, " - level must be in (10, 100)")
                continue          
    return alphaD, omegaD, lowRatioD, highRatioD


def colorFract(fractM, mode = "DEFAULT", start = 30, stop = 160, percStep = 3, percMid = 50):
    '''
    Покраска "карты" локальной фрактальной размерности
    ----------
    Parameters
    ----------
    fractM : Карта фрактальных размерностей
    mode : string, optional
        "DEFAULT" - lin space in [start,stop],
        "PERCENT" - percents from np.max(fractM). 

    start : float, optional
        The default is 30.
    stop : TYPE, optional
        The default is 160.
    percStep : TYPE, optional
        The default is 3.
    percMid : TYPE, optional
        The default is 50.

    Returns
    -------
    None.

    '''
    
    COLOR_PALLETE25 = [[255,225,225],[255,132,105],[224,67,39],[210,65,89],[110,16,0],[188,98,48],[222,147,39],
[186,175,69],[214,220,53],[126,219,72],[0,184,40],[109,182,79],[102,201,140],[99,187,182],[111,150,205],[120,120,215],
[140,81,229],[88,43,165],[96,59,126],[200,78,142],[206,83,203],[185,119,139],[174,174,170],[117,113,113],[0,0,0]]
    cl = np.shape(COLOR_PALLETE25)[0]
    if (mode == "DEFAULT"):
        if (stop < start):
            a = stop
            stop = start
            start = a
        if (start < 0 or stop < 0):
            print ("Something negative")
            return -1
        lvl = np.linspace(start = start, stop = stop, num = cl - 2)
        stepsLvl = lvl / 100.
    elif (mode == "PERCENT"):#нужно написать изменение положения центрального "распределения"
        if (percStep == 2):
            lvl = np.array([15, 20, 25, 30, 34, 37, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 63, 66, 70, 75, 80, 85])
        elif (percStep == 3):
            lvl = np.array([15, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 57, 60, 63, 66, 69, 72, 75, 80, 85])
        else:# (percStep == 4):
            lvl = np.array([6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94])
        stepsLvl = lvl * np.max(fractM) / 100.
    else:
        print("Error. Wrong mode")
        return -1
    #np.set_printoptions(threshold=sys.maxsize)
    #print(len(lvl), len(COLOR_PALLETE25))
    #for i in range(len(stepsLvl)):
    #    print(f'{stepsLvl[i]:.2f}')
    height, width = np.shape(fractM)
    genM = np.zeros((height, width, 3), dtype = np.uint8)

    
    ii, jj = np.where(fractM <= stepsLvl[0])#1
    genM[ii, jj] = COLOR_PALLETE25[0]
    
    ii, jj = np.where(fractM > stepsLvl[-1])#25
    genM[ii, jj] = COLOR_PALLETE25[-1]
    
    for ii in range(height):
        for jj in range(width):
            for ic in range(1, cl-2):#2-24
                if (stepsLvl[ic - 1] < fractM[ii, jj] <= stepsLvl[ic]):
                    genM[ii,jj] = COLOR_PALLETE25[ic]
    return genM

def imsaveFromArray(array, name, invFlag = True):

    img = IM.fromarray(array)
    img = img.convert("L")
    print("saved")
    if (invFlag == True):
        img = PIL.ImageOps.invert(img)
        img.save(name+'.jpg')
    else:
        img.save(name+"_noInv"+'.jpg')

def imshowFromArray(array, invFlag = True):

    img = IM.fromarray(array)
    img = img.convert("L")
    if (invFlag == True):
        img = PIL.ImageOps.invert(img)
    img.show()
    
def imsaveColorFromArray(colorM, name):

    img = IM.fromarray(colorM, 'RGB')
    img.save(name+'.jpg')
    print("saved")
    
def imshowColorFromArray(colorM):

    img = IM.fromarray(colorM, 'RGB')
    img.show()

def errno(flag):

    if(flag == 0):
        print("Not an int number!\n")
    elif(flag == 1):
        print("Must be a real number\n")