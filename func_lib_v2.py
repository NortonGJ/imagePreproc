
# -*- coding: utf-8 -*-
import numpy as np
import math as m
from scipy import signal as sg
from scipy import ndimage
from sklearn.linear_model import LinearRegression
import queue
     
def lum (r, g, b):
    '''
    Стандартная формула вычисления яркости пикселя в RGB-пространстве
    ----------
    Parameters
    ----------
    r : red
    g : green
    b : blue

    Returns
    -------
    float; Значение яркости пикселя
    '''
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

#prefer to use IM.convert("L")
def toGray(r, g, b):
    '''
    Стандартная формула преобразования пикселя RGB-пространства в оттенки серого
    ----------
    Parameters
    ----------
    r : red
    g : green
    b : blue

    Returns
    -------
    float; pixel's gray value
    '''    
    return 0.299 * r + 0.587 * g + 0.114 * b
    
def filtAMP(M_orig):# разностный амплитудный фильтр; без центра
    '''
    Разностный амплитудный фильтр
    ----------
    Parameters
    ----------
    M_orig : Матрица содержащая в себе значения пикселей изображения после преобразования в оттенки серого
    
    Returns
    -------
    G : Модуль градиента результирующей матрицы
    angle : Угловая составляющая градиента результирующей матрицы
    '''
    Gx = ndimage.convolve(M_orig, amp_x)
    Gy = ndimage.convolve(M_orig, amp_y)    
    G = np.hypot(Gx, Gy)
    G = G / np.max(G) * 255.  
    theta = np.arctan2(Gy,Gx)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180.    
    return G, angle

def filtMAX_AMP(M_orig):# максимальный разностный амплитудный фильтр; без центра
    '''
    Максимальный разностный амплитудный фильтр
    ----------
    Parameters
    ----------
    M_orig : Матрица содержащая в себе значения пикселей изображения после преобразования в оттенки серого
    
    Returns
    -------
    G : Модуль градиента результирующей матрицы
    angle : Угловая составляющая градиента результирующей матрицы
    '''
    height, width = np.shape(M_orig)
    G = np.zeros((height, width))
    angle = np.zeros((height, width))
    for i in range (height):
        for j in range (width):
            try:
                area = M_orig[i:(i+2), j:(j+2)]
                G[i,j] = abs(np.max(area) - np.min(area))
                angle[i,j] = angleMAX_AMP(area)
            except IndexError:
                pass
    return G, angle

def angleMAX_AMP(area):
    '''
    Функция определения угловой составляющей градиента в случае применения макс. амп. фильтра
    ----------
    Parameters
    ----------
    area : int matrix 2х2

    Returns
    -------
    np.int32 : Угловая составляющая градиента 

    '''
    angle = 0
    iMax = np.argmax(area)
    iMin = np.argmin(area)
    if ((iMax & 1 + iMin & 1) == 0):
        angle = np.int32(90)
    elif ((iMax + iMin == 5) or (iMax + iMin == 1) or iMin == iMax):
        angle = np.int32(0)
    elif (np.max((iMax, iMin)) == 2):
        angle = np.int32(45)
    elif (np.max((iMax, iMin)) == 3):
        angle = np.int32(135)
    return angle

def filtRoberts(M_orig):# фильтр Робертса; без центра
    '''
    Фильтр Робертса
    ----------
    Parameters
    ----------
    M_orig : Матрица содержащая в себе значения пикселей изображения после преобразования в оттенки серого
    
    Returns
    -------
    G : Модуль градиента результирующей матрицы
    angle : Угловая составляющая градиента результирующей матрицы
    '''
    Gx = ndimage.convolve(M_orig,roberts_x)
    Gy = ndimage.convolve(M_orig,roberts_y)
    G = np.hypot(Gx,Gy)
    theta = np.arctan2(Gy,Gx)   
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180.
    return G, angle

def filtPrevitt(M_orig):
    '''
    Фильтр Превитта
    ----------
    Parameters
    ----------
    M_orig : Матрица содержащая в себе значения пикселей изображения после преобразования в оттенки серого
    
    Returns
    -------
    G : Модуль градиента результирующей матрицы
    angle : Угловая составляющая градиента результирующей матрицы
    '''
    Gx = ndimage.convolve(M_orig,previtt_x)
    Gy = ndimage.convolve(M_orig,previtt_y)  
    G = np.hypot(Gx,Gy)
    theta = np.arctan2(Gy,Gx)    
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180.
    return G, angle

def filtSobel(M_orig):
    '''
    Фильтр Собела
    ----------
    Parameters
    ----------
    M_orig : Матрица содержащая в себе значения пикселей изображения после преобразования в оттенки серого
    
    Returns
    -------
    G : Модуль градиента результирующей матрицы
    angle : Угловая составляющая градиента результирующей матрицы
    '''
    Gx = ndimage.convolve(M_orig,sobel_x)
    Gy = ndimage.convolve(M_orig,sobel_y)   
    G = np.hypot(Gx,Gy)
    theta = np.arctan2(Gy,Gx)    
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180.
    return G, angle

def gkern(kerlen, std):
    '''
    Функция генерации Гауссиана
    ----------
    Parameters
    ----------
    kerlen : int;
        Размер ядра матрицы Гаусса
    std : float;
        Значение отклонения (разброса) для построения матрицы Гаусса.

    Returns
    -------
    gkern2d : Гауссиан размером [kerlen, kerlen]

    '''
    gkern1d = sg.gaussian(kerlen, std=std).reshape(kerlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    scale = gkern2d[0,0]
    gkern2d /= scale
    gkern2d = np.int32(gkern2d)
    gkern2d = np.float64(gkern2d)
    gkern2d /= np.sum(gkern2d)
    return gkern2d

def filtMarrHildret (M_orig, kerlen = 15, deviation = 3, lvl = 0.65):
    '''
    Фильтр Марра-Хилдрета
    ----------
    Parameters
    ----------
    M_orig : Матрица содержащая в себе значения пикселей изображения после преобразования в оттенки серого
    kerlen : int, optional
        Размер ядра матрицы Гаусса. The default is 15.
    deviation : float, optional
        Значение отклонения (разброса) для построения матрицы Гаусса. The default is 3.
        optional: kerlen >= 3 * deviation + 1
    lvl : float, optional
        Уровень сравнения с нулем. The default is 0.65.

    Returns
    -------
    M3 : Матрица изображения обработанного фильтром Марра-Хилдрета

    '''
    kernel_gauss = gkern(kerlen, deviation) #Матрица Гаусса    
    M_orig_s = M_orig / 255.
    M = ndimage.convolve(M_orig_s, kernel_gauss)
    M2 = ndimage.convolve(M, laplassian_mask)    
    height, width = np.shape(M2) 
    M3 = np.zeros((height-2, width-2))        
    for i in range (1, height - 1, 1):
        for j in range (1, width - 1, 1):           
            #diagonal upleft-downright
            if(M2[i-1, j-1] * M2[i+1, j+1] < 0):
                if(abs(M2[i-1, j-1] - M2[i+1, j+1])>lvl):
                    M3[i-2, j-2] = 255
            #diagonal downleft-upright
            elif(M2[i+1, j-1] * M2[i-1, j+1] < 0):
                if(abs(M2[i+1, j-1] - M2[i-1, j+1])>lvl):
                    M3[i-2, j-2] = 255
            #horisontal
            elif(M2[i, j-1] * M2[i, j+1] < 0):
                if(abs(M2[i, j-1] - M2[i, j+1])>lvl):
                    M3[i-2, j-2] = 255                  
            #vertical
            elif(M2[i-1, j] * M2[i+1, j] < 0):
                if(abs(M2[i-1, j] - M2[i+1, j])>lvl):
                    M3[i-2, j-2] = 255
    return M3

def filtCanny(M_orig, kerlen = 15, deviation = 3, lowRatio = 10, highRatio = 30):
    '''
    Фильтр Кэнни
    ----------
    Parameters
    ----------
    M_orig : Матрица содержащая в себе значения пикселей изображения после преобразования в оттенки серого
    kerlen : int, optional
        Размер ядра матрицы Гаусса. The default is 15.
    deviation : float, optional
        Значение отклонения (разброса) для построения матрицы Гаусса. The default is 3.
        optional: kerlen >= 3 * deviation + 1
    lowRatio : float, optional; must be in [0,100]
        Нижний порог двухуровневой фильтрации. The default is 10.
    highRatio : TYPE, optional; must be in [0,100]
        Верхний порог двухуровневой фильтрации. The default is 30.

    Returns
    -------
    res : Матрица изображения обработанная фильтром Кэнни

    '''
    kernel_gauss = gkern(kerlen, deviation)    
    M = ndimage.filters.convolve(M_orig, kernel_gauss)    
    Gx = ndimage.filters.convolve(M, sobel_x)
    Gy = ndimage.filters.convolve(M, sobel_y) 
    G = np.hypot(Gx,Gy)
    G = G / G.max() * 255   
    theta = np.arctan2(Gy,Gx)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180.    
    M_sup = nonMaxSuppression(G, angle)
    res = doubleThreshold(M_sup, lowRatio, highRatio)    
    return res

def filtDeriche(M_orig, alpha = 0.75, omega = 0.2, lowRatio = 0.05, highRatio = 0.2):
    '''
    
    ----------
    Parameters
    ----------
    M_orig : Матрица содержащая в себе значения пикселей изображения после преобразования в оттенки серого
    alpha : float, optional
        Параметр alpha для расчета IIR-фильтра. The default is 0.75.
    omega : float, optional
        Параметр omega для расчета IIR-фильтра. The default is 0.2.
    lowRatio : float, optional
        Нижний порог двухуровневой фильтрации. The default is 0.05.
    highRatio : float, optional
        Верхний порог двухуровневой фильтрации. The default is 0.2.

    Returns
    -------
    res : Матрица изображения обработанная фильтром Дерише

    '''
    kerlen = 15
    X, Y = DericheMask(kerlen, alpha, omega)   
    Gx = ndimage.filters.convolve(M_orig, X)
    Gy = ndimage.filters.convolve(M_orig, Y) 
    G = np.hypot(Gx,Gy)
    G = G / G.max() * 255
    theta = np.arctan2(Gy,Gx)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180.  
    M_sup = nonMaxSuppression(G, angle)
    res = doubleThreshold(M_sup, lowRatio, highRatio)   
    return res

def nonMaxSuppression (G, angle):
    '''
    Функция немаксимального подавления
    ----------
    Parameters
    ----------
    G : Матрица модуля градиента матрицы изображения
    angle : Матрица содержащая угловой коэффициент градиента матрицы изображения

    Returns
    -------
    res : Матрица фильтрованного изображения после немаксимального подавления

    '''
    height, width = np.shape(G)
    res = np.zeros((height, width))
    for i in range (1, height-1):
        for j in range(1, width-1):
            try:
                q = 255
                r = 255    
                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = G[i,j+1]
                    r = G[i,j-1]    
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = G[i+1,j-1]
                    r = G[i-1,j+1]    
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = G[i+1,j]
                    r = G[i-1,j]   
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = G[i-1,j-1]
                    r = G[i+1,j+1]   
                if (G[i,j] >= q) and (G[i,j] >= r):
                    res[i,j] = G[i,j]
                else:
                    res[i,j] = 0   
            except IndexError:
                pass
    return res

def doubleThreshold (M, lowRatio = 10, highRatio = 30):
    '''
    Функция двухуровневой фильтрации
    ----------
    Parameters
    ----------
    M : Матрица фильтрованного изображения после немаксимального подавления
    lowRatio : float, optional
        Нижний порог фильтрации (в процентах). The default is 10.
    highRatio : TYPE, optional
        Верхний порог фильтрации (в процентах). The default is 30.

    Returns
    -------
    res : Результирующая матрица

    '''
    highThreshold = M.max() * highRatio / 100.
    lowThreshold = highThreshold * lowRatio / 100.
    height, width = np.shape(M)  
    res = np.zeros((height,width))
    weak = np.int32(25)
    strong = np.int32(255)   
    strong_i, strong_j = np.where(M >= highThreshold)        
    weak_i, weak_j = np.where((M <= highThreshold) & (M >= lowThreshold))    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak    
    for i in range(1, height - 1):
        for j in range (1, width - 1):
            if (res[i,j] == weak):
                try:
                    if (  (res[i+1,j-1] == strong) or (res[i+1,j] == strong) or (res[i+1,j+1] == strong)
                        or(res[i,  j-1] == strong) or (res[i,j+1] == strong)
                        or(res[i-1,j-1] == strong) or (res[i-1,j] == strong) or (res[i-1,j+1] == strong)):
                        res[i,j] = strong
                    else:
                        res[i,j] = 0
                except IndexError:
                    pass
    return res

def DericheMask (kerlen = 15, alpha = 0.75, omega = 0.75):
    '''
    Функция для расчета значений маски фильтра Дерише
    ----------
    Parameters
    ----------
    kerlen : int, optional
        Размер ядра. The default is 15.
    alpha : float, optional
        Параметр alpha для расчета IIR-фильтра. The default is 0.75.
    omega : float, optional
        Параметр omega для расчета IIR-фильтра. The default is 0.75.

    Returns
    -------
    X : float array; length = kerlen
        Маска фильтра Дерише по оси Ох.
    Y : float array; length = kerlen
        Маска фильтра Дерише по оси Оу.
    '''
    c = (1 - 2 * m.exp(-1. * alpha) * m.cos(omega) + m.exp(-2. * alpha))/(m.exp(-1. * alpha) * m.sin(omega))
    k = ((1 - 2 * m.exp(-1. * alpha) * m.cos(omega) + m.exp(-2. * alpha))*(m.pow(alpha, 2) + m.pow(omega, 2))) / (2. * alpha * m.exp(-1. * alpha) *m.sin(omega) + omega - omega * m.exp(-2. * alpha))
    ax = np.linspace(-(kerlen - 1) / 2., (kerlen-1) / 2, kerlen)
    xx, yy = np.meshgrid(ax,ax)
    Xc = -1. * c * np.exp(-1. * alpha * abs(xx)) * (m.sin(omega)) * xx 
    Xt = k * np.exp(alpha *m.sin(omega) * abs(yy) + omega * m.cos(omega) * abs(yy)) * (np.exp(-1. * alpha * abs(yy)))
    Yc = -1. * c * np.exp(-1. * alpha * abs(yy)) * (m.sin(omega)) * yy
    Yt = k * np.exp(alpha *m.sin(omega) * abs(xx) + omega * m.cos(omega) * abs(xx)) * (np.exp(-1. * alpha * abs(xx)))
    X = Xc * Xt / (alpha**2 + omega**2)
    Y = Yc * Yt / (alpha**2 + omega**2)
    return X, Y

def thresholdGlobal(M_image, lvl):
    '''
    Глобальная пороговая обработка
    ----------
    Parameters
    ----------
    M_image : Исходное изображение
    lvl : Уровень фильтрации

    Returns
    -------
    M : Монохромное изображение

    '''
    M = M_image.copy()
    for i in range(len(M)):
        for j in range(len(M[0])):
            if (M[i,j] < lvl):
                M[i,j] = 0
            else:
                M[i,j] = 255
    return M

def thresholdLocMidsum (M_image, ClassLvl, BgLvl, ObjLvl, size = 5):
    '''
    Пороговая фильтрация на основании разбиения на классы
    ----------
    Parameters
    ----------
    M_image : Исходное изображение
    ClassLvl : Уровень разбиения на классы
    BgLvl : Уровень для объектов фона
    ObjLvl : Уровень для искомых объектов
    size : int, optional
        Размер области. The default is 5.

    Returns
    -------
    M_loc_midsum : Монохромное изображение

    '''
    height, width = np.shape(M_image)
    ind_height = height//size
    ind_width = width//size
    h_mod = np.int32((height % size) // 2)
    w_mod = np.int32((width % size) // 2)
    local_m = np.zeros((ind_height, ind_width))       
    for i in range (ind_height):
        for j in range (ind_width):
            mid_sum = 0
            for k in range (size):
                for l in range (size):
                    mid_sum +=  M_image[i*size + k + h_mod, j*size + l + w_mod]
            mid_sum /= (size*size)
            if (mid_sum >= ClassLvl):
                local_m[i,j] = 1
            else:
                local_m[i,j] = 0       
    M_loc_midsum = np.zeros((ind_height * size, ind_width * size))
    for i in range (ind_height):
        for j in range (ind_width):
            for k in range (size):
                for l in range (size):
                    if (local_m[i,j] == 0):
                        if (M_image[i*size+k+h_mod, j*size+l+w_mod] < BgLvl):
                            M_loc_midsum[i*size+k, j*size+l] = 0
                        else:
                            M_loc_midsum[i*size+k, j*size+l] = 255
                    else:
                        if (M_image[i*size+k, j*size+l] < ObjLvl):
                            M_loc_midsum[i*size+k, j*size+l] = 0
                        else:
                            M_loc_midsum[i*size+k, j*size+l] = 255
    return M_loc_midsum

def filtKirsch(M_orig):
    '''
    Фильтр Кирша
    ----------
    Parameters
    ----------
    M_orig : Исходное изображение

    Returns
    -------
    res : Модуль градиента результирующей матрицы
    angle : Угловая составляющая градие

    '''
    height, width = np.shape(M_orig)
    res = np.zeros((height, width))
    angle = np.zeros((height,width), dtype = np.int32)
    for i in range(1,height-1):
        for j in range(1,width-1):
            area = M_orig[(i-1):(i+1), (j-1):(j+1)]
            dN = np.inner(area,kirschN)
            dNW = np.inner(area,kirschNW)
            dW = np.inner(area,kirschW)
            dSW = np.inner(area,kirschSW)
            dS = np.inner(area,kirschS)
            dSE = np.inner(area,kirschSE)
            dE = np.inner(area,kirschE)
            dNE = np.inner(area,kirschNE)
            d = np.array([dW,dNW,dN,dNE,dE,dSE,dS,dSW])
            res[i,j] = np.max(d)
            theta = np.argmax(d) * np.int32(45)
            if (theta >= 180):
                theta -= 180
            angle[i,j] = np.int32(theta)
    return res, angle

def XYrelation(xL, yL, sign = 1, added = 0):
    '''
    Линейная регрессия
    ----------
    Parameters
    ----------
    xL : array; float
        Значения X
    yL : array; float
        Значения Y
    sign : int (bool), optional
        1 equals +, 0 equals -. The default is 1.
    added : float, optional
        added value to result. The default is 0.

    Returns
    -------
    res : Коэффициент угла наклона прямой

    '''
    x = []
    y = []
    flag = 0
    for i in range(len(yL)):
        if (yL[i] != 0 and xL[i]):
            x.append(xL[i])
            y.append(yL[i])
            flag = 1
    if (flag != 0):
        x = np.log2(np.asarray(x)).reshape((-1, 1))
        y = np.log2(np.asarray(y))
        model = LinearRegression().fit(x,y)
        res = model.coef_
    else:
        res = 0
    res = sign * res + added
    return res

def fractLocBoxCounting(imgM):
    '''
    Построение "карты" локальных фрактальных размерностей 
    монохромного изображения методом box-counting
    ----------
    Parameters
    ----------
    imgM : Исходное изображение в оттенках серого

    Returns
    -------
    fractM : "Карта" локальных фрактальных размерностей

    '''
    height, width = np.shape(imgM)
    Lmax = 32
    allWork = height*width
    workStep = np.int32(allWork / 10)
    flag = 0
    currpos = 0
    q = np.int32(Lmax // 2) #половина от максимальной длины кластера
    fractM = np.zeros((height, width))
    for ii in range(q, height - q, 2): #идем по изображению. Отступаем на q, не доходим до конца на q, шаг - 1
        currpos += width
        if (currpos >= workStep):
            currpos -= workStep
            flag += 1
            print (flag*10)
        for jj in range(q, width - q, 1):
            imgCut = imgM[ii-q : ii + q, jj-q : jj+q]
            if (np.count_nonzero(imgCut) < 10):
                continue
            fractCut = fractLoc2DBox(imgCut)
            fractM[ii-1:ii+1,jj-1:jj+1] = fractCut
    print(np.max(fractM))
    return fractM

def fractLoc2DBox(M):
    '''
    Подсчет фрактальной размерности участка изображения
    монохромного изображения методом box-counting
    ----------
    Parameters
    ----------
    M : Участок изображения

    Returns
    -------
    res : Значение фрактальной размерности

    '''
    height, width = np.shape(M)
    if (height == 32):
        xL = np.array([2,4,8])
    elif (height == 64):
        xL = np.array([2,4,8,16])     
    q = len(xL)
    
    yL = np.zeros(q)
    
    for i in range(q):
        for ii in range(0, height, xL[i]):
            for jj in range(0, width, xL[i]):
                for iL in range(xL[i]):
                    for jL in range(xL[i]):
                        flag = False
                        try:
                            if (M[ii+iL, jj+jL] != 0):
                                yL[i] += 1
                                flag = True
                        except IndexError:
                            pass
                        if (flag == True):
                            break
                    if (flag == True):
                        break                        
    res = XYrelation(xL, yL, 2)
    return res

def fractLocGray(grayImg, frtype = 0):
    '''
    Подсчет "карты" локальных фрактальных размерностей изображения
    в оттенках серого методами box-counting и triangle
    ----------
    Parameters
    ----------
    grayImg : Исходное изображение в оттенках серого
    frtype : int, optional
        0 - box-counting,
        1 - triangle method.
        The default is 0.
    Returns
    -------
    genM : "Карта" локальных фрактальных размерностей

    '''
    height, width = np.shape(grayImg)
    maxL = 5
    d2x = height % (2**(maxL))
    d2y = width % (2**(maxL))
    sted = 2**(maxL-1)#стартовый/конечный отступ от края картинки; половина от макс размера "кластера"
    dxl = np.int32(d2x // 2)
    dyl = np.int32(d2y // 2)
    if(d2x & 1 == 1):
        dxr = dxl + 1
    else:
        dxr = dxl
    if(d2y & 1 == 1):
        dyr = dyl + 1
    else:
        dyr = dyl
    nheight = height - d2x
    nwidth = width - d2y
    imgCut = grayImg[dxl:(height-dxr+1), dyl:(width-dyr+1)]
    genM = np.zeros((nheight, nwidth))
    allWork = height*width
    workStep = np.int32(allWork / 10)
    flag = 0
    currpos = 0
    for ii in range(sted, nheight - sted, 2):
        currpos += width
        if (currpos >= workStep):
            currpos -= workStep
            flag += 1
            print (flag*10)
        for jj in range(sted, nwidth - sted, 2):
            locCut = imgCut[(ii-sted):(ii+sted), (jj-sted):(jj+sted)]
            if(frtype == 0):
                fDim = fractGrayBoxCount(locCut)
                genM[ii-1:ii+1,jj-1:jj+1] = fDim
            elif(frtype == 1):
                locCut = imgCut[(ii-sted):(ii+sted), (jj-sted):(jj+sted)]
                fDim = fractGrayTriangleLocal(locCut)#???
                genM[ii-1:ii+1,jj-1:jj+1] = fDim
    print(np.max(genM))
    return genM

def fractGrayBoxCount (grayImg):
    '''
    Подсчет фрактальной размерности изображения
    в оттенках серого методом box-counting
    ----------
    Parameters
    ----------
    grayImg : Изображение в оттенках серого

    Returns
    -------
    res : Значение фрактальной размерности

    '''
    height, width = np.shape(grayImg)
    Lmax = np.int32(np.log2(height))
    L=[]
    for i in range(1, Lmax-1):
        L.append(2**i)
    L = np.asarray(L)
    G = np.max(grayImg)
    hL = L*G / height
    cL = len(L) 
    nL = np.zeros(cL)
    for i in range(cL):
        for ii in range(0, height, L[i]):
            for jj in range(0, width, L[i]):
                l = 0
                k = 256
                for kk in range (L[i]):
                    for ll in range (L[i]):
                        if   (grayImg[ii + kk, jj + ll] >= l):
                            l = grayImg[ii + kk, jj + ll]
                        if (grayImg[ii + kk, jj + ll] <= k):
                            k = grayImg[ii + kk, jj + ll]
                nL[i] += (l - k + 1)
                #print(l,k,nL)
        #print(nL[i])
        if(hL[i] != 0):
            nL[i] /= hL[i]
    #print(L,nL)
    res = XYrelation(L, nL, 2)
    return res

def fractGrayTriangleLocal(grayImg):
    '''
    Подсчет "карты" локальных фрактальных размерностей изображения
    в оттенках серого методом triangle
    ----------
    Parameters
    ----------
    grayImg : Изображение в оттенках серого

    Returns
    -------
    fractM : Значение фрактальной размерности

    '''
    height, width = np.shape(grayImg)
    Lmax = 8 #np.int32(height // 2)
    fractM = np.zeros((height-Lmax+1, width-Lmax+1))
    stepWork = height/20 #equal to 100%/20=5%
    flag = 0
    curr = 0
    for ii in range (height - Lmax):
        curr += 1
        if (curr >= stepWork):
            curr -= stepWork
            flag += 5
            print(flag)
        for jj in range (width - Lmax):
            imgCut = grayImg[ii:(ii + Lmax), jj:(jj + Lmax)]
            res = fractGrayTriangleGlobal(imgCut)
            fractM[ii, jj] = res   
    return fractM

def fractGrayTriangleGlobal (grayImg):
    '''
    Подсчет фрактальной размерности изображения
    в оттенках серого методом triangle
    ----------
    Parameters
    ----------
    grayImg : Изображение в оттенках серого

    Returns
    -------
    res : Значение фрактальной размерности

    '''
    height, width = np.shape(grayImg)
    Lmax = 0
    if (height == 8):
        Lmax = 8 #np.int32(np.log2(height)-1)
        L = np.array([2, 4, 8])
        xL = np.array([4, 16, 64])
    elif (height == 12):
        Lmax = 12
        L = np.array([2, 3, 4, 6, 12])
        xL = np.array([4, 9, 16, 36, 144])
    if(Lmax == 0):
        print("Error. Lmax = 0")
        SystemExit(-1) 
    cL = len(L)
    yL = np.zeros(cL)
    for i in range(cL):
        for ii in range(0, height - L[i] + 1, L[i]):
            for jj in range(0, width - L[i] + 1, L[i]):
                hA = grayImg[ii, jj]
                hB = grayImg[ii + L[i] - 1, jj]
                hC = grayImg[ii + L[i] - 1, jj + L[i] - 1]
                hD = grayImg[ii, jj + L[i] - 1]
                hO = (hA + hB + hC + hD)/4               
                if (hO == 0):
                    continue                   
                sqL = L[i] / np.sqrt(2) #половина гипотенузы                
                AB = np.hypot(np.abs(hA - hB), L[i])
                BC = np.hypot(np.abs(hB - hC), L[i])
                CD = np.hypot(np.abs(hC - hD), L[i])
                AD = np.hypot(np.abs(hA - hD), L[i])
                
                AO = np.hypot(np.abs(hA - hO), sqL)
                BO = np.hypot(np.abs(hB - hO), sqL)
                CO = np.hypot(np.abs(hC - hO), sqL)
                DO = np.hypot(np.abs(hD - hO), sqL)

                sABO = GeronS(AB, AO, BO)
                sADO = GeronS(AD, AO, DO)
                sBCO = GeronS(BC, BO, CO)
                sDCO = GeronS(CD, CO, DO)

                yL[i] += (sABO + sADO + sBCO + sDCO)
    res = XYrelation(xL, yL)
    return res

def fractGrayTriangleLocalV2(grayImg):
    '''
    Подсчет "карты" локальных фрактальных размерностей изображения
    в оттенках серого методом triangle v2
    ----------
    Parameters
    ----------
    grayImg : Изображение в оттенках серого

    Returns
    -------
    fractM : Значение фрактальной размерности

    '''
    height, width = np.shape(grayImg)
    Lmax = 9
    sL = Lmax // 2
    fractM = np.zeros((height, width))
    stepWork = height/20 #equal to 100%/20=5%
    flag = 0
    curr = 0
    for ii in range (sL, height - sL - 1):
        curr += 1
        if (curr >= stepWork):
            curr -= stepWork
            flag += 5
            print(flag)
        for jj in range (sL, width - sL - 1):
            imgCut = grayImg[(ii - sL):(ii + sL + 1), (jj - sL):(jj + sL + 1)]
            res = fractGrayTriangleV2(imgCut)
            fractM[ii, jj] = res   
    return fractM

def fractGrayTriangleLocalV3(grayImg):
    '''
    Подсчет "карты" локальных фрактальных размерностей изображения
    в оттенках серого методом triangle v3
    ----------
    Parameters
    ----------
    grayImg : Изображение в оттенках серого
    
    Returns
    -------
    fractM : Значение фрактальной размерности

    '''
    height, width = np.shape(grayImg)
    Lmax = 12
    sL = Lmax // 2
    fractM = np.zeros((height, width))
    stepWork = height/20 #equal to 100%/20=5%
    flag = 0
    curr = 0
    for ii in range (sL, height - sL):
        curr += 1
        if (curr >= stepWork):
            curr -= stepWork
            flag += 5
            print(flag)
        for jj in range (sL, width - sL):
            imgCut = grayImg[(ii - sL):(ii + sL), (jj - sL):(jj + sL)]
            res = fractGrayTriangleGlobal(imgCut)
            fractM[ii, jj] = res   
    return fractM

def fractGrayTriangleV2 (grayImgCut):
    '''
    Подсчет фрактальной размерности изображения
    в оттенках серого методом triangle
    ----------
    Parameters
    ----------
    grayImgCut : Изображение в оттенках серого

    Returns
    -------
    res : Значение фрактальной размерности

    '''
    EPS = 1e-12
    height, width = np.shape(grayImgCut)
    #L = [3, 5, 7, 9]; cL = np.zeros(L)
    cL = 4
    xL = np.array([9, 25, 49, 81]) #xL == L^2
    yL = np.zeros(cL)
    #L = 3
    for ii in range(0, height, 3):
        for jj in range(0, width, 3):
            hA = grayImgCut[ii, jj]
            hB = grayImgCut[ii + 2, jj]
            hC = grayImgCut[ii + 2, jj + 2]
            hD = grayImgCut[ii, jj + 2]
            
            hO = max((hA + hB + hC + hD)/4, np.float64(grayImgCut[ii + 1, jj + 1]))
            if (hO < EPS):
                continue

            sqL = 3 / np.sqrt(2) #половина гипотенузы

            AB = np.hypot(np.abs(hA - hB), 3)
            BC = np.hypot(np.abs(hB - hC), 3)
            CD = np.hypot(np.abs(hC - hD), 3)
            AD = np.hypot(np.abs(hA - hD), 3)

            AO = np.hypot(np.abs(hA - hO), sqL)
            BO = np.hypot(np.abs(hB - hO), sqL)
            CO = np.hypot(np.abs(hC - hO), sqL)
            DO = np.hypot(np.abs(hD - hO), sqL)

            sABO = GeronS(AB, AO, BO)
            sADO = GeronS(AD, AO, DO)
            sBCO = GeronS(BC, BO, CO)
            sDCO = GeronS(CD, CO, DO)
            
            yL[0] = (sABO + sADO + sBCO + sDCO)
    #L = 5
    for ii in (0, 4):
        for jj in (0, 4):
            hA = grayImgCut[ii, jj]
            hB = grayImgCut[ii + 4, jj]
            hC = grayImgCut[ii + 4, jj + 4]
            hD = grayImgCut[ii, jj + 4]
            
            hO = max((hA + hB + hC + hD)/4, np.float64(grayImgCut[ii + 2, jj + 2]))
            if (hO < EPS):
                continue

            sqL = 5 / np.sqrt(2) #половина гипотенузы

            AB = np.hypot(np.abs(hA - hB), 5)
            BC = np.hypot(np.abs(hB - hC), 5)
            CD = np.hypot(np.abs(hC - hD), 5)
            AD = np.hypot(np.abs(hA - hD), 5)

            AO = np.hypot(np.abs(hA - hO), sqL)
            BO = np.hypot(np.abs(hB - hO), sqL)
            CO = np.hypot(np.abs(hC - hO), sqL)
            DO = np.hypot(np.abs(hD - hO), sqL)

            sABO = GeronS(AB, AO, BO)
            sADO = GeronS(AD, AO, DO)
            sBCO = GeronS(BC, BO, CO)
            sDCO = GeronS(CD, CO, DO)
            
            yL[1] += (sABO + sADO + sBCO + sDCO)
    #L = 7
    for ii in (0, 2):
        for jj in (0, 2):
            hA = grayImgCut[ii, jj]
            hB = grayImgCut[ii + 6, jj]
            hC = grayImgCut[ii + 6, jj + 6]
            hD = grayImgCut[ii, jj + 6]
            
            hO = max((hA + hB + hC + hD)/4, np.float64(grayImgCut[ii + 4, jj + 4]))
            if (hO < EPS):
                continue
                
            AB = np.hypot(np.abs(hA - hB), 7)
            BC = np.hypot(np.abs(hB - hC), 7)
            CD = np.hypot(np.abs(hC - hD), 7)
            AD = np.hypot(np.abs(hA - hD), 7)
            
            sqL = 7 / np.sqrt(2) #половина гипотенузы
            AO = np.hypot(np.abs(hA - hO), sqL)
            BO = np.hypot(np.abs(hB - hO), sqL)
            CO = np.hypot(np.abs(hC - hO), sqL)
            DO = np.hypot(np.abs(hD - hO), sqL)
            
            sABO = GeronS(AB, AO, BO)
            sADO = GeronS(AD, AO, DO)
            sBCO = GeronS(BC, BO, CO)
            sDCO = GeronS(CD, CO, DO)
            
            if (ii == 0 and jj == 0):
                yL[2] += (sABO + sADO)
            elif(ii == 0 and jj != 0):
                yL[2] += (sABO + sBCO)
            elif(ii != 0 and jj != 0):
                yL[2] += (sBCO + sDCO)
            elif(ii != 0 and jj == 0):
                yL[2] += (sDCO + sADO)
    #CENTER
    hA = grayImgCut[1, 1]
    hB = grayImgCut[7, 1]
    hC = grayImgCut[7, 7]
    hD = grayImgCut[1, 7]

    hO = (hA + hB + hC + hD)/4

    AB = np.hypot(np.abs(hA - hB), 7)
    BC = np.hypot(np.abs(hB - hC), 7)
    CD = np.hypot(np.abs(hC - hD), 7)
    AD = np.hypot(np.abs(hA - hD), 7)

    sqL = 7 / np.sqrt(2) #половина гипотенузы
    AO = np.hypot(np.abs(hA - hO), sqL)
    BO = np.hypot(np.abs(hB - hO), sqL)
    CO = np.hypot(np.abs(hC - hO), sqL)
    DO = np.hypot(np.abs(hD - hO), sqL)
    
    sABO = GeronS(AB, AO, BO)
    sADO = GeronS(AD, AO, DO)
    sBCO = GeronS(BC, BO, CO)
    sDCO = GeronS(CD, CO, DO)
    yL[2] += (sABO + sADO + sBCO + sDCO)

    #L = 9
    hA = grayImgCut[0, 0]
    hB = grayImgCut[8, 0]
    hC = grayImgCut[8, 8]
    hD = grayImgCut[0, 8]

    hO = max((hA + hB + hC + hD)/4, np.float64(grayImgCut[4, 4]))

    sqL = 9 / np.sqrt(2) #половина гипотенузы

    AB = np.hypot(np.abs(hA - hB), 9)
    BC = np.hypot(np.abs(hB - hC), 9)
    CD = np.hypot(np.abs(hC - hD), 9)
    AD = np.hypot(np.abs(hA - hD), 9)

    AO = np.hypot(np.abs(hA - hO), sqL)
    BO = np.hypot(np.abs(hB - hO), sqL)
    CO = np.hypot(np.abs(hC - hO), sqL)
    DO = np.hypot(np.abs(hD - hO), sqL)

    sABO = GeronS(AB, AO, BO)
    sADO = GeronS(AD, AO, DO)
    sBCO = GeronS(BC, BO, CO)
    sDCO = GeronS(CD, CO, DO)
    
    yL[3] += (sABO + sADO + sBCO + sDCO)
    
    res = XYrelation(xL, yL)
    #print(xL, yL, res)
    return res

def estimMaxL (height, width, estimMin = 100):
    '''
    Поиск максимального значения ячейки при минимизации обрезаемой области
    ----------
    Parameters
    ----------
    height : int
    width : int
    estimMin : int, optional
        Максимальное обрезание области. The default is 100.

    Returns
    -------
    maxL : int
        Максимальный размер ячейки

    '''
    q = np.min((height, width))
    maxL = 0
    for i in range(7):
        p = 2**i
        if (p < q):
            dx = height % p
            dy = width % p
            if (dx + dy <= estimMin):
                maxL = i
        else:
            break
    return maxL

#linear regression, пометка правильного вызова функции
def TANGENS(x, y):
    x = x.reshape((-1, 1))
    #y = np.array([...])
    model = LinearRegression().fit(x,y)
    return model.coef_

def slidingAverage(imgM, n = 20, T = 120):
    '''
    Фильтрация основанная на подсчете скользящего среднего значения яркости
    ----------
    Parameters
    ----------
    imgM : Матрица изображения
    n : int, optional
        Число учитываемых пикселей в подсчете. The default is 20.
    T : int, optional
        Порог. The default is 120.

    Returns
    -------
    res : Обработанное изображение

    '''
    height, width = np.shape(imgM)
    m = np.zeros((height, width), dtype = np.float64)
    m[0,0] = imgM[0,0]/n
    ii, jj = 0, 0
    for i in range (1, height * width):
        ip, jp = ii, jj
        ii, jj = slIndex(i, height, width)
        ie, je = slIndex(i-n, height, width)
        m[ii,jj] = m[ip,jp] + imgM[ii,jj]/n
        if((ie >= 0) and (je >= 0)):
            m[ii,jj] -= (imgM[ie,je]/n)
    res = np.zeros((height,width), dtype = np.int8)
    for i in range(height):
        for j in range(width):
            if (m[i,j] < T):
                res[i,j] = 255
    return res

def slIndex(i,height,width):
    '''
    Функция определения индекса при проходе по змейке сверху вниз слева направо
    ----------
    Parameters
    ----------
    i : int
        Текущий "выпрямленный" индекс
    height : int
    width : int

    Returns
    -------
    ii : int height
    jj : int width

    '''
    ii = i//width
    if (ii & 1 == 0): #from left to right
        jj = i % width
    else:             #from right to left
        jj = width - i % width - 1
    return ii, jj

def GeronS(a, b, c):
    '''
    Формула Герона подсчета площади
    ----------
    Parameters
    ----------
    a : float
    b : float
    c : float

    Returns
    -------
    float
        Площадь треугольника

    '''
    p = (a+b+c)/2
    return np.sqrt(p*(p-a)*(p-b)*(p-c))   

def fractMassRadiusRelation(img):#local 'connected'
    '''
    Подсчет "карты" локальных фрактальных размерностей
    монохромного изображения методом Mass-Radius relation
    ----------
    Parameters
    ----------
    img : Матрица изображения
    
    Returns
    -------
    fractM : "Карта" локальных фрактальных размерностей
        
    '''
    imgM = np.copy(img)
    height, width = np.shape(imgM)
    flagsM = [np.int32(-1)] * (height * width)#инициализация матрицы флагов
    flagsM = np.asarray(flagsM).reshape((height, width))
    xi, yj = np.where(imgM != 0)
    flagsM[xi,yj] = 0
    fractM = np.zeros((height, width))
    Lmax = 5
    xL = [0, 1, 2, 3, 4, 5]
    fractM = np.zeros((height, width))
    for ii in range(Lmax+1, height-Lmax-1):
        for jj in range(Lmax+1, width - Lmax-1):
            if (flagsM[ii,jj] == 0):
                imgCut = imgM[(ii-Lmax-1):(ii+Lmax),(jj-Lmax-1):(jj+Lmax)]
                flagsCut = flagsM[(ii-Lmax-1):(ii+Lmax),(jj-Lmax-1):(jj+Lmax)]
                yL = count_connected(imgCut, flagsCut)
                fractM[ii,jj] = XYrelation(xL, yL, 1)
    return fractM

def count_connected(imgM, flagsM, fAll = False):
    '''
    Функция подсчета числа соединенных пикселей
    ----------
    Parameters
    ----------
    imgM : Матрица изображения
    flagsM : Матрица помеченной области
    fAll : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    count : int
        Кол-во соединенных пикселей

    '''
    q = queue.Queue()
    qAll = []
    height, width = np.shape(imgM)
    midInd = np.int32(height // 2)
    count = np.zeros(midInd + 1)
    count[0] = 0
    flagsM[midInd,midInd] = 1
    q, flagsM = findNear(imgM, flagsM, midInd, midInd)
    while not q.empty():
        ii, jj = q.get()
        flagsM[ii, jj] = 1
        x = [ii, jj]
        if x not in qAll:
            qAll.append([ii, jj])
        qn = queue.Queue()
        if (ii < height and jj < width):
            qn, flagsM = findNear(imgM, flagsM, ii, jj)
        while not qn.empty():
            q.put(qn.get())
    qAll = np.asarray(qAll)
    for i in range (len(qAll)):
        dist = np.max((np.abs(qAll[i,0] - midInd), np.abs(qAll[i,1] - midInd)))
        count[dist] += 1 
    if(fAll):
        for i in range(1, len(count)):
            count[i] += count [i-1]
    return count      

def findNear(imgM, flagsM, iInd, jInd):
    '''
    Алгоритм поиска ближайших непомеченных элементов
    ----------
    Parameters
    ----------
    imgM : Матрица изображения
    flagsM : Матрица флагов
    iInd : i-индекс "стартового" элемента 
    jInd : j-индекс "стартового" элемента

    Returns
    -------
    q : queue
        Очередь из соседствующих элементов
    flagsM : Обновленная матрица флагов

    '''
    q = queue.Queue()
    height, width = np.shape(imgM)
    for ii in range(3):
        for jj in range(3):
            if ((0 < iInd + ii - 1 < height) and (0 < jInd + jj - 1 < width)):
                if (flagsM[iInd + ii - 1, jInd + jj - 1] == 0 and imgM[iInd + ii - 1,jInd + jj - 1] != 0):
                    q.put([iInd + ii - 1, jInd + jj - 1])
    #nex = input("waitkey")
    return q, flagsM

def fractMassRadiusRelationV2Loc(imgGray):
    '''
    Подсчет "карты" локальных фрактальных размерностей изображения
    в оттенках серого методом Mass-Radius relation v2    
    ----------
    Parameters
    ----------
    imgGray : Изображение в оттенках серого

    Returns
    -------
    fractM : "Карта" локальных фрактальных размерностей
 
    '''
    height, width = np.shape(imgGray)
    Lmax = 4
    fractM = np.zeros((height - 2*Lmax, width - 2*Lmax))
    stepWork = (height - 2*Lmax)/20 #equal to 100%/20=5%
    flag = 0
    curr = 0
    for ii in range(Lmax, height - Lmax - 1):
        curr += 1
        if (curr >= stepWork):
            curr -= stepWork
            flag += 5
            print(flag)
        for jj in range(Lmax, width - Lmax - 1):
            imgGrayCut = imgGray[ii-Lmax:ii+Lmax, jj-Lmax:jj+Lmax]
            fractM[ii-Lmax, jj-Lmax] = massRadiusRelationV2Cut(imgGrayCut)
    return fractM

def massRadiusRelationV2Cut(imgGrayCut):
    '''
    Подсчет фрактальной размерности изображения
    в оттенках серого методом Mass-Radius relation v2    
    ----------
    Parameters
    ----------
    imgGrayCut : Участок изображения в оттенках-серого

    Returns
    -------
    res : Значение фрактальной размерности участка изображения

    '''
    height, width = np.shape(imgGrayCut)
    Xc, Yc = massCenter(imgGrayCut)
    Rmax = np.max((Xc, height - Xc - 1, Yc, width - Yc - 1))
    xL = np.zeros(Rmax + 1)
    yL = np.zeros(Rmax + 1)
    for i in range(Rmax):
        xL[i] = i + 1
    for ii in range(height):
        for jj in range(width):
            R = np.max(np.abs((ii - Xc, jj - Xc)))
            if (R == 0):
                continue
            yL[R] += imgGrayCut[ii, jj]
    res = XYrelation(xL, yL)
    return res
    
    
def massCenter(M):
    '''
    Поиск центра масс яркостей участка изображения
    ----------
    Parameters
    ----------
    M : Матрица

    Returns
    -------
    res : Массив Xc, Yc - координат центра масс

    '''
    h, w = np.shape(M)
    sAll = 0; xSum = 0; ySum = 0
    for ii in range(h):
        for jj in range(w):
            sAll += M[ii, jj]
            xSum += ii*M[ii, jj]
            ySum += jj*M[ii, jj]
    Xc = xSum/sAll
    Yc = ySum/sAll
    res = np.int32(np.array((Xc, Yc)))
    return res

#Матричное описание фильтров
amp_x = np.array([[1, -1], [1, -1]])
amp_y = np.array([[1,  1],[-1, -1]])
#Максимальный амплитудный фильтр нельзя записать через ядро
roberts_x = np.array([[1,  0], [0, -1]])
roberts_y = np.array([[0, -1], [1,  0]])

previtt_y  = np.array([[-1,-1,-1],  [0, 0, 0], [1, 1, 1]])
previtt_x  = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
previtt_d1 = np.array([[ 0, 1, 1], [-1, 0, 1], [-1,-1, 0]])
previtt_d2 = np.array([[-1,-1, 0], [-1, 0, 1], [ 0, 1, 1]])

sobel_y = np.array([[-1,-2,-1], [ 0, 0, 0], [1, 2, 1]])
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]])
sobel_d1 = np.array([[0, 1, 2], [-1, 0, 1], [-2,-1, 0]])
sobel_d2 = np.array([[-2,-1, 0], [-1, 0, 1], [ 0, 1, 2]])

laplassian_mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

kirschN  = np.array([ 5, 5, 5, -3, 0,-3, -3,-3,-3])
kirschNW = np.array([-3, 5, 5, -3, 0, 5, -3,-3,-3])
kirschW  = np.array([-3,-3, 5, -3, 0, 5, -3,-3, 5])
kirschSW = np.array([-3,-3,-3, -3, 0, 5, -3, 5, 5])
kirschS  = np.array([-3,-3,-3, -3, 0,-3,  5, 5, 5])
kirschSE = np.array([-3,-3,-3,  5, 0,-3,  5, 5,-3])
kirschE  = np.array([ 5,-3,-3,  5, 0,-3,  5,-3,-3])
kirschNE = np.array([ 5, 5,-3,  5, 0,-3, -3,-3,-3])