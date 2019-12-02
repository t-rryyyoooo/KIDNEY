import numpy as np
import os
def makeCDF(imgArrayList, alpha = 0):#imgArrayList:int64
    imgArrayList = [ x + 1024 for x in imgArrayList ]
    
    imgArrayList = [np.where(x<0, 0, x) for x in imgArrayList]
    imgArrayList = [np.where(x>2048, 2048,x) for x in imgArrayList]
    
    ctRange = 2048 + 1
    #uniformDistribution = np.array([ctRange]*ctRange)
    
    HIST = np.array([0.0]*ctRange)
    for x in imgArrayList:
        hist, _  = np.histogram(x.flatten(), ctRange, [0, 2048+1])
        HIST += hist
    
    
    print(HIST)
    HIST /= sum(HIST)
    HIST = HIST * (1 - alpha) + alpha / 2048
    print(HIST)
    cdf = HIST.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    temp = (cdf_m - cdf_m.min())/(cdf_m.max()-cdf_m.min())
    cdf_m = 2048*temp
    cdf = np.ma.filled(cdf_m,0).astype('int64')
    
    return cdf,imgArrayList

def equalizingHistogram(imgArrayList, alpha):
    cdf, imgArrayList = makeCDF(imgArrayList, alpha)
    
    equalizedImageArrayList = []
    for imgAr in imgArrayList:
        x = cdf[imgAr] - 1024
        equalizedImageArrayList.append(x)
    
    return equalizedImageArrayList

def equalizingHistogramSummed(imageArrayList, npy, alpha):
    #Make CDF
    print('Loading npy file...')
    npyFile = os.path.expanduser(npy)
    HIST = np.load(npyFile)
    print('Loading it has done. ')
    aHIST = HIST * alpha + (1 - alpha) / 2048

    cdf = aHIST.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    temp = (cdf_m - cdf_m.min())/(cdf_m.max()-cdf_m.min())
    cdf_m = 2048*temp
    cdf = np.ma.filled(cdf_m,0).astype('int64')
    print('Making CDF has done.' )

    #Equalize histogram
    equImgArrayList = []
    for imgArray in imageArrayList:
        imgArray = imgArray + 1024
        imgArray = np.clip(imgArray, 0, 2048)
        imgArray = cdf[imgArray] - 1024

        equImgArrayList.append(imgArray)
    
    return equImgArrayList

def equalizingHistogramSummedFloat(imageArrayList, npy, alpha, times=10**3, types="float32"):
    #Make CDF
    print('Loading npy file...')
    npyFile = os.path.expanduser(npy)
    HIST = np.load(npyFile)
    print('Loading it has done. ')
    aHIST = HIST * alpha + (1 - alpha) / (2048 * times)

    cdf = aHIST.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    temp = (cdf_m - cdf_m.min())/(cdf_m.max()-cdf_m.min())
    cdf_m = 2048 * times * temp
    cdf = np.ma.filled(cdf_m,0).astype(types)
    print('Making CDF has done.' )

    #Equalize histogram
    equImgArrayList = []
    for imgArray in imageArrayList:
        #Preprocessing
        imgArray = imgArray + 1024
        imgArray = np.clip(imgArray, 0, 2048)
        imgArray = imgArray * times
        imgArray = np.array(imgArray, dtype=np.int)
        
        #Equalize hisotgram
        imgArray = cdf[imgArray]
        imgArray = imgArray / times - 1024

        equImgArrayList.append(imgArray)
    
    return equImgArrayList