import gc
import math
import numpy as np
from cut import caluculate_area
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import pi
import SimpleITK as sitk


def searchBound(labelArray, axis):
    '''
    Argument
    axis: "sagittal" or "coronal" or "axial"
    
    Return the kidney borders in axis derection.
    
    startIndex: [Index of the beginning of the first kidney area, second, ...]
    endIndex: [Index of the ending of the first kidney area, second, ...]
    
    '''
    encounter = False
    startIdx = []
    endIdx = []
    if axis == 'sagittal':
        length, _, _ = labelArray.shape
        
        for l in range(length):
            sliceArray = labelArray[l, :, :]
            area = caluculate_area(sliceArray)
            if area != 0 and not encounter:
                encounter = True
                startIdx.append(l)

            if area == 0 and encounter:
                encounter = False
                endIdx.append(l)

        return startIdx, endIdx

    if axis == 'coronal':
        _, length, _ = labelArray.shape
        
        for l in range(length):
            sliceArray = labelArray[:, l, :]
            area = caluculate_area(sliceArray)
            if area != 0 and not encounter:
                encounter = True
                startIdx.append(l)

            if area == 0 and encounter:
                encounter = False
                endIdx.append(l)

        return startIdx, endIdx        

    if axis == 'axial':
        _, _, length = labelArray.shape
        for l in range(length):
            sliceArray = labelArray[:, :, l]
            area = caluculate_area(sliceArray)
            if area != 0 and not encounter:
                encounter = True
                startIdx.append(l)

            if area == 0 and encounter:
                encounter = False
                endIdx.append(l)

        return startIdx, endIdx

## Not used
def extractBoundingBox(labelArray):
    print("From {} to ".format(labelArray.shape), end="")
    
    startIdx, endIdx = searchBound(labelArray, 'axial')
    labelArray = labelArray[:, :, startIdx[0] -1 : endIdx[0] + 1]
    
    print(labelArray.shape)
    
    print("From {} to ".format(labelArray.shape), end="")
    startIdx, endIdx = searchBound(labelArray, 'coronal')
    labelArray = labelArray[:, startIdx[0] -1 : endIdx[0] + 1, :]
    
    print(labelArray.shape)
    
    return labelArray

def pythagorian(x, y):
    '''
    Argument
    x, y = (x cordinate, y cordinate, zcordinate)
    
    Return the distance between x and y.
    
    '''
    
    l = 0
    d = len(x)
    for i in range(d):
        l += (x[i] - y[i])**2
    
    return math.sqrt(l)


def getSortedDistance(obbPoints, index):
    '''
    Argument
    obbPoints: It is obb.points. [array([1,2,3]), array([2,3,4]),...]
    index: ((0, 1), (0, 3), (0, 5))
    
    Return the sorted some of obb.points by distance between index[i][0] and index[i][1].
    out: [np.array([1,2,3]), np.array([4,5,6]), ...]
    
    '''
    length = []
    out = []
    for i in index:
        l = pythagorian(obbPoints[i[0]], obbPoints[i[1]])
        length.append((l, (obbPoints[i[0]], obbPoints[i[1]])))
    
    length = sorted(length, reverse=True)
    for _ , i in length:
        out.append(i)
    
    return np.array(out)

def getRadianFromCoords(x, y, axis):
    '''
    Argument
    x, y = (x coords, y coords, z coords)
    axis: x->0, y->1, z->2
    
    
    when axis=0, return the gradient of the straight line xy in the yz plane.
    when axis=1, return the gradient of the straight line xy in the xz plane.
    when axis=2, return the gradient of the straight line xy in the xy plane.
    
    '''
    g = []
    d = len(x)
    esp = 10**(-9)
    for i in range(d):
        g.append(y[i] - x[i])
        
    if axis == 0:
        return atan2(g[2], g[1])
        #return g[1] / g[2]
    
    if axis == 1:
        return atan2(g[2], g[0])
        #return g[0] / g[2]
    
    if axis == 2:
        return atan2(g[1], g[0])
        #return g[1] / g[2]
        
        
def makeRotationMatrix(radian, axis):
    '''
    Argument
    axis: x->0, y->1, z->2
    
    Return rotation matrix with axis as the axis of the rotation.
    
    '''
    if axis == 0:
        matrix = np.array([[1, 0, 0], 
                           [0, cos(radian), (-1) * sin(radian)],
                           [0, sin(radian), cos(radian)]])
        
        return matrix
    
    if axis == 1:
        matrix = np.array([[cos(radian), 0, (-1) * sin(radian)],
                           [0, 1, 0], 
                           [sin(radian), 0, cos(radian)]])
        
        return matrix
    
    if axis == 2:
        matrix = np.array([[cos(radian), (-1) * sin(radian), 0], 
                           [sin(radian), cos(radian), 0],
                           [0, 0, 1]])
        
        return matrix


def determineAxis(radian):
    '''
    Return the axis to be parallel.
    
    '''
    if pi/4 < radian < 3*pi/4 or -3*pi/4 < radian < -pi/4:
        return "vertical"
    
    if -pi/4 <= radian <= pi/4 or 3*pi/4 <= radian <= 2*pi or -2*pi <= radian <= -3*pi/4: 
        return "horizontal"
    
def determineRadian(radian):
    '''
    Return an angle(radian) such that bounding box is parallel to the appropriate axis.
    
    '''
    axis = determineAxis(radian)
    if axis=="horizontal":
        if -pi/4 <= radian <= pi/4:
            return -radian, axis
        
        else:
            if radian > 0:
                return pi - radian, axis
            
            else:
                return -pi - radian, axis
            
    if axis=="vertical":
        if radian > 0:
            return pi/2 - radian, axis
        
        else:
            return -pi/2 - radian, axis
    
def makeCompleteMatrix(points):
    '''
    Argument
    points: return value of getSortedDistance
    
    Return rotation matrix that rotates bounding box so that it is parallel to the xyz axis.
    '''
    axisList = ("yz", "xz", "xy")
    from math import degrees
    
    p = np.array([points[0][0], points[0][1], points[1][1], points[2][1]])
    
    axis = 0
    orgRadianX = getRadianFromCoords(p[0], p[1], axis)
    radianX, directionX = determineRadian(orgRadianX)
    
    print("The gradient in the {} plane is {}.".format(axisList[axis], degrees(orgRadianX)))
    print("So, rotate by {} so that it is parallel to {} direction.\n".format(degrees(radianX), directionX))
    
    matrixX = makeRotationMatrix(radianX, 0)
    p = np.einsum("ij, kj->ki", matrixX, p)
    
    #print(p.astype(int))
    
    if directionX == "horizontal":
        axis = 2    
    else:
        axis = 1
    
    orgRadianY = getRadianFromCoords(p[0], p[1], axis)
    
    
    radianY, directionY = determineRadian(orgRadianY)
    
   
    
    print("The gradient in the {} plane is {}.".format(axisList[axis], degrees(orgRadianY)))
    print("So, rotate by {} so that it is parallel to {} direction.\n".format(degrees(radianY), directionY))

    matrixY = makeRotationMatrix(radianY, axis)
    p = np.einsum("ij, kj->ki", matrixY, p)
    
    #sprint(p.astype(int))
    
    if directionX == "horizontal":
        if directionY == "horizontal":
            axis = 0
        else:
            axis = 1
            
    
    else:
        if directionY == "horizontal":
            axis = 0
        else:
            axis = 2
        
    orgRadianZ = getRadianFromCoords(p[0], p[2], axis)


    radianZ, directionZ = determineRadian(orgRadianZ)

    print("The gradient in the {} plane is {}.".format(axisList[axis], degrees(orgRadianZ)))
    print("So, rotate by {} so that it is parallel to {} direction.\n".format(degrees(radianZ), directionZ))

    matrixZ = makeRotationMatrix(radianZ, axis)
    p = np.einsum("ij, kj->ki", matrixZ, p)

    #print(np.where(p > 0, p + 0.5, p - 0.5).astype(int))
    
    return matrixZ @ matrixY @ matrixX, p


def clipXYZ(array, imgShape):
    '''
    Turn the minimun of array into 0 and the maximun of array into imgshape per axis. 
    
    '''
    check = True
    for i in range(3):
        if check:
            b = np.clip(array[..., i : i + 1], 1, imgShape[i] - 1)
            check = False
        else:
            b = np.concatenate([b, np.clip(array[..., i : i + 1], 1, imgShape[i] - 1)], axis=-1)
    
    return b

def determineClipSize(boundingVertics, imgShape, expansion=0):
    o = boundingVertics[0].astype(int)
    
    imgShape = np.array(imgShape) - 1
    o = o - expansion
    #o = clipXYZ(o, imgShape)
    o = np.clip(o, 1, imgShape)

    c = sum(boundingVertics) - 3 * boundingVertics[0]
    c = (c + 1).astype(int)

    c = c + expansion
    #c = clipXYZ(c, imgShape)
    c = np.clip(c, 1, imgShape)

    origin = np.where(o < c, o, c)
    clipSize = np.where(o < c, c, o)


    return origin, clipSize


def makeRefCoords(imgArray, rotationMatrix):
    # For affine transformation, make dummyArray that is a matrix with the coordinates for each pixels
    SL, CL, AL = imgArray.shape
    # For affine transformation, make dummyArray that is a matrix with the coordinates for each pixels
    x, y, z = np.mgrid[:SL, :CL, :AL]
    dummyArray = np.stack([x, y, z, np.ones((SL, CL, AL))], axis=-1)
    
    # Caluculate the coordinates before conversion that each pixel after conversion should refer to
    refCoords = np.einsum('ijkm, lm->ijkl', dummyArray, rotationMatrix)
    
    return refCoords[...,:3]

def transformImageArray(imgArray, refCoords, interpolation):
    coordsLimits = np.array(imgArray.shape) - 1
    if interpolation == "nearest":
        refCoordsNearest = np.where(refCoords > 0, refCoords + 0.5, refCoords - 0.5).astype(int)
        #refCoordsNearest = clipXYZ(refCoordsNearest, imgArray.shape)
        refCoordsNearest = np.clip(refCoordsNearest, 1, coordsLimits)
        
        
        rotatedImageArray = imgArray[refCoordsNearest[...,0], refCoordsNearest[..., 1], refCoordsNearest[..., 2]]

    if interpolation == "linear":
        # Caluculate 8 vertics around
        linearCoords = [0]*8
        linearCoords[0] = np.clip(np.copy(refCoords.astype(int)), 1, coordsLimits)
        linearCoords[1] = np.clip(linearCoords[0] + [1, 0, 0], 1, coordsLimits)
        linearCoords[2] = np.clip(linearCoords[1] + [0, 1, 0], 1, coordsLimits)
        linearCoords[3] = np.clip(linearCoords[0] + [0, 0, 1], 1, coordsLimits)
        linearCoords[4] = np.clip(linearCoords[0] + [1, 1, 0], 1, coordsLimits)
        linearCoords[5] = np.clip(linearCoords[0] + [1, 0, 1], 1, coordsLimits)
        linearCoords[6] = np.clip(linearCoords[0] + [0, 1, 1], 1, coordsLimits)
        linearCoords[7] = np.clip(linearCoords[0] + [1, 1, 1], 1, coordsLimits)
        linearCoords = np.stack(linearCoords, axis=0)
        
        diff = np.copy(refCoords - linearCoords[-1,:,:,:,:])

        del refCoords
        gc.collect()

        #linearCoords = np.clip(linearCoords, 1, coordsLimits)
        #Caluculate weights
        linearWeight = [0] * 8
        linearWeight[0] = (1 - diff[...,0]) * (1 - diff[...,1]) * (1 - diff[...,2])
        linearWeight[1] = diff[...,0] * (1 - diff[...,1]) * (1 - diff[...,2])
        linearWeight[2] = (1 - diff[...,0]) * diff[...,1] * (1 - diff[...,2])
        linearWeight[3] = (1 - diff[...,0]) * (1 - diff[...,1]) * diff[...,2]
        linearWeight[4] = diff[...,0] * diff[...,1] * (1 - diff[...,2])
        linearWeight[5] = diff[...,0] * (1 - diff[...,1]) * diff[...,2]
        linearWeight[6] = (1 - diff[...,0]) * diff[...,1] * diff[...,2]
        linearWeight[7] = diff[...,0] * diff[...,1] * diff[...,2]

        linearWeight = np.stack(linearWeight, axis=-1)
        
        #linearCoords = clipXYZ(linearCoords, imgArray.shape)
        rotatedImageArray = np.einsum('ijkm, mijk->ijk', linearWeight,
                imgArray[linearCoords[...,0], linearCoords[...,1], linearCoords[...,2]])
        
    return rotatedImageArray


def determineSlide(boundindVertics, rotationMatrix, imgArray):
    # Slide to the minimum location is 0.

    Min = boundindVertics.min(axis=0)
    print("Minimum list : ", Min)
    
    slideMin = np.where(Min < 0, -Min, 0).astype(int)
    print("So, slide by ", slideMin)

    # Slide to the maximum location is imgArray.shape - 1
    Max = boundindVertics.max(axis=0)
    print("imgArray shape : ", imgArray.shape)
    print("MaximumList : ", Max)
    slideMax = np.where(Max >= imgArray.shape, -(Max - imgArray.shape) - 1, 0).astype(int)
    print("So, slide by ", slideMax)


    slide = slideMin + slideMax

    # Integrate slide into rotationMatrix
    affineMatrix = np.array([[*rotationMatrix[0], slide[0]], 
                             [*rotationMatrix[1], slide[1]],
                             [*rotationMatrix[2], slide[2]],
                             [0, 0, 0, 1]])
    
    return slide, affineMatrix

def reverseImage(imgArray):
    arg = np.argsort(imgArray.shape)[::-1]
    print(arg)

    if arg[1] == 0:
        return imgArray[::-1, :, :]
    elif arg[1] == 1:
        return imgArray[:, ::-1, :]
    else:
        return imgArray[:, :, ::-1]
    

def Resizing(source, ref, initerpolation):
    magnification = np.array(ref.shape) / np.array(source.shape)
    print("magnification : ", magnification)
    print("Transorm source image From {} shape into {} shape".format(source.shape, ref.shape))
    zoomMatrix = np.array([[1 / magnification[0], 0, 0, 0], 
                           [0, 1 / magnification[1], 0, 0], 
                           [0, 0, 1 / magnification[2], 0],
                           [0, 0, 0, 1]])
    
    refCoords = makeRefCoords(ref, zoomMatrix)
    
    zoomedArray =  transformImageArray(source, refCoords, initerpolation)
    
    print("zoomedArray shape : ", zoomedArray.shape)
    
    return zoomedArray
