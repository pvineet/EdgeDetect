
import numpy as np
from scipy.ndimage.filters import convolve, gaussian_filter
from scipy.misc import imread, imshow

def gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask
    """
    m,n = [(i-1.)/2. for i in shape] # Centre the indices around 0
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma))
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    #h = h/np.sqrt(2*np.pi*sigma*sigma)
    return h

def LoGKernel(shape=(5,5), sigma=0.6):
    """
    2D laplacian of gaussian mask
    """
    m,n = [(i-1.)/2. for i in shape] # Centre the indices around 0
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma))
    m = (x*x + y*y - (2*sigma*sigma))/(sigma**4)
    h *= m
    h = h/np.sqrt(2*np.pi*sigma*sigma)
    return h

def DoGKernel(shape=(5,5), sigma1=0.5, sigma2=1.0):
    """
    2D difference of gaussian mask
    """
    if sigma1 > sigma2:
        return gauss2D(shape=shape, sigma=sigma1) - gauss2D(shape=shape, sigma=sigma1)
    elif sigma2 > sigma1:
        return gauss2D(shape=shape, sigma=sigma2) - gauss2D(shape=shape, sigma=sigma1)
    else:
        raise ValueError("Sigma1 and Sigma2 have same value.")


def non_max_suppression(mag, theta):
    """
    Perform Non Max. Suppression for Canny Edge Detector.
    """
    w, h = theta.shape
    non_max_sup = np.zeros((w,h))
    for i in range(1, w-1):
        for j in range(1, h-1):
            if (theta[i][j] >= -22.5 and theta[i][j] <= 22.5) or (theta[i][j] < -157.5 and theta[i][j] >= -180):
                if mag[i][j] >= mag[i][j+1] and mag[i][j] >= mag[i][j-1]:
                    non_max_sup[i][j]= mag[i][j]
                else:
                    non_max_sup[i][j]=0;
            elif (theta[i][j] >= 22.5 and theta[i][j] <= 67.5) or (theta[i][j] < -112.5 and theta[i][j] >= -157.5):
                if mag[i][j] >= mag[i+1][j+1] and mag[i][j] >= mag[i-1][j-1]:
                    non_max_sup[i][j]= mag[i][j]
                else:
                    non_max_sup[i][j]=0;
            elif (theta[i][j] >= 67.5 and theta[i][j] <= 112.5) or (theta[i][j] < -67.5 and theta[i][j] >= -112.5):
                if mag[i][j] >= mag[i][j+1] and mag[i][j] >= mag[i][j-1]:
                    non_max_sup[i][j]= mag[i][j]
                else:
                    non_max_sup[i][j]=0;
            elif (theta[i][j] >= 112.5 and theta[i][j] <= 157.5) or (theta[i][j] < -22.5 and theta[i][j] >= -67.5):
                if mag[i][j] >= mag[i][j+1] and mag[i][j] >= mag[i][j-1]:
                    non_max_sup[i][j]= mag[i][j]
                else:
                    non_max_sup[i][j]=0;
    return non_max_sup

def hysteresis(non_max_sup, mag):
    highThreshold = 91
    lowThreshold = 31
    #Double threshold
    strongEdges = (non_max_sup > highThreshold)

    # Thresholded Egdes 
    thresholdedEdges = np.bitwise_and((non_max_sup < highThreshold), (non_max_sup > lowThreshold))
    #Tracing edges with hysteresis  
    #Find weak edge pixels near strong edge pixels
    w,h = non_max_sup.shape

    for i in range (1, w-2):
        for j in range(1, h-2):
            if thresholdedEdges[i][j]:
                if not np.max(strongEdges[i-1:i+2, j-1:j+2]):
                    thresholdedEdges[i][j] = False

    finalEdges = mag*np.bitwise_or(strongEdges, thresholdedEdges)
    return finalEdges
