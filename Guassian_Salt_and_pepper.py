'''
Created on Sep 18, 2016

@author: Kashyap
'''
from PIL import Image  
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageOps import gaussian_blur


I = array(Image.open('input_image2.jpg').convert('L'))
print(I)
Iact= array(Image.open('output_image2.png').convert('L'))
print(Iact)

#This method applies Gaussian and Salt pepper filters separately. 1st Gaussian filter is applied, its canny edges and performance 
#Parameters and calculated. Secondly, to the true image, Salt pepper filter is applied and then its performance parameters
#are calculated separately. This is done for the problem (2b)
def GaussianAndSaltPepper(I):
    sigma = .5
    mid =  10
    result = np.zeros( 2*mid + 1)
    G = [(1/(sigma*np.sqrt(2*np.pi)))*(1/(np.exp((i**2)/(2*sigma**2)))) for i in range(-mid,mid+1)]  
    n,m=I.shape
    Ig=zeros((n,m))
    for i in range(n):
        Ig[i,:] = np.convolve(I[i,mid:m - mid] , G)
    for j in range(m):
        Ig[:,j] = np.convolve(I[mid:n - mid,j] , G)
    

    plt.subplot(231)
    plt.imshow(Ig)
    title('Guassian Filter')
    plt.show() 
    
    TP1, FP1, TN1, FN1=perf_measure(Iact,calculateHys(Ig))
    calculatePerformance(TP1, FP1, TN1, FN1)
    
    
    Irand=zeros((n,m))
    for j in range(m):
        for i in range(n):
            Irand[i,j]=np.random.randint(0,10)
    
    print(Irand)
    
    for j in range(m):
        for i in range(n):
            if(Irand[i,j]==0):
                I[i,j]=0
            if(Irand[i,j]==10):
                I[i,j]=255
    
    print(Ig) 
    
    plt.subplot(231)
    plt.imshow(Ig)
    title('Salt and Pepper Filter')
    plt.show()    
    
    TP2, FP2, TN2, FN2=perf_measure(Iact,calculateHys(I))
    calculatePerformance(TP2, FP2, TN2, FN2)
        
    
#Calculates the performance parameters
def calculatePerformance(TP,FP,TN,FN):
    sensitivity=0
    specificity=0
    precision=0
    negativePredValue=0
    fallout=0
    FNR=0
    FDR=0
    accuracy=0
    fscore=0
    MCC=0
    
    sensitivity=TP/(TP+FN)
    print('Sensitivity='+str(sensitivity))
    
    specificity=TN/(TN+FP)
    print('specificity= '+str(specificity))
    
    precision=TP/(TP+FP)
    print('precision= '+str(precision))
    
    negativePredValue=TN/(TN+FN)
    print('negativePredValue= '+str(negativePredValue))
    
    fallout=FP/(TP+TN)
    print('fallout= '+str(fallout))
    
    FNR=FN/(FN+TP)
    print('FNR= '+str(FNR))
    
    FDR=FP/(FP+TP)
    print('FDR= '+str(FDR))
    
    accuracy=(TP+TN)/(TP+FN+TN+FP)
    print('accuracy= '+str(accuracy))
    
    fscore=2*TP/(2*TP+FP+FN)
    print('fscore= '+str(fscore))
    
    MCC=((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    print('MCC= '+str(MCC))
    

#Calculates the required parameters, TP,TN,FP,FN
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    n,m=y_actual.shape
    print(n,m)
    for j in range(m):
        for i in range(n):
            if(y_actual[i,j]==y_hat[i,j]==255):
                TP+=1
                #print("I came here")
                
    for j in range(m):
        for i in range(n):
            if(y_actual[i,j]==255 and y_hat[i,j]!=y_actual[i,j]):
                FP+=1
                
    for j in range(m):
        for i in range(n):
            if(y_actual[i,j]==0 and y_hat[i,j]!=y_actual[i,j]):
                FN+=1
                
    for j in range(m):
        for i in range(n):
            if(y_actual[i,j]==y_hat[i,j]==0):
                TN+=1

    return(TP, FP, TN, FN)

#returns the Canny edge matrix
def calculateHys(I):
    
    sigma = .5
    mid =  10
    result = np.zeros( 2*mid + 1 )
    G = [(1/(sigma*np.sqrt(2*np.pi)))*(1/(np.exp((i**2)/(2*sigma**2)))) for i in range(-mid,mid+1)]  


    Gx = zeros(len(G))
    for i in range(1,len(G)):
        Gx[i] = G[i] - G[i-1]


    n, m = I.shape
 
    Ix = zeros((n,m))
    Iy = zeros((n,m))
    for i in range(n):
        Ix[i,:] = np.convolve(I[i,mid:m - mid] , G)
    figure()
    gray()
    plt.subplot(231)
    plt.imshow(Ix)
    title('X filter')
  
    for j in range(m):
        Iy[:,j] = np.convolve(I[mid:n - mid,j] , G)
    plt.subplot(232)
    plt.imshow(Iy)
    title('Y filter')
   

    n, m = I.shape
    IxPrime = zeros((n,m))
    IyPrime= zeros((n,m))
    for i in range(n):
        IxPrime[i,:] = np.convolve(Ix[i,mid:m - mid] , Gx)

    plt.subplot(233)
    plt.imshow(IxPrime)
    title('Prime X')
  
    for j in range(m):
        IyPrime[:,j] = np.convolve(Iy[mid:n - mid,j] , Gx)

    plt.subplot(234)
    plt.imshow(IyPrime)
    title('Prime Y')
   

    magnetude = zeros((n,m))

    for i in range(n):
        for j in range(m):
            magnetude[i,j]= np.sqrt(IxPrime[i,j]**2 + IyPrime[i,j]**2)        
    plt.subplot(235)
    plt.imshow(magnetude)
    title('Magnitude')


    theta = zeros((n,m))
    count = 0
 
    for i in range(1,n - 1):
        for j in range(1,m - 1):
            theta[i,j] = math.atan2(IyPrime[i,j], IxPrime[i,j])
 
        if((-np.pi/2) <= theta[i,j] < (-3*np.pi/8) or (3*np.pi/8) <=  theta[i,j] <=  (np.pi/2)):
            if (magnetude[i,j] <= magnetude[i-1,j] or magnetude[i,j] <= magnetude[i+1,j]):
                magnetude[i,j] = 0
                count +=1
        elif((-3*np.pi/8) <=  theta[i,j] < (-np.pi/8)):
            if (magnetude[i,j] <= magnetude[i-1,j-1] or magnetude[i,j] <= magnetude[i+1,j+1]):
                magnetude[i,j] = 0
                count +=1
        elif((-np.pi/8) <=  theta[i,j] < (np.pi/8)):
            if (magnetude[i,j] <= magnetude[i,j-1] or magnetude[i,j] <= magnetude[i,j+1]):
                magnetude[i,j] = 0
                count +=1
        elif((np.pi/8) <=  theta[i,j] < (3*np.pi/8)):
            if (magnetude[i,j] <= magnetude[i+1,j-1] or magnetude[i,j] <= magnetude[i-1,j+1]):
                magnetude[i,j] = 0
                count +=1



    low = 10
    high = 70 
    count = 0
    for i in range(1,n-1):
        for j in range(1,m-1):
            if (magnetude[i,j] <= low):
                magnetude[i,j] = 0
                count +=1
            elif (magnetude[i,j] >= high):
                magnetude[i,j] = 255
                count +=1
            elif (low < magnetude[i,j] < high):
                if (magnetude[i-1,j-1] == 255 and magnetude[i-1,j]== 255 and magnetude[i,j-1] == 255 
                        and magnetude[i+1,j+1] == 255 and magnetude[i+1,j] == 255 and magnetude[i,j+1] == 255 
                        and magnetude[i,j] == 255 and magnetude[i,j] == 255):
                        magnetude[i,j] = 255
                        count +=1
                else:
                    magnetude[i,j] = 0
                    count +=1
    
    plt.subplot(236)
    plt.imshow(magnetude)
    title('Hysteresis')
    plt.show()
    return magnetude


#This is for the PROB 2a where Canny eDges are found out and performance parameters are found out. This is done with actual image
#and the image after finding the Canny edges. 
TP, FP, TN, FN=perf_measure(Iact,calculateHys(I))

print(TP)
print(FP)
print(TN)
print(FN)

#Performance is calculated (2a)
calculatePerformance(TP, FP, TN, FN)

GaussianAndSaltPepper(I)
