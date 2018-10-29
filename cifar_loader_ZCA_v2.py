'''
Preprocessing ZCA
1. /255.
2. mean normalized accross features
3. zca application e=0.1
4. change shape 
'''

import numpy as np
import cPickle
import gzip
import theano
import theano.tensor as T
import cv2
import time
import matplotlib.pyplot as plt

def printImages(arr,noImages,imgname):
    xx=[]
    for i in xrange(32):
        for j in xrange(noImages):
            xx.append(arr[j][i*32*3:i*32*3+32*3])
            
    zz = np.reshape(np.concatenate(xx),(32,noImages*32,3))
    cv2.imwrite(imgname,zz)



def printcifar(dataL,labelsL,imgname):

    w = h = 32
    nclass = 10
    nimg = 10
    gap = 4

    width  = nimg * ( w + gap ) + gap
    height = nclass * ( h + gap ) + gap
    img = np.zeros( ( height, width, 3 ), dtype = int ) 
    #cv2.imwrite( 'hoge1.png', img )

    print dataL.shape
    print labelsL.shape
    for iy in range( nclass ):
        lty = iy * ( h + gap ) + gap
        #print lty
        idx = np.where( labelsL == iy )[0]  #indexes of all labels = class
        #print idx
        
        for ix in range( nimg ):
            ltx = ix * ( w + gap ) + gap
            #print 'data = ',dataL[idx[ix], :]
            #print dataL[idx[ix], :].shape
            tmp = dataL[idx[ix], :].reshape( ( h, w,3 ) )
            #print tmp.shape
            #print 'tmp = ',tmp
            
            #print tmp.shape
            # BGR <= RGB
            #print tmp
            img[lty:lty+h, ltx:ltx+w, 0] = tmp[ :, :,2]
            img[lty:lty+h, ltx:ltx+w, 1] = tmp[ :, :,1]
            img[lty:lty+h, ltx:ltx+w, 2] = tmp[ :, :,0]

    
    cv2.imwrite( imgname, img )



def printcifarZCA(dataL,labelsL,imgname):

    w = h = 32
    nclass = 10
    nimg = 10
    gap = 4

    width  = nimg * ( w + gap ) + gap
    height = nclass * ( h + gap ) + gap
    img = np.zeros( ( height, width, 3 ), dtype = int ) 

    print dataL.shape
    print labelsL.shape
    for iy in range( nclass ):
        lty = iy * ( h + gap ) + gap
        #print lty
        idx = np.where( labelsL == iy )[0]  #indexes of all labels = class
        #print idx
        
        for ix in range( nimg ):
            ltx = ix * ( w + gap ) + gap
            absmax = np.max( np.abs( dataL[idx[ix], :] ) )
            tmp = dataL[idx[ix], :].reshape( ( h, w,3 ) )/absmax *127 +128
            img[lty:lty+h, ltx:ltx+w, 0] = tmp[ :, :,2]
            img[lty:lty+h, ltx:ltx+w, 1] = tmp[ :, :,1]
            img[lty:lty+h, ltx:ltx+w, 2] = tmp[ :, :,0]

    
    cv2.imwrite( imgname, img )



def loadData(dataset = 'datasets/cifar-10-batches-py', noValid=10000, noTest=10000):

    inputX=[]
    inputY=[]
    
    def unpickle(datafile):
        f = open(datafile,'rb')
        d = cPickle.load(f)
        f.close()
        return d
        
    # reading train data
    for i in xrange(5):
        data = unpickle(dataset+'/data_batch_'+str(i+1))
        x = data['data']
        y = data['labels']
        inputX.append(x)
        inputY.append(y)

    # reading test data
    data = unpickle(dataset+'/test_batch')
    inputX.append(data['data'])
    inputY.append(data['labels'])

    x = np.concatenate(inputX)
    y = np.concatenate(inputY)

    inputX = np.dstack( (x[:,:1024], x[:,1024:2048], x[:,2048:]) )
    inputY = y
    print inputX.shape,inputY.shape


    size = inputX.shape[0]
    dimension = 3*32*32
    noTrain = size - noTest - noValid
    
    inputY = np.reshape(inputY,(size,)) #1D vector
    
    #--------- ZERO CENTERING ------------
    validX = np.reshape(inputX[:noValid],(noValid,dimension))
    trainX = np.reshape(inputX[noValid:(size-noTest)],(noTrain,dimension))
    testX = np.reshape(inputX[(size-noTest):],(noTest,dimension))


    printImages(trainX,10,'1.raw.jpg')
    print inputY[noValid:(size-noTest)].shape
    print trainX.shape
    printcifar(trainX,inputY[noValid:(size-noTest)],'1.raw2.png')

 
    trainX=trainX.astype(dtype='float32')
    validX=validX.astype(dtype='float32')
    testX=testX.astype(dtype='float32')

    trainX = trainX/255.
    validX = validX/255.
    testX = testX/255.
    
    #Preprocessing by calculating mean over all pixels of only Training data
    ''' ZERO CENTERING -> BY SUBTRACTING MEAN '''
    trainMean = np.mean(trainX,axis=0)
    trainStd = np.std(trainX,axis=0)
    print "Mean : ",trainMean
    print "Std Dev :",trainStd
    print trainStd.shape, trainMean.shape
    print '-----------------------------------------------------------'
    
    
    print trainX[0]
    ''' Mean Normalization '''
    trainX -= trainMean
    validX -= trainMean
    testX -= trainMean
    print trainX[0]

    printImages(trainX,10,'2.meannorm.jpg')
    printcifar(trainX,inputY[noValid:(size-noTest)],'2.meannorm2.jpg')
    
    print '---------------Mean Normalization Done------------------'
    
    '''trainX /= trainStd
    validX /= trainStd
    testX /= trainStd'''
    
    '''trainX = np.round(trainX,decimals=3)
    validX = np.round(validX,decimals=3)
    testX = np.round(testX,decimals=3)
    '''
    print trainX[0]

    #print '---------------Variance Normalization Done------------------'
    cov = np.dot(trainX.T,trainX) / trainX.shape[0]
    print '	* Covariance Calc Done *	'
    #print cov
    U, S, V = np.linalg.svd(cov)
    print '	* SVD Done *	'
    #print U.shape,S.shape,V.shape #(3072x3072)(3072,) (3072x3072)
    
    epsilon = 0.1
    #print zcaWhiteMat.shape  #(3072, 3072)


    sqlam = np.sqrt(S+epsilon)
    zcaWhiteMat = np.dot(U/sqlam[np.newaxis , :],U.T)


    trainX = np.dot(trainX,zcaWhiteMat.T)
    print '	* TrainX ZCA Done *	'
    validX = np.dot(validX,zcaWhiteMat.T)
    print '	* ValidX ZCA Done *	'
    testX = np.dot(testX,zcaWhiteMat.T)
    
    print '---------------Whitening Done------------------'
    
    print trainX[0]    

    printImages(trainX,10,'3.whitening.jpg')
    printcifarZCA(trainX,inputY[noValid:(size-noTest)],'3.whitening2.jpg')


    #Convert nx1024x3 -> nx3x1024 Traindata
    trainX = np.reshape(trainX,(noTrain,1024,3)) #->40000,1024,3
    c = [ np.reshape((np.concatenate((oo[:,0].flatten(),
                                      oo[:,1].flatten(),
                                      oo[:,2].flatten()),
                       axis=0)),(3,1024)) for oo in trainX ]
    #print len(c)
    trainX = np.array(c) #40000,3,1024

    
    trainY = inputY[noValid:(size-noTest)].tolist()
    trainData = [(x,y) for (x,y) in zip(trainX,trainY)] #change for randomization setting

    #Convert nx1024x3 -> nx3x1024 validdata
    validX = np.reshape(validX,(noValid,1024,3)) #->40000,1024,3
    c = [ np.reshape((np.concatenate((oo[:,0].flatten(),
                                      oo[:,1].flatten(),
                                      oo[:,2].flatten()),
                       axis=0)),(3,1024)) for oo in validX ]
    #print len(c)
    validX = np.array(c) #40000,3,1024    
    validInp = [ x for x in validX ]
    validData = [validInp, inputY[:noValid]]



    #Convert nx1024x3 -> nx3x1024 Traindata
    testX = np.reshape(testX,(noTest,1024,3)) #->40000,1024,3
    c = [ np.reshape((np.concatenate((oo[:,0].flatten(),
                                      oo[:,1].flatten(),
                                      oo[:,2].flatten()),
                       axis=0)),(3,1024)) for oo in testX ]
    print len(c)
    testX = np.array(c) #40000,3,1024   
    testInp = [ x for x in testX ]
    testData = [testInp,inputY[(size-noTest):]]
    
    stime = time.time()
    f = open("ZCANormalized.pkl","wb")
    cPickle.dump([trainData,validData,testData],f,protocol=cPickle.HIGHEST_PROTOCOL)
    print "Time Test Save :",round((time.time() - stime),2)



if __name__ == '__main__' :

    loadData()
    
