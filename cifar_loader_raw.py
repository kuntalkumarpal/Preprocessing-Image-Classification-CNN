'''
Loads raw data into shared memory
No preprocessing done
'''

import numpy as np
import cPickle
import gzip
import theano
import theano.tensor as T
import random


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
    

    #--------------------------------------------------------------------
    #Convert nx1024x3 -> nx3x1024 Traindata
    trainX = np.reshape(inputX[noValid:(size-noTest)],(noTrain,1024,3)) #->40000,1024,3
    c = [ np.reshape((np.concatenate((oo[:,0].flatten(),oo[:,1].flatten(),oo[:,2].flatten()),
                       axis=0)),(3,1024)) for oo in trainX ]
    trainX = np.array(c) #40000,3,1024
    trainY = inputY[noValid:(size-noTest)].tolist()
    trainData = [(x,y) for (x,y) in zip(trainX,trainY)] #change for randomization setting

    #--------------------------------------------------------------------
    #Convert nx1024x3 -> nx3x1024 validdata
    validX = np.reshape(inputX[:noValid],(noValid,1024,3)) #->40000,1024,3
    c = [ np.reshape((np.concatenate((oo[:,0].flatten(),
                                      oo[:,1].flatten(),
                                      oo[:,2].flatten()),
                       axis=0)),(3,1024)) for oo in validX ]
    #print len(c)
    validX = np.array(c) #40000,3,1024    
    validInp = [ x for x in validX ]
    validData = [validInp, inputY[:noValid]]
    #--------------------------------------------------------------------
    #Convert nx1024x3 -> nx3x1024 Traindata
    testX = np.reshape(inputX[(size-noTest):],(noTest,1024,3)) #->40000,1024,3
    c = [ np.reshape((np.concatenate((oo[:,0].flatten(),
                                      oo[:,1].flatten(),
                                      oo[:,2].flatten()),
                       axis=0)),(3,1024)) for oo in testX ]
    testX = np.array(c) #40000,3,1024   
    testInp = [ x for x in testX ]
    testData = [testInp,inputY[(size-noTest):]]
    #--------------------------------------------------------------------

    print trainData[0]
    random.shuffle(trainData)
    print "--------------------shuffling done--------------------"
    print trainData[0]

    
    trainX = [x[0] for x in trainData]
    trainY = [x[1] for x in trainData]
    trainData = [trainX,trainY]
    
    def shared(data):
        sharedX = theano.shared(np.asarray(data[0],dtype=theano.config.floatX),
                                borrow=True)
        sharedY = theano.shared(np.asarray(data[1],dtype=theano.config.floatX),
                                borrow=True)
        return sharedX, T.cast(sharedY,"int32")

    return [shared(trainData), shared(validData), shared(testData)]





if __name__ == '__main__' :

    trainData, validData, testData = loadData()
    print trainData[0].get_value(borrow=True).dtype
    print trainData[0]
    print trainData[1]
