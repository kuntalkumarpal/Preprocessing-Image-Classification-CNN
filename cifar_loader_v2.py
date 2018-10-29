#CIFAR10 Mean Normalized

import numpy as np
import cPickle
import gzip
import theano
import theano.tensor as T


def loadData(dataset = 'datasets/cifar-10-batches-py', noValid=10000, noTest=10000):

    inputX=[]
    inputY=[]
    print "Cifar Loader : ",theano.config.device
    
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
    #print inputY.ndim
    print x.ndim, x.shape, x.dtype

    #--------- ZERO CENTERING ------------
    validX = np.reshape(inputX[:noValid],(noValid,dimension))
    trainX = np.reshape(inputX[noValid:(size-noTest)],(noTrain,dimension))
    testX = np.reshape(inputX[(size-noTest):],(noTest,dimension))
    
    print trainX.shape
    #Preprocessing by calculating mean over all pixels of only Training data
    ''' ZERO CENTERING -> BY SUBTRACTING MEAN '''
    trainMean = np.mean(trainX, axis=0)
    trainStd = np.std(trainX, axis=0)
    print "Mean : ",trainMean
    print "Std Dev :",trainStd
    print trainStd.shape, trainX.shape
    #print trainX
    print '-----------------------------------------------------------'
    trainX=trainX.astype(float)
    validX=validX.astype(float)
    testX=testX.astype(float)

    ''' Mean Normalization '''
    trainX -= trainMean
    validX -= trainMean
    testX -= trainMean

    print '---------------Mean Normalization Done------------------'
    
    validInp = [ np.reshape(x,(1024,3)) for x in validX ]
    validData = [validInp, inputY[:noValid]]

    trainInp = [ np.reshape(x,(1024,3)) for x in trainX ]
    trainData = [trainInp, inputY[noValid:(size-noTest)]]

    testInp = [ np.reshape(x,(1024,3)) for x in testX ]
    testData = [testInp,inputY[(size-noTest):]]


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
    
    print len(trainData[0].get_value(borrow=True)),len(validData[0].get_value(borrow=True)),len(testData[0].get_value(borrow=True))
