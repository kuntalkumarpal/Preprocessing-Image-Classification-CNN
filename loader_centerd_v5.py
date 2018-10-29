'''
#used randomizations
'''
import numpy as np
import time
import cPickle
import theano
import theano.tensor as T
import random

def loadNormData(filename="ZCANormalized.pkl"):
    stime = time.time()
    trainfile=open(filename,"rb")
    data = cPickle.load(trainfile)
    print "Time Load :",round((time.time() - stime),2)
    print len(data)

    print data[0][0]
    random.shuffle(data[0])
    print "--------------------shuffling done--------------------"
    print data[0][0]
    #unzipped = zip(*data[0]) #decoupling
    #print unzipped[0][0], unzipped[1][0]
    #kk()
    trainX = [x[0] for x in data[0]]
    trainY = [x[1] for x in data[0]]
    train = [trainX,trainY]
    print len(trainX)
    print len(trainY)
    '''print trainX[:10]
    print trainY[:10]
    '''
    stime = time.time()
    def shared(data):
        sharedX = theano.tensor._shared(np.asarray(data[0],dtype=theano.config.floatX),
                                borrow=True)
        sharedY = theano.tensor._shared(np.asarray(data[1],dtype=theano.config.floatX),
                                borrow=True)
        return sharedX, T.cast(sharedY,"int32")

    return [shared(train), shared(data[1]), shared(data[2])]


if __name__ == '__main__' :

    trainData, validData, testData = loadNormData()
    
    print trainData[0].get_value(borrow=True).dtype
    print trainData[0]
    print trainData[1]
