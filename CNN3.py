#10. convnet_2CP1F_ReLU_vLR_shuf.py
'''
2 convolutional-pooling layer, 1 Fully-connected layer, 1 softmax layer, ReLU, variable learning rate, shuffling of data  
'''

import numpy as np
import gzip

#import cifar_loader_v2 as cifarLoader
import loader_centerd_v5 as cifarLoader #shuffled data
#import cifar_loader_raw as  cifarLoader

import theano
import theano.tensor as T

from theano.tensor.nnet import conv
from theano.tensor.nnet import sigmoid
from theano.tensor.nnet import softmax

from theano.tensor.signal import downsample
from theano.tensor import shared_randomstreams

import time


def ReLU(z):
    return T.maximum(0.0, z)

def dropoutLayer(layer, pDropout):
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-pDropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)

def size(data):
    return data[0].get_value(borrow=True).shape[0]



class Network(object):

    def __init__(self, layers, mbSize):

        self.layers = layers
        self.mbSize = mbSize
        self.params = [ p for layer in self.layers for p in layer.params]
        print "Layers : ",self.layers
        print "MiniBatch Size : ",self.mbSize
        print "Params : ",self.params
        
        #self.x = T.matrix("x")
        self.x = T.ftensor3("x")
        self.y = T.ivector("y")

        initLayer = self.layers[0] #ConvPoolLayer
        initLayer.setInputOutput(self.x, self.x, self.mbSize)
        
        for j in xrange(1, len(self.layers)):
            prevLayer, currLayer = self.layers[j-1], self.layers[j]
            currLayer.setInputOutput(prevLayer.out, prevLayer.outDropout, self.mbSize)
        self.out = self.layers[-1].out
        self.outDropout = self.layers[-1].outDropout


    def mbSGD(self, trainData, validData, testData, epochs, mbSize, learnRate, lmbda=0.0 ):

        trainX, trainY = trainData
        validX, validY = validData
        testX, testY = testData

        # Minibatch Calculation
        noTrainBatch = size(trainData)/mbSize
        noValidBatch = size(validData)/mbSize
        noTestBatch = size(testData)/mbSize
        print "Train Batch = ",noTrainBatch
        print "Valid Batch = ",noValidBatch
        print "Test Batch = ",noTestBatch

        # Symbolic assignments
        lrate = T.scalar('lr')
        lrate = learnRate
        regCostL2 = sum( [ (layer.w**2).sum() for layer in self.layers] )
        cost = self.layers[-1].cost(self) + 0.5*lmbda*regCostL2/noTrainBatch
        gradients = T.grad(cost, self.params)
        updates = [ (param, param-lrate*grad)
                    for param, grad in zip(self.params, gradients) ]

        i = T.lscalar() #mb index
        print "DTYPE : ",trainY[i*self.mbSize: ].dtype,trainY[i*self.mbSize: ].ndim
        print "DTYPE : ",trainX[i*self.mbSize: ].dtype,trainX[i*self.mbSize: ].ndim
        trainMB = theano.function( [i],
                                   cost,
                                   updates = updates,
                                   givens = { self.x: trainX[i*self.mbSize: (i+1)*self.mbSize],
                                              self.y: trainY[i*self.mbSize: (i+1)*self.mbSize]
                                            }
                                 )
        validMBAccuracy = theano.function( [i],
                                           self.layers[-1].accuracy(self.y),
                                           givens = { self.x: validX[i*self.mbSize: (i+1)*self.mbSize],
                                                      self.y: validY[i*self.mbSize: (i+1)*self.mbSize]
                                                    }
                                         )
        testMBAccuracy = theano.function( [i],
                                           self.layers[-1].accuracy(self.y),
                                           givens = { self.x: testX[i*self.mbSize: (i+1)*self.mbSize],
                                                      self.y: testY[i*self.mbSize: (i+1)*self.mbSize]
                                                    }
                                         )
        testMBPredictions = theano.function( [i],
                                             self.layers[-1].yOut,
                                             givens = { self.x: testX[i*self.mbSize: (i+1)*self.mbSize]
                                                      }
                                           )

        # Training
        print '----------Training Begun-------------'
        bestValidAccuracy = 0.0
        bestIteration = 0
        lastUpdtEpoch = 0
        lrChange = 0
        for epoch in xrange(epochs):
            if(epoch - lastUpdtEpoch)>10 and lrChange<4:
                lrate = lrate/10
                print 'Learn Rate changed',lrate
                lastUpdtEpoch = epoch
                lrChange +=1
            for mbIndex in xrange(noTrainBatch):
                iteration = noTrainBatch*epochs+mbIndex
                if iteration%1000 == 0:
                    print 'Training MB no : ',iteration
                costij = trainMB(mbIndex)
                if (iteration+1)%noTrainBatch == 0:
                    validAccuracy = np.mean([ validMBAccuracy(j) for j in xrange(noValidBatch)])
                    print ("Epoch {0} : Validation Accuracy : {1:.2%}".format(epoch,validAccuracy))
                    if validAccuracy >= bestValidAccuracy :
                        print "BEST TILL DATE"
                        bestValidAccuracy = validAccuracy
                        bestIteration = iteration
                        lastUpdtEpoch = epoch
                        if testData :
                            testAccuracy = np.mean( [testMBAccuracy(j) for j in xrange(noTestBatch)] )
                            print("\tTest Accuracy : {0:.2%}".format(testAccuracy))
                            predOP = [testMBPredictions(j) for j in xrange(noTestBatch)]


        print("------------------Finished training network------------------")
        print predOP
        print("Best validation accuracy : {0:.2%} at iteration {1}".format(
            bestValidAccuracy, bestIteration))
        print("Corresponding test accuracy of {0:.2%}".format(testAccuracy))



        

class ConvPoolLayer(object):

    def __init__(self, lrfShape, imgShape, poolSize=(2,2), activation=sigmoid):
        self.lrfShape = lrfShape
        self.imgShape = imgShape
        self.poolSize = poolSize
        self.activation = activation
        print 'Filter Shape : ',self.lrfShape
        print 'Image Shape : ',self.imgShape
        print 'Maxpool Size : ',self.poolSize
        print 'Activation : ',self.activation

        noOutNeuron = ( lrfShape[0]*np.prod(lrfShape[2:])/np.prod(poolSize) )
        print "Neurons for weights of filter : ",noOutNeuron   #4*(5*5*3)/(2*2)
        
        self.w = theano.shared(np.asarray(np.random.normal(loc=0,
                                                           scale=np.sqrt(1.0/noOutNeuron),
                                                           size=lrfShape),
                                          dtype=theano.config.floatX),
                               borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(loc=0,
                                                           scale=1.0,
                                                           size=(lrfShape[0],)),
                                          dtype=theano.config.floatX),
                               borrow=True)
        self.params = [self.w, self.b]


    def setInputOutput(self, inp, inpDropout, mbSize):
        
        self.inp = inp.reshape(self.imgShape)
        convOutput = conv.conv2d( input=self.inp,
                                       filters=self.w,
                                       filter_shape=self.lrfShape,
                                       image_shape=self.imgShape
                                  )
        #ds - downsize
        pooledOutput = downsample.max_pool_2d( input=convOutput,
                                               ds=self.poolSize, 
                                               ignore_border=True
                                               )
        self.out = self.activation( pooledOutput + self.b.dimshuffle('x',0,'x','x') )
        self.outDropout = self.out     #nodropout convlayer
    

class FullyConnectedLayer(object):

    def __init__(self, nInp, nOut, activation=sigmoid, pDropout=0.0):
        self.nInp = nInp
        self.nOut = nOut
        self.activation = activation
        self.pDropout = pDropout

        self.w = theano.shared( np.asarray( np.random.normal(loc=0.0,
                                                             scale =np.sqrt(1.0/nOut),
                                                             size=(nInp,nOut)
                                                             ),
                                            dtype= theano.config.floatX
                                            ),
                                name = 'w',
                                borrow = True
                                )
        self.b = theano.shared( np.asarray(np.random.normal(loc=0.0,
                                                            scale=1.0,
                                                            size=(nOut,)
                                                            ),
                                            dtype = theano.config.floatX
                                            ),
                                name = 'b',
                                borrow = True
                                )
        self.params = [self.w, self.b]


    def setInputOutput(self, inp, inpDropout, mbSize):
        self.inp = inp.reshape(( mbSize, self.nInp ))
        self.out = self.activation( (1.0-self.pDropout)*T.dot(self.inp,self.w)+self.b )
        self.yOut = T.argmax( self.out,axis=1 )

        self.inpDropout = dropoutLayer( inpDropout.reshape(( mbSize, self.nInp )),
                                        self.pDropout )
        self.outDropout = self.activation( T.dot(self.inpDropout,self.w)+self.b )

        
    def accuracy(self, y):
        return  T.mean(T.eq(self.yOut,y))


class SoftmaxLayer(object):

    def __init__(self, nInp, nOut, pDropout=0.0):
        self.nInp = nInp
        self.nOut = nOut
        self.pDropout = pDropout
        self.w = theano.shared(np.zeros((nInp,nOut),
                                        dtype = theano.config.floatX
                                        ),
                               name = 'w',
                               borrow = True
                               )
        self.b = theano.shared(np.zeros((nOut,),
                                        dtype = theano.config.floatX
                                        ),
                               name = 'b',
                               borrow = True
                               )
        self.params = [self.w, self.b]


    def setInputOutput(self, inp, inpDropout, mbSize):
        self.inp = inp.reshape((mbSize, self.nInp))
        self.out = softmax( (1-self.pDropout)*T.dot(self.inp,self.w) + self.b )
        self.yOut = T.argmax(self.out, axis=1)

        self.inpDropout = dropoutLayer( inpDropout.reshape((mbSize, self.nInp)),
                                         self.pDropout )
        self.outDropout = softmax( T.dot(self.inpDropout,self.w)+self.b)


    def cost(self, net):
        ''' Log-likelihood cost'''
        return -T.mean( T.log(self.outDropout)[T.arange(net.y.shape[0]), net.y] )

    
    def accuracy(self, y):
        return  T.mean(T.eq(self.yOut,y))

    
    
if __name__ == "__main__" :

    print "Main Controller : ",theano.config.device
    #------------------Layers Params Initialization-----------------
    noOfClasses = 10
    #Convolutional & Pooling params
    mbSize = 50
    inpChannel = 3 #RGB
    imgDimension = (32,32)
    noFilters1 = 40
    noFilters2 = 40
    fiterDimension = (5,5)
    poolDimension = (2,2)
    strideLength = 1
    #FullyConnected Layer Params
    # (I - F)/S + 1 
    cLayerOutNeuron = ((imgDimension[0]-fiterDimension[0])/strideLength+1,
                       (imgDimension[1]-fiterDimension[1])/strideLength+1
                       )
    print "CONV O/P DIM : ",cLayerOutNeuron
    #fLayerInpNeuron =  noFilters*np.prod(cLayerOutNeuron)/np.prod(poolDimension)             
    #print "Output Neurons after Convolution & Pooling : ",fLayerInpNeuron
    fLayerOutNeuron = 200 #No of Hidden neurons
    #Softmax Layer parameters
    sLayerInpNeuron = fLayerOutNeuron
    sLayerOutNeuron = noOfClasses
    print "---------------Layers Params Initialization Done---------------"
    
    #Setting the Convolutional and Pooling Layer
    print "----------Setting the Convolutional and Pooling Layer 1----------"
    cpLayer1 = ConvPoolLayer(imgShape=(mbSize,inpChannel,imgDimension[0],imgDimension[1]),
                            lrfShape=(noFilters1,inpChannel,fiterDimension[0],fiterDimension[1]),
                            poolSize=poolDimension,
                            activation=ReLU
                            )
    
    print "----------Setting the Convolutional and Pooling Layer 2----------"
    cpLayer2 = ConvPoolLayer(imgShape=(mbSize,noFilters1,14,14),
                            lrfShape=(noFilters2,noFilters1,fiterDimension[0],fiterDimension[1]),
                            poolSize=poolDimension,
                            activation=ReLU
                            )
    
    #Setting the Fully Connected Layer
    print "----------Setting the Fully Connected Layer----------"
    fLayer = FullyConnectedLayer(nInp=noFilters2*5*5, nOut=fLayerOutNeuron,
                            activation=ReLU)

    #Setting the Softmax Layer
    print "----------Setting the Softmax Layer----------"
    smLayer = SoftmaxLayer(nInp=sLayerInpNeuron, nOut=sLayerOutNeuron)

    netArch = [cpLayer1, cpLayer2, fLayer, smLayer]
    n = Network(netArch, mbSize)
    
    print "---------------Data Loading---------------"
    trainData, validData, testData = cifarLoader.loadNormData()

    #param set
    epochs = 100
    learnRate = 0.01
    lmbda = 10.0

    startTime = time.time()     
    n.mbSGD(trainData, validData, testData, epochs, mbSize, learnRate, lmbda)
    endTime = time.time()

    print "Epochs : ",epochs
    print "Learn rate : ",learnRate
    print "lambda : ",lmbda
    print "Time taken : ",round(endTime-startTime)
    print "Feature Maps L1 : ", noFilters1
    print "Feature Maps L2 : ", noFilters2
    print "Hidden L1 : ",fLayerOutNeuron

        
