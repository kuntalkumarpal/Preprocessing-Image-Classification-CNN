# Preprocessing-Image-Classification-CNN

The Implementation of the Paper [Preprocessing for image classification by convolutional neural networks](https://ieeexplore.ieee.org/document/7808140?reload=true)

### Requirements :
* [Theano](http://deeplearning.net/software/theano/install.html)
* Python 2.7
* Numpy
* cPickle
* NVIDIA drivers (If using GPU) 
* The training and testing is done on GPU (GeForce 820M) with python 2.7 and theano with cuda compilation tools (release 5.5, V5.5.0) on a machine having 8GB RAM and Intel Core i3 processor

### Folder Structure :
/datasets/ - This will have the [cifar10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html) which can be downloaded and plugged in.


### Files :
**Architectures & Training**
* CNN1.py - Referred to as Convolutional Neural Network 1. It has 1 Convolutional-Pooling layer followed by 1 Fully-connected layer followed by final softmax layer. Also Sigmoid activation have been used
* CNN2.py - Referred to as Convolutional Neural Network 2. It has 1 Convolutional-Pooling layer followed by 1 Fully-connected layer followed by final softmax layer. Also ReLU activation have been used
* CNN3.py - Referred to as Convolutional Neural Network 3. It has 2 Convolutional-Pooling layer followed by 1 Fully-connected layer followed by final softmax layer. Also ReLU activation have been used.

**Preprocessing**
* cifar_loader_raw.py - No preprocessing done to the raw data. It also acts as loader to the above architectures
* cifar_loader_v2.py - Mean normalization and Loader
* cifar_loader_ZCA_v2.py - ZCA normalization. This takes as input raw data and then produces output file ZCANormalized.pkl. Hence a loader is needed to load the ZCA normalized data.
* loader_centerd_v5.py - Loader for the ZCA normalized file
* Preprocessing standardization - TO BE UPLOADED SOON

### Parameters :
* Are hardcoded into the main program of the architecture. There is no separate file.
* By changing the loader files in the architecture and importing them into the corresponding files different preprocessing can be used.

### Note:
* The code have various hardcoding.
* Parameter setup is present in the architecture files in main.
* The code was written long ago and was not at all maintained since then. After two years its being arranged from the bits and pieces found. You may use parts of logic from the code if it suits your work.
* The code is not well structured and designed in some areas.
* The architectures were manually written to gather better understanding of neural networks. The architectures are influenced from the [excellent tutorials of Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
* No separate training, validations and testing modules all are done sequentially together in same function
* The naming of the files are inconsistent.
* The preprocessing techniques do no involve any theano code. They are written in simple python 2.7. But needs to be loaded using Theano.
