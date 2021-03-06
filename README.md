# face-image-classification
An implementation of python and tensorflow of a facial recognition system using the AT&T Laboratories Cambridge 'Database of Faces'.

Facial images Classification
============================

An implementation of a convolutional neural network image classification 
algorithm. 40 people’s faces are used to train and test this classifier. 
The best test accuracy achieved so far as been 96.25% when the data is shared 
80% training and 20% testing. Ratio can be changed on first line of code.


Instructions
============
The 'Database of Faces' from the AT&T Laboratories Cambridge was
used in this implementation.

Download the dataset from; 

https://drive.google.com/open?id=1TBP5pZmuVgUXO3JuFFDtu293xH2OtiNL

After downloading save the directory named 'faces', containing the 40 classes, 
into the same directory containing the script "facial_recognition.py" .

To start the execution, run the script .... facial_recognition.py


Dependencies
============

Implemented using python3 and tensorflow with dependencies;

>> tensorflow

>> pandas

>> numpy

>> cv2

Model
=====
>> Input image size = 224 x 224

>> Covolutional Layer

-size = 11 x 11

-strides = 4 x 4

-activation function =  Relu


>> Pooling layer

-size = 2 x 2

-strides = 3 x 3


>> Covolutional Layer

-size = 5 x 5

-strides = 4 x 4

-activation function =  Relu

>> Pooling layer

-size = 2 x 2

-strides = 3 x 3

>> Covolutional Layer

-size = 3 x 3

-strides = 4 x 4

-activation function =  Relu

>> Covolutional Layer

-size = 3 x 3

-strides = 4 x 4

-activation function =  Relu

>> Covolutional Layer

-size = 3 x 3

-strides = 4 x 4

-activation function =  Relu

>> Fully Connected layer

-size 1024
-activation function =  Relu

>> Output layer

-size = 40

Training
========

8 out of 10 images of each of the different 40 classes were used for training and validation. 2 out of 10 of each of the faces were used for testing.

>> Number of batches = 32

>> Loss function: cross entropy

>> Optimising algorithm: momentum optimizer (learning rate = 0.001, momentum = 0.9)

>> Total number of epochs: 150



Results
=======

The model correctly classified 77 out of 80 samples in the test set. 

