# face-image-classification
An implementation of python and tensorflow of a facial recognition system using the AT&T Laboratories Cambridge 'Database of Faces'.

Facial images Classification
============================

An implementation of a convolutional neural network image classification 
algorithm. 40 people’s faces are used to train and test this classifier. 
The best test accuracy achieved so far as been 96.25% when the data is shared 
80% training and 20% testing. Ratio can be changed at first line of code.


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
