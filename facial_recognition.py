import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import glob
import csv
import cv2




#train algorithm

trno = 8    # this values is the percentage of training data (trno/10)*100 ie default is 80%
            #change this value to experiment with different training/test ratios.
classes = 40


def data(fls,trno):

    trgotc = []
    trlbls = []
    tegotc = []
    telbls = []
    
    for n in range(len(fls)):
      
        data = pd.read_csv(fls[n],header=None)
        d = data.as_matrix()
    
        for cloe in range(trno):
            a = fls[n]
            b = a.strip('s')
            c = b.strip('.csv')
            trlbls.append(int(c)-1)   

        for cloe in range(10-trno):
            telbls.append(int(c)-1)   


        if (n == 0):
            trgotc = d[:trno]
            tegotc = d[trno:]
        else:
            trgotc = np.vstack([trgotc,d[:trno]])
            tegotc = np.vstack([tegotc,d[trno:]])


    trgotc = np.divide(trgotc.astype(np.float32),255)
    trlbls = np.array(trlbls)

    tegotc = np.divide(tegotc.astype(np.float32),255)
    telbls = np.array(telbls)
    
    return trgotc,trlbls,tegotc,telbls



def training_set(ep,fls,nrows):

    gotc = []
    lbls = []
    
    for n in range(len(fls)):
      
        data = pd.read_csv(fls[n], skiprows = ep, nrows = nrows,header=None)
        d = data.as_matrix()
        getfl = fls[n]
        label = 1
        
        for cloe in range(nrows):
            lbls.append(label)

        if (n == 0):
            gotc = d
        else:
            gotc = np.vstack([gotc,d])



    gotc = gotc.astype(np.float32)
    lbls = np.array(lbls)
    
    return gotc,lbls

def func_get_data(fname):

    gotc = []
    with open(fname,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            #print(row)
            for k in range(len(row)):

                row[k] = float(row[k])

            gotc.append(row)

    gotc = np.array(gotc).astype(np.float32)

    return gotc



def func_get_onehot_labels(lb,n):

    oneh = np.zeros(n)  #number of zeros equal to number of classes

    oneh[lb] = 1          
    
    return oneh


def next_batch(lower,separation,det):
    
    higher = 0
    
    if(det == 0):
        higher = lower + separation
        
    else:
        lower += separation
        higher = lower + separation

    return [lower,higher]





#preprocess dataset

xd,yd = [224,224]
print('\nProcessing dataset \n')
for i in range(40):

    img = glob.glob('faces/s'+str(i+1)+'/*.pgm') #get all images
    
    for j in range(len(img)):
        
        one1 = cv2.imread(img[j],0)


        one1 = cv2.resize(one1,None,fx=(xd/one1.shape[1]), fy=(yd/one1.shape[0]), interpolation = cv2.INTER_AREA) # reshape image to xd by yd pixels

        imgcsv = one1.reshape(1,xd*yd)
        if(j == 0):
            grid = imgcsv
        else:
            grid = np.vstack((grid,imgcsv))
            

    wr = grid
    
    #save to csv file

    with open('s'+str(i+1)+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar=' ')

        for q in range(len(wr)):

            spamwriter.writerow(wr[q])

            




classes = 40


faces = glob.glob('*.csv')

trfeat,trlbls,tefeat,telbls = data(faces,trno)

trlabels = []
telabels = []


for i in trlbls:  #create one ot labels for output

    trlabels.append(func_get_onehot_labels(i,classes))

for i in telbls:  #create one hot labels for output

    telabels.append(func_get_onehot_labels(i,classes))


trlabels = np.array(trlabels)
telabels = np.array(telabels)

print(trfeat.shape)
print(tefeat.shape)
print(trlabels.shape)
print(telabels.shape)



batch_size = 32
imgw = 224 #img width
imgh = 224 #img height
featno = imgw*imgh
lblno = classes
chan = 1
hl1 = 4096



print('Started Training....')

def deepnn(x):
    
    x_image = tf.reshape(x, [-1,imgw,imgh, chan])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([11, 11, 1, 96])
    b_conv1 = bias_variable([96])
    conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,4,4,1], padding='SAME') + b_conv1
    nconv1 = tf.nn.local_response_normalization(conv1)
    h_conv1 = tf.nn.relu(nconv1)

    # Pooling layer - downsamples by 2X.
    pool_c1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],strides=[1, 3, 3, 1], padding='SAME')


    # 2nd convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv2 = weight_variable([5, 5, 96, 256])
    b_conv2 = bias_variable([256])
    conv2 = tf.nn.conv2d(pool_c1, W_conv2, strides=[1,4,4,1], padding='SAME') + b_conv2
    nconv2 = tf.nn.local_response_normalization(conv2)
    h_conv2 = tf.nn.relu(nconv2)

    # Pooling layer - downsamples by 2X.
    pool_c2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],strides=[1, 3, 3, 1], padding='SAME')

    # 3rd convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv3 = weight_variable([3, 3, 256, 384])
    b_conv3 = bias_variable([384])
    conv3 = tf.nn.conv2d(pool_c2, W_conv3, strides=[1,4,4,1], padding='SAME') + b_conv3
    h_conv3 = tf.nn.relu(conv3)

    # 4th convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv4 = weight_variable([3, 3, 384, 384])
    b_conv4 = bias_variable([384])
    conv4 = tf.nn.conv2d(h_conv3, W_conv4, strides=[1,4,4,1], padding='SAME') + b_conv4
    h_conv4 = tf.nn.relu(conv4)

    # 5th convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv5 = weight_variable([3, 3, 384, 256])
    b_conv5 = bias_variable([256])
    conv5 = tf.nn.conv2d(h_conv4, W_conv5, strides=[1,4,4,1], padding='SAME') + b_conv5
    nconv5 = tf.nn.local_response_normalization(conv5)
    h_conv5 = tf.nn.relu(nconv5)

    #fully connected layer
    W_fc1 = weight_variable([128*int(imgw/(4*4*3*3*4*4) * imgh/(4*4*3*3*4*4) * 256), hl1])     #4*4 from two 2x2 pooling layers
    b_fc1 = bias_variable([hl1])

    h_flat1 = tf.reshape(h_conv5, [-1, 128*int(imgw/(4*4*3*3*4*4) * imgh/(4*4*3*3*4*4) * 256)])
    h_fc11 = tf.nn.relu(tf.matmul(h_flat1, W_fc1)+ b_fc1)


    keep_prob = tf.placeholder(tf.float32)
    h_fc11 = tf.nn.dropout(h_fc11, keep_prob)

    # Map the 1024 features to number of classes, one for each digit
    W_fco = weight_variable([hl1, classes])
    b_fco = bias_variable([classes])
    output = tf.matmul(h_fc11, W_fco) + b_fco

  
    return  output, keep_prob


def weight_variable(shape):

  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):

  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):

  # Create the model
  x = tf.placeholder(tf.float32, [None, int(imgw*imgh)])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, classes])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

    

  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  

  train_step = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9).minimize(cost)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  epoch_loss = 0
  i = 0
  saver = tf.train.Saver()

  with tf.Session() as sess:
      
    sess.run(tf.global_variables_initializer())
    folder = os.path.dirname('latest/')
    if not os.path.exists(folder):
        os.makedirs(folder)

    while(i<150):
  

      u = 0 #the starting index of each batch ie first sample
      v = 0
      epoch_loss = 0
      for iteration in range(int(len(trfeat)/batch_size)):
      
          u,v = next_batch(u,batch_size,iteration)  #returns indices of batch boundaries            
          _,c = sess.run([train_step,cost],feed_dict = {x: trfeat[u:v], y_: trlabels[u:v],keep_prob: 0.5})
          epoch_loss +=c

          save_path = saver.save(sess,'latest/model.ckpt')

          
      i += 1
      print('Epoch: ',i)  
      print('Epoch loss: ',epoch_loss)




    #test algorithm
    print('\n\nTesting algorithm')  
    total = 0
    acc = 0
       
    for n in range(len(tefeat)):
                
                pred = y_conv.eval(feed_dict = {x:tefeat[n:n+1], y_:telabels[n:n+1],keep_prob: 1.0})

                bpred = np.argmax(pred)
                bt = np.argmax(telabels[n])

                #print(bpred,bt)
                total += 1
                if(bpred == bt):
                    acc += 1

  
    print('\nTesting accuracy ',(acc/total)*100)
      
      
  tf.reset_default_graph()
  sess = tf.InteractiveSession()           #these routines clear the tensorflow graph. Necessarry for clearing memory 
  tf.contrib.keras.backend.clear_session()
  

if __name__ == "__main__":
  tf.app.run()


