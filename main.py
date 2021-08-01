import cv2
import scipy
import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import h5py
import time 
import re 
import os 
import random
import tensorflow.compat.v1 as tf
from CNN import * 
import math
from tensorflow.python.framework import ops
tf.disable_v2_behavior()
initializer = tf.truncated_normal_initializer(stddev=0.1)

X_train_orig=[]
Y_train_orig = []
X_test_orig=[]
Y_test_orig=[]
classes= []



def set_up_data_filelist():

    AllObj = {}
    SharkFileNameList = []   
    SnakeFileNameList = []
    DuckFileNameList = [] 
    path = './Image'
    

    for filename in os.listdir(path):
        if 'Snake' in filename:
            SnakeFileNameList.append(path+'/'+filename)
        elif 'Shark' in filename:
            SharkFileNameList.append(path+'/'+filename)
        elif 'Duck' in filename:
            DuckFileNameList.append(path+'/'+filename)
    AllObj['1'] = SharkFileNameList
    AllObj['2'] = SnakeFileNameList
    AllObj['3'] = DuckFileNameList

    return AllObj
  
def separate_data(NameList):
    
    testing_and_training_ratio = 0.2
    size = len(NameList)
    testing_data_size = int(size*testing_and_training_ratio)
    training_data_size =  size - testing_data_size
    random.shuffle(NameList)

    return NameList[:testing_data_size], NameList[testing_data_size:], testing_data_size, training_data_size


def transform_to_image(NameList):
    for index in range(0,len(NameList)):
        image = cv2.imread(NameList[index])

        NameList[index] = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    return NameList


def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
    ### START CODE HERE ###
    
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name='C')
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    ### END CODE HERE ###
    
    return one_hot


if __name__ == '__main__':

    Obj =  set_up_data_filelist()

    for Listkey, List in Obj.items():
        Testing, Training, testing_size , training_size= separate_data(List)
        X_train_orig.extend(Training)
        Y_train_orig.extend(training_size*[int(Listkey),])
        X_test_orig.extend(Testing)
        Y_test_orig.extend(testing_size*[int(Listkey),])
    

    X_train = np.array(transform_to_image(X_train_orig))
    X_test = np.array(transform_to_image(X_test_orig))
    Y_train = one_hot_matrix(Y_train_orig, 4).T
    Y_test = one_hot_matrix(Y_test_orig, 4).T
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))

    _, _, parameters = model(X_train, Y_train, X_test, Y_test,learning_rate=0.009,
          num_epochs=500, minibatch_size=10, print_cost=True)




    ######Print diagram 
    # index = 6
    # cv2.imshow('Result',X_train_orig[index])
    # print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

    # index = 7
    # image = cv2.imread(X_train_orig[index])
    # image = image/255.
    # print(type(image))
    # print(np.array(X_train_orig))

    # print(Y_train_orig[index])
    # image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    # cv2.imshow('Result',image)
    # plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # my_image = 'Duck_0964.jpeg'
    # fname = './Image/'+my_image
    # print(fname)
    # image = cv2.imread(fname)
    # print(image)
    # image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    # X_train_orig.append(image)
    # print(image.shape[1])
    # cv2.imshow('Result',X_train_orig[0])
    # plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
