from keras import backend as K
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from model_const_hw import Network
import time
import tensorflow as tf
import keras

def replace_no(labels):
    # even number -> 0, odd number -> 1
    labels[labels%2 == 0] = 0
    labels[labels%2 == 1] = 1 
    return labels

def replace_5(labels):
    # number < 5 -> 0, number >= 5 -> 1
    labels[labels < 5] = 0
    labels[labels >= 5] = 1 
    return labels

def plot_epochs(hisotry):

    epochs = range(len(history.history['loss']))
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('mean squared error')
    plt.plot( epochs , history.history['val_loss'] ,label = 'val_loss')
    plt.plot( epochs , history.history['loss'] ,label = 'train_loss')
    plt.ylim([0,0.15])
    plt.grid(which='major',color='black',linestyle='-')
    #plt.grid(which='minor',color='red',linestyle='-')
    plt.legend()
    plt.savefig('epoch_eo_class_5.png')
    #plt.show()

if __name__ == "__main__":

    start = time.time()
    (train_images, train_labels), (test_images, test_labels) = np.array(mnist.load_data())
    train_labels = replace_5(train_labels)
    test_labels = replace_5(test_labels)
    train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2],1)
    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2],1)
    #train_images = train_images.astype('float32')
    #test_images = test_images.astype('float32')
    #train_images /= 255
    #test_images /= 255
    
    train_labels = keras.utils.to_categorical(train_labels,2)
    test_labels = keras.utils.to_categorical(test_labels, 2)
    
    K.set_image_data_format("channels_last")
    EPOCHS = 1
    channels = 1
    BATCH = 50
    
    #get_model
    height = train_images.shape[1]
    width = train_images.shape[2]
    model_ins = Network(channels,width,height)
    network = model_ins.get_model()
    
    #optimizer = tf.train.RMSPropOptimizer(0.000001)
    optimizer = tf.train.AdamOptimizer(0.00005)
    
    network.compile(loss="binary_crossentropy", 
                    optimizer=optimizer, 
                    metrics=["acc"])
    
    history = network.fit(train_images,
                          train_labels, 
                          epochs = EPOCHS, 
                          verbose = 1,
                          batch_size = BATCH,
                          validation_data = (test_images, test_labels)
    )

    network.save('ts_5.h5' , include_optimizer = False)
    plot_epochs(history)

    pred = network.predict(test_images[:10])
    pred[pred < 0.5] = 0
    pred[pred > 0.5] = 1

    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2])
    for i,img in enumerate(test_images[:10]):
        plt.figure()
        plt.title(pred[i,0])
        plt.imshow(img,cmap = 'gray')
        plt.savefig('5_images/' + str(i) + '.jpg')
        plt.close()
        
    end = time.time()
    print("time : {}[m]".format((end - start)/60))
        
