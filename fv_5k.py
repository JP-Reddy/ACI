
# coding: utf-8

# In[14]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten, RepeatVector, Reshape, concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.engine.input_layer import Input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab, gray2rgb
from skimage.io import imsave
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import numpy as np
import os
import random
import tensorflow as tf
from PIL import Image
import glob
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from keras.callbacks import ModelCheckpoint


# In[15]:


# Get images
X = []
for filename in os.listdir('dataset/images/train/'):
    X.append(img_to_array(load_img('dataset/images/train/'+filename)))
X = np.array(X, dtype=float)
Xtrain = 1.0/255*X
Xtrain, XValid = train_test_split(Xtrain, test_size=0.15)

#Load weights
# inception = InceptionResNetV2(weights=None, include_top=True)
# inception.load_weights('dataset/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
# inception.graph = tf.get_default_graph()
embed_input = Input(shape=(5000,))


# In[16]:


#Encoder
encoder_input = Input(shape=(256, 256, 1,))
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
#Fusion
fusion_output = RepeatVector(32 * 32)(embed_input) 
fusion_output = Reshape(([32, 32,5000]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
fusion_output = Conv2D(1024, (1, 1), activation='relu', padding='same')(fusion_output)
#Decoder
decoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
# Image transformer


# In[17]:




import math
import sys
import os.path


from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.python.framework import ops
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor

slim = tf.contrib.slim
FLAGS = None


# def PreprocessImage(image,image_size, central_fraction=0.875):
#   """Load and preprocess an image.

#   Args:
#     image: a tf.string tensor with an JPEG-encoded image.
#     central_fraction: do a central crop with the specified
#       fraction of image covered.
#   Returns:
#     An ops.Tensor that produces the preprocessed image.
#   """

#   # Decode Jpeg data and convert to float.
#   #image = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32)

#   image = tf.image.central_crop(image, central_fraction=central_fraction)
#   # Make into a 4D tensor by setting a 'batch size' of 1.
#   #image = tf.expand_dims(image, [0])
#   image = tf.image.resize_bilinear(image,
#                                  [image_size, image_size],
#                                  align_corners=False)

#   # Center the image about 128.0 (which is done during training) and normalize.
#   image = tf.multiply(image, 1.0/127.5)
#   return tf.subtract(image, 1.0)


# def LoadLabelMaps(num_classes, labelmap_path, dict_path):
#   """Load index->mid and mid->display name maps.

#   Args:
#     labelmap_path: path to the file with the list of mids, describing predictions.
#     dict_path: path to the dict.csv that translates from mids to display names.
#   Returns:
#     labelmap: an index to mid list
#     label_dict: mid to display name dictionary
#   """
#   labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path).readlines()]
#   if len(labelmap) != num_classes:
#     tf.logging.fatal(
#         "Label map loaded from {} contains {} lines while the number of classes is {}".format(
#             labelmap_path, len(labelmap), num_classes))
#     sys.exit(1)

#   label_dict = {}
#   for line in tf.gfile.GFile(dict_path).readlines():
#     words = [word.strip(' "\n') for word in line.split(',', 1)]
#     label_dict[words[0]] = words[1]

#   return labelmap, label_dict

checkpoint='data/2016_08/model.ckpt'
labelmap='data/2016_08/labelmap.txt'
dict='dict.csv'
image_size=299
num_classes=6012
n=10

g = tf.Graph()
with g.as_default():
    input_image = tf.placeholder(tf.float32,shape=[None,299,299,3])
    #processed_image = PreprocessImage(input_image,image_size)
    processed_image=input_image

    with slim.arg_scope(inception.inception_v3_arg_scope()):
      logits, end_points = inception.inception_v3(
          processed_image, num_classes=num_classes, is_training=False)

    predictions = end_points['multi_predictions'] = tf.nn.sigmoid(
        logits, name='multi_predictions')
    saver = tf_saver.Saver()
    sess = tf.Session()
    saver.restore(sess, checkpoint)


def classify(images):
#     embedding=np.empty([len(images),1000])
    with g.as_default():
        embedding=[]
        # Run the evaluation on the image

    #     print(images_x_x)
    #     labelmap, label_dict = LoadLabelMaps(num_classes, labelmap, dict)
        for im in images:
    #         print(im.shape)
            img=np.expand_dims(im,axis=0)
            predictions_eval = np.squeeze(sess.run(predictions,
                                               {input_image: img}))

    #         print(type(predictions_eval))
    #         embedding.append([predictions_eval[:1000]],axis=0)
            embedding.append(predictions_eval[0:5000])
    #         top_k = predictions_eval.argsort()[-n:][::-1]
    #         for idx in top_k:
    #       	    mid = labelmap[idx]
    #       	    display_name = label_dict.get(mid, 'unknown')
    #       	    score = predictions_eval[idx]
    #       	    print('{}: {} - {} (score = {:.2f})'.format(idx, mid, display_name, score))
    #     print(embedding.shape)
    return embedding


# In[18]:


#Create embedding
def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    embed=classify(grayscaled_rgb_resized)
    return embed


# In[19]:


# datagen = ImageDataGenerator(
#         shear_range=0.4,
#         zoom_range=0.4,
#         rotation_range=40,
#         horizontal_flip=True,
#         validation_split=0.15
#         )

# #Generate training data
# batch_size = 40
# def image_a_b_gen(batch_size):
#     for batch in datagen.flow(Xtrain, batch_size=batch_size):
#         grayscaled_rgb = gray2rgb(rgb2gray(batch))
#         embed = np.array(create_inception_embedding(grayscaled_rgb))
#         print(embed.shape)
#         lab_batch = rgb2lab(batch)
#         X_batch = lab_batch[:,:,:,0]
#         X_batch = X_batch.reshape(X_batch.shape+(1,))
#         Y_batch = lab_batch[:,:,:,1:] / 128
#         yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)


# In[26]:


#Modified generator
datagen = ImageDataGenerator(
        shear_range=0.4,
        zoom_range=0.4,
        rotation_range=40,
        horizontal_flip=True,
        )

# train_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True, seed=13,
#                                                      class_mode='categorical', batch_size=BATCH_SIZE, subset="training")

# validation_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True, seed=13,
#                                                      class_mode='categorical', batch_size=BATCH_SIZE, subset="validation")

#Generate training data
batch_size = 40
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = np.array(create_inception_embedding(grayscaled_rgb))
#         print(embed.shape)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([np.array(X_batch), embed], np.array(Y_batch))

def validation_gen(batch_size):
    for batch in datagen.flow(XValid, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = np.array(create_inception_embedding(grayscaled_rgb))
#         print(embed.shape)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([np.array(X_batch), embed], np.array(Y_batch))


# In[34]:


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


# In[ ]:

steps_per_epoch=len(Xtrain)//batch_size
val_spe=len(XValid)//60
#Train model      
tensorboard = TensorBoard(log_dir="output/")
model.compile(optimizer='adam', loss='mse')
filepath='saved_model/model.ckpt.{epoch:04d}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,period=1)
callbacks_list = [checkpoint,tensorboard]
model.fit_generator(image_a_b_gen(batch_size), validation_data=validation_gen(batch_size),validation_steps=50,callbacks=callbacks_list, epochs=5,steps_per_epoch=steps_per_epoch)


# In[22]:


#Make a prediction on the unseen images
color_me = []
for filename in os.listdir('dataset/images/test/'):
    color_me.append(img_to_array(load_img('dataset/images/test/'+filename)))
color_me = np.array(color_me, dtype=float)
color_me = 1.0/255*color_me
color_me = gray2rgb(rgb2gray(color_me))
color_me_embed = create_inception_embedding(color_me)
color_me = rgb2lab(color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))


# In[25]:


# Test model
output = model.predict([np.array(color_me), np.array(color_me_embed)])
output = output * 128
# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("result/img_"+str(i)+".png", lab2rgb(cur))

os.system("shutdown")
