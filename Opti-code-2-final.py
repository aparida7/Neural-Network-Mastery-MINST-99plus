#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import anvil.media
import anvil.server
from PIL import Image


# In[2]:


def preprocess_uploaded_image(image_array, n, m, block_size):

        # img_array is already 28x28 as required
        # Prepare the image in the same way as the training and test data were prepared
        raveled_image = np.zeros((1, 7*7, 16))  # Adding batch dimension
        ind = 0
        image_array = np.array(image_array[0].reshape(28, 28))
        for row in range(n):
            for col in range(m):
                raveled_image[0, ind, :] = image_array[(row*4):((row+1)*4), (col*4):((col+1)*4)].ravel()
                ind += 1
        # Prepare positional encoding
        pos_feed = np.array([list(range(n*m))])
        return raveled_image, pos_feed

def predict_uploaded_image(image_array, model, n, m, block_size):
        raveled_image, pos_feed = preprocess_uploaded_image(image_array, 7, 7, 16)
        predicted_label = model.predict([raveled_image, pos_feed])
        predicted_class = np.argmax(predicted_label)
        return predicted_class


# In[3]:


# Connect to Anvil
anvil.server.connect("server_M2P2TUKEV6EXKP5QPW6ZB22O-EPY7DVS5RKWN7ZMG")

# Function that will predict the labels and images
@anvil.server.callable
def predictions(file):
    # Load the model weights and create the model
    model = tf.keras.models.load_model('CNN_mnist.h5')
    #model2 = tf.keras.models.load_model('/home/bitnami/opti_code/my_transformer_model.h5')
    #model = tf.keras.models.load_model('CNN_mnist.h5')
    
    # Load the TF model
    class ClassToken(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(ClassToken, self).__init__(**kwargs)  # Pass the keyword arguments to the superclass

        def build(self, input_shape):
            w_init = tf.random_normal_initializer()
            self.w = tf.Variable(
                initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype='float32'),
                trainable=True,
            )

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            hidden_dim = self.w.shape[-1]
            cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
            cls = tf.cast(cls, dtype=inputs.dtype)
            return cls
    
    # Register and load the model containing the custom layer
    custom_objects = {'ClassToken': ClassToken}
    model2 = load_model('Transformer_model.h5', custom_objects=custom_objects)
    
    # Validate the file to check the CSV file and get the image array
    flag,img_array = preprocess_image(file)
    
    if(flag == 0):
        return flag,None,None,None
    
    else:
        # Make predictions
        predictions = model.predict(img_array)
        predictions_2 = predict_uploaded_image(img_array, model2, 7, 7, 16)

        # Get the predicted label (class with highest probability)
        predicted_label = tf.argmax(predictions, axis=1).numpy()[0]
        predicted_label_2 = predictions_2

        # Convert the image array to an image object
        first_image_data = img_array[0].reshape(28, 28) * 255  # Rescale back to 0-255 if normalized
        img = Image.fromarray(first_image_data.astype(np.uint8), 'L')
        #print(img)
    
        flag = 1
        with anvil.media.TempFile() as file:
            img.save(file, format="PNG")
            #print(predicted_label)
            return flag,predicted_label,anvil.media.from_file(file, "image/png"),predicted_label_2


# In[4]:


def preprocess_image(csv_file_path):
    with anvil.media.TempFile(csv_file_path) as filename:# Read the CSV file
        df = pd.read_csv(filename, header=None)
    flag = 0
    
    # Verify shape of the dataframe
    if df.shape != (28, 28):
        return flag,None
        
    else:
        # Check intensity values and scale if necessary
        max_intensity = df.max().max()
        if max_intensity > 1:
            df = df / 255.0

        # Convert dataframe to numpy array
        image_array = df.values

        # Convert to Keras image array format
        image_array = np.expand_dims(image_array, axis=-1)
        image_array = np.expand_dims(image_array, axis=0)
        flag = 1

        return flag,image_array


# In[ ]:


anvil.server.wait_forever()


# In[ ]:




